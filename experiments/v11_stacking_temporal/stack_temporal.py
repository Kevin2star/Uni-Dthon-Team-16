"""센서 모델에 이미지 정보를 '요약해서' 넣는 두 방법 비교.

① 센서만                 : 베이스라인 피처에서 img_* 컬럼 제거 + LightGBM
② 센서 + 이미지 확률 6개 : ConvNeXt-Tiny 임베딩 → PCA 64 → LightGBM 이미지 모델의 6단계 확률 (스태킹)
③ 센서 + 임베딩 시간 변화 : 같은 실험의 30·60초 전 프레임과의 임베딩 차이 / 실제 경과 시간
④ ②+③ (참고)
(참고) 센서+열퍼짐 = 지금까지 최고 모델

누수 방지 (중첩 교차검증)
- 바깥 GroupKFold(5)의 평가 실험은 이미지 모델·최종 모델 학습 모두에서 제외
- 바깥 학습 실험의 이미지 확률 = 그 안에서 다시 GroupKFold(5)로 만든 OOF
- 평가 실험의 이미지 확률 = 바깥 학습 실험 전체로 학습한 이미지 모델
- PCA도 매번 그 시점의 학습 쪽 데이터로만 fit
Validation: Train 안 OOF 확률로 최종 모델 학습, Validation 확률은 Train 전체로 학습한 이미지 모델

시간 변화는 같은 experiment_id 안에서 relative_time_s <= t - Δ 인 마지막 관측만 사용 (미래 사용 없음).
사용법: python stack_temporal.py   (약 10~15분)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'uni_d_baseline_v7')]
import model as BASE  # noqa: E402
import model_image_lgbm as MI  # noqa: E402
from lgbm_common import build_lgbm  # noqa: E402

DATA = Path('/home/ubuntu/data/battery_data_light')
LABELS = [1, 2, 3, 4, 5, 6]
SEED = 42
ARCH = 'convnext_tiny'
LAGS = [30.0, 60.0]
IMG_PCA, DIFF_PCA = 64, 16


def macro(y, p):
    return f1_score(y, p, labels=LABELS, average='macro', zero_division=0)


def groupk(groups):
    return list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))


# ---------------------------------------------------------------- 데이터
def load(split):
    f = pd.read_csv(DATA / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    img_b = pd.read_csv(HERE / 'uni_d_baseline_v7' / 'work' / f'img_features_{split}.csv', dtype={'sample_id': 'string'},
                        float_precision='round_trip').set_index('sample_id')
    Xb = BASE.build_features(f, img_b)
    sensor = Xb[[c for c in Xb.columns if not c.startswith('img_')]].values
    img_t = pd.read_csv(HERE / 'work_model_image' / f'img_{split}.csv', dtype={'sample_id': 'string'},
                        float_precision='round_trip').set_index('sample_id')
    best = MI.build_features(f, img_t).values
    E = np.load(HERE / 'work_imgrep' / f'cnn_{ARCH}_{split}.npy')
    return f, sensor, best, E


def lag_index(frame, lag_s):
    """행마다 같은 실험에서 relative_time_s <= t-lag 인 마지막 행의 위치(없으면 -1)."""
    out = np.full(len(frame), -1)
    t_all = frame.relative_time_s.to_numpy(float)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        t = t_all[idx]
        j = np.searchsorted(t, t - lag_s, side='right') - 1
        ok = j >= 0
        out[idx[ok]] = idx[j[ok]]
    return out


def temporal_raw(frame, E):
    """시간 변화 기본량: Δ마다 (변화 속도 노름, 코사인 거리), 그리고 30초 차이 벡터/경과시간 (PCA 전)."""
    t = frame.relative_time_s.to_numpy(float)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    cols, diff30 = [], None
    for lag in LAGS:
        j = lag_index(frame, lag)
        ok = j >= 0
        dt = np.where(ok, t - t[np.maximum(j, 0)], np.nan)
        d = np.where(ok[:, None], E - E[np.maximum(j, 0)], np.nan)
        rate = np.linalg.norm(d, axis=1) / dt
        cos = np.where(ok, 1 - (En * En[np.maximum(j, 0)]).sum(1), np.nan) / dt
        cols += [rate, cos]
        if lag == 30.0:
            diff30 = d / dt[:, None]
    return np.column_stack(cols), diff30


def diff_pca(D_fit, D_list):
    ok = ~np.isnan(D_fit).any(1)
    p = PCA(n_components=DIFF_PCA, random_state=SEED).fit(D_fit[ok])
    out = []
    for D in D_list:
        Z = np.full((len(D), DIFF_PCA), np.nan)
        m = ~np.isnan(D).any(1)
        Z[m] = p.transform(D[m])
        out.append(Z)
    return out


# ---------------------------------------------------------------- 이미지 모델 (스태킹용)
def img_model_fit_predict(E_fit, y_fit, E_pred):
    p = PCA(n_components=IMG_PCA, random_state=SEED).fit(E_fit)
    m = build_lgbm().fit(p.transform(E_fit), y_fit)
    return m.predict_proba(p.transform(E_pred))


def img_probs(E, y, groups, tr_idx, ev_idx):
    """tr_idx 안에서는 내부 GroupKFold OOF 확률, ev_idx는 tr_idx 전체로 학습한 모델의 확률."""
    oof = np.zeros((len(tr_idx), 6))
    for a, b in groupk(groups[tr_idx]):
        oof[b] = img_model_fit_predict(E[tr_idx[a]], y[tr_idx[a]], E[tr_idx[b]])
    ev = img_model_fit_predict(E[tr_idx], y[tr_idx], E[ev_idx])
    return oof, ev


# ---------------------------------------------------------------- 평가
def report(name, scores, yva, pv):
    per = f1_score(yva, pv, labels=LABELS, average=None, zero_division=0)
    print(f'{name:<30} | CV {np.mean(scores):.4f} ± {np.std(scores):.4f} | fold {" ".join(f"{s:.3f}" for s in scores)} '
          f'| Val {macro(yva, pv):.4f} | Val S1~6 {np.round(per, 2).tolist()}', flush=True)


def main():
    t0 = time.time()
    ftr, Str, Btr, Etr = load('train')
    fva, Sva, Bva, Eva = load('validation')
    ytr, yva = ftr.label.values, fva.label.values
    g = ftr.experiment_id.values
    Ttr, Dtr = temporal_raw(ftr, Etr)
    Tva, Dva = temporal_raw(fva, Eva)
    print(f'준비 {time.time() - t0:.0f}s | 센서 피처 {Str.shape[1]}, 시간변화 기본 {Ttr.shape[1]} + PCA {DIFF_PCA}', flush=True)

    outer = groupk(g)
    names = ['① 센서만', '② 센서 + 이미지 확률 6', '③ 센서 + 임베딩 시간 변화', '④ ②+③ (참고)', '(참고) 센서+열퍼짐 현재 최고']
    cv = {n: [] for n in names}
    for k, (a, b) in enumerate(outer):
        p_tr, p_ev = img_probs(Etr, ytr, g, a, b)
        z_tr, z_ev = diff_pca(Dtr[a], [Dtr[a], Dtr[b]])
        sets = {
            names[0]: (Str[a], Str[b]),
            names[1]: (np.hstack([Str[a], p_tr]), np.hstack([Str[b], p_ev])),
            names[2]: (np.hstack([Str[a], Ttr[a], z_tr]), np.hstack([Str[b], Ttr[b], z_ev])),
            names[3]: (np.hstack([Str[a], p_tr, Ttr[a], z_tr]), np.hstack([Str[b], p_ev, Ttr[b], z_ev])),
            names[4]: (Btr[a], Btr[b]),
        }
        for n, (A, B) in sets.items():
            cv[n].append(macro(ytr[b], build_lgbm().fit(A, ytr[a]).predict(B)))
        print(f'  바깥 fold {k + 1}/5 완료 ({time.time() - t0:.0f}s)', flush=True)

    # Validation
    all_idx = np.arange(len(ytr))
    p_tr = np.zeros((len(ytr), 6))
    for a, b in groupk(g):
        p_tr[b] = img_model_fit_predict(Etr[a], ytr[a], Etr[b])
    p_va = img_model_fit_predict(Etr, ytr, Eva)
    z_tr, z_va = diff_pca(Dtr, [Dtr, Dva])
    vsets = {
        names[0]: (Str, Sva),
        names[1]: (np.hstack([Str, p_tr]), np.hstack([Sva, p_va])),
        names[2]: (np.hstack([Str, Ttr, z_tr]), np.hstack([Sva, Tva, z_va])),
        names[3]: (np.hstack([Str, p_tr, Ttr, z_tr]), np.hstack([Sva, p_va, Tva, z_va])),
        names[4]: (Btr, Bva),
    }
    print(f'\n=== 결과 (바깥 GroupKFold 5 + Validation) | 이미지 = {ARCH}')
    for n, (A, B) in vsets.items():
        report(n, cv[n], yva, build_lgbm().fit(A, ytr).predict(B))
    img_only_cv = [macro(ytr[b], p_tr[b].argmax(1) + 1) for _, b in groupk(g)]
    report('(참고) 이미지 모델 단독', img_only_cv, yva, p_va.argmax(1) + 1)
    np.savez(HERE / 'work_imgrep' / f'stack_probs_{ARCH}.npz', train_oof=p_tr, validation=p_va)
    print(f'총 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
