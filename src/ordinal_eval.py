"""'경보 5개' 순서형(ordinal) 판정기 vs 6클래스 분류기 비교.

경보 k (k=2..6): "지금 k단계 이상인가?"를 이진 LightGBM으로 판정 → 단계 = 1 + (예라고 답한 경보 수)
비교:
  6클래스 LGBM                  (기준)
  경보 5개 · 기준값 0.5
  경보 5개 · 경계별 기준값 조정  (CV: 평가 fold를 뺀 나머지 fold의 OOF로 조정 / Val: Train OOF 전체로 조정)
피처는 model_image_lgbm(열 퍼짐 피처)와 같고, 같은 GroupKFold(5) 분할을 씁니다.

사용법: python ordinal_eval.py [model_image_lgbm]
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from paths import DATA_ROOT  # 데이터 위치: 환경변수 DATA_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                 # ~/yeonwoo
P_CACHE = ROOT / 'cache'           # 이미지 피처·임베딩 캐시
P_OUT = ROOT / 'outputs'           # 그림·로그·예측 결과
sys.path[:0] = [str(HERE), str(ROOT / 'uni_d_baseline_v7')]
from lgbm_common import build_lgbm  # noqa: E402

DATA = DATA_ROOT
LABELS = [1, 2, 3, 4, 5, 6]
KS = [2, 3, 4, 5, 6]
GRID = np.round(np.arange(0.10, 0.91, 0.05), 2)


def macro(y, p):
    return f1_score(y, p, labels=LABELS, average='macro', zero_division=0)


def decode(Q, th):
    """Q: n×5 경보 확률(≥2..≥6), th: 길이 5 기준값 → 단계."""
    return 1 + (Q >= np.asarray(th)).sum(1)


def tune(Q, y, passes=3):
    """경계별 기준값을 Macro F1 최대가 되도록 좌표 탐색."""
    th = np.full(5, 0.5)
    for _ in range(passes):
        for j in range(5):
            scores = []
            for g in GRID:
                t = th.copy(); t[j] = g
                scores.append(macro(y, decode(Q, t)))
            th[j] = GRID[int(np.argmax(scores))]
    return th


def fit_alarms(X, y):
    return [build_lgbm().fit(X, (y >= k).astype(int)) for k in KS]


def alarm_proba(models, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models])


def per_stage(y, p):
    return f1_score(y, p, labels=LABELS, average=None, zero_division=0)


def main(mod_name):
    M = importlib.import_module(mod_name)
    cd = P_CACHE / f"work_{getattr(M, 'FEATURE_CACHE', mod_name)}"

    def load(s):
        f = pd.read_csv(DATA / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        img = pd.read_csv(cd / f'img_{s}.csv', dtype={'sample_id': 'string'},
                          float_precision='round_trip').set_index('sample_id')
        return f, M.build_features(f, img), f.label.astype(int).values

    tr, Xtr, ytr = load('train')
    va, Xva, yva = load('validation')
    folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, groups=tr.experiment_id))

    # --- Train OOF: 6클래스 / 경보 5개
    oof_mc = np.zeros(len(ytr), int)
    oof_Q = np.zeros((len(ytr), 5))
    for a, b in folds:
        oof_mc[b] = M.build_model(None).fit(Xtr.iloc[a], ytr[a]).predict(Xtr.iloc[b])
        oof_Q[b] = alarm_proba(fit_alarms(Xtr.iloc[a], ytr[a]), Xtr.iloc[b])

    # 기준값 조정: 평가 fold를 뺀 나머지 fold의 OOF로 (정직한 CV 점수)
    oof_tuned = np.zeros(len(ytr), int)
    fold_th = []
    for i, (_, b) in enumerate(folds):
        rest = np.concatenate([folds[j][1] for j in range(5) if j != i])
        th = tune(oof_Q[rest], ytr[rest])
        fold_th.append(th)
        oof_tuned[b] = decode(oof_Q[b], th)
    th_all = tune(oof_Q, ytr)

    # --- Validation
    va_mc = M.build_model(None).fit(Xtr, ytr).predict(Xva)
    va_Q = alarm_proba(fit_alarms(Xtr, ytr), Xva)

    def fold_scores(pred):
        return [macro(ytr[b], pred[b]) for _, b in folds]

    rows = {
        '6클래스 LGBM': (oof_mc, va_mc),
        '경보 5개 · 기준값 0.5': (decode(oof_Q, [0.5] * 5), decode(va_Q, [0.5] * 5)),
        '경보 5개 · 기준값 조정': (oof_tuned, decode(va_Q, th_all)),
    }
    out = []
    for name, (po, pv) in rows.items():
        fs = fold_scores(po)
        out.append({'방식': name, 'CV 평균': np.mean(fs), 'CV 표준편차': np.std(fs),
                    'fold별': ' '.join(f'{x:.3f}' for x in fs), 'Validation': macro(yva, pv),
                    **{f'Val S{l}': v for l, v in zip(LABELS, per_stage(yva, pv))}})
    res = pd.DataFrame(out).set_index('방식')
    pd.set_option('display.width', 250)
    print(f'\n=== {mod_name} 피처 | Train GroupKFold(5) OOF + Validation')
    print(res.to_string(float_format=lambda x: f'{x:.3f}'))

    print('\nCV 단계별 F1 (OOF 전체):')
    print(pd.DataFrame({n: per_stage(ytr, po) for n, (po, _) in rows.items()}, index=[f'S{l}' for l in LABELS]).T
          .to_string(float_format=lambda x: f'{x:.3f}'))

    print('\n경계별 기준값 (≥2 ≥3 ≥4 ≥5 ≥6):')
    for i, th in enumerate(fold_th):
        print(f'  fold {i + 1} 평가용(나머지 fold로 조정): {th.tolist()}')
    print(f'  Validation용(Train OOF 전체로 조정): {th_all.tolist()}')

    for name, Q, th in [('Train OOF', oof_Q, th_all), ('Validation', va_Q, th_all)]:
        yes = Q >= th
        bad = (yes[:, 1:] & ~yes[:, :-1]).any(1).mean()
        print(f'경보 순서 어긋남 비율 ({name}, "≥k+1 예인데 ≥k 아니오"): {bad * 100:.2f}%')

    pd.DataFrame({'sample_id': list(tr.sample_id) + list(va.sample_id),
                  'split': ['train_oof'] * len(tr) + ['validation'] * len(va),
                  **{f'q_ge{k}': np.concatenate([oof_Q[:, i], va_Q[:, i]]) for i, k in enumerate(KS)},
                  'label': np.concatenate([ytr, yva])}).to_csv(P_OUT / f'ordinal_probs_{mod_name}.csv', index=False)
    print('경보 확률 보관:', P_OUT / f'ordinal_probs_{mod_name}.csv')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'model_image_lgbm')
