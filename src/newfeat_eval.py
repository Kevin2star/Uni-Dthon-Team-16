"""단계 관찰 피처 12개(thermal.STAGE_FEATURES) 평가.

1) 경계별 효과 크기: 같은 실험 안에서 k→k+1 단계로 갈 때 얼마나 일관되게 바뀌나 (기존 77개와 비교)
2) 이미지만 LightGBM: 기존(퍼짐 13 + 분포 64) vs 기존 + 새 12  (GroupKFold 5 + Validation)
3) 센서 + 이미지 확률 6개 (중첩 CV 스태킹): 직접 만든 피처 기반 이미지 모델
   (참고: 같은 코드·같은 분할에서 ConvNeXt 스태킹 = CV 0.8731, Val 0.7898 / 센서만 = CV 0.8678, Val 0.7604)

캐시: work_imgrep/stage_{split}.npy
사용법: python -u newfeat_eval.py | tee ../outputs/newfeat.log
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from paths import DATA_ROOT  # 데이터 위치: 환경변수 DATA_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                 # ~/yeonwoo
P_CACHE = ROOT / 'cache'           # 이미지 피처·임베딩 캐시
P_OUT = ROOT / 'outputs'           # 그림·로그·예측 결과
sys.path[:0] = [str(HERE), str(ROOT / 'uni_d_baseline_v7')]
import thermal as T  # noqa: E402
from firstframe_eval import img_fit_predict  # noqa: E402
from lgbm_common import build_lgbm  # noqa: E402
from stack_temporal import load as load_sensor  # noqa: E402

DATA = DATA_ROOT
CACHE = P_CACHE / 'work_imgrep'
SPLITS = ['train', 'validation', 'test']
LABELS = [1, 2, 3, 4, 5, 6]


def macro(y, p):
    return f1_score(y, p, labels=LABELS, average='macro', zero_division=0)


def groupk(groups):
    return list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))


def stage_feats(split, frame):
    f = CACHE / f'stage_{split}.npy'
    if not f.is_file():
        rows = Parallel(n_jobs=-1, batch_size=64)(
            delayed(T.stage_features)(str(DATA / split / p)) for p in frame.image_path)
        np.save(f, np.asarray(rows, np.float32))
    return np.load(f)


def shifts(X, names, y, exp):
    """경계마다 피처별 (효과 크기 중앙값, 방향 일치율)."""
    out = []
    df = pd.DataFrame(X, columns=names)
    for k in range(1, 6):
        eff = []
        for e in np.unique(exp):
            m = exp == e
            a, b = df[m & (y == k)], df[m & (y == k + 1)]
            if len(a) < 3 or len(b) < 3:
                continue
            sd = df[m].std().replace(0, np.nan)
            eff.append(((b.mean() - a.mean()) / sd).values)
        eff = np.array(eff)
        for j, n in enumerate(names):
            v = eff[:, j][~np.isnan(eff[:, j])]
            if len(v) >= 5:
                med = np.median(v)
                out.append(dict(경계=f'{k}→{k + 1}', 피처=n, 효과=med, 일치=(np.sign(v) == np.sign(med)).mean()))
    r = pd.DataFrame(out)
    r['점수'] = r.효과.abs() * r.일치
    return r


def main():
    t0 = time.time()
    fr = {s: pd.read_csv(DATA / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'}) for s in SPLITS}
    OLD, NEW = {}, {}
    for s in SPLITS:
        spread = pd.read_csv(P_CACHE / 'work_model_image' / f'img_{s}.csv', dtype={'sample_id': 'string'}) \
            .set_index('sample_id').loc[fr[s].sample_id].values
        OLD[s] = np.hstack([spread, np.load(CACHE / f'rich_{s}.npy')])
        NEW[s] = stage_feats(s, fr[s])
    print(f'피처 준비 {time.time() - t0:.0f}s | 기존 {OLD["train"].shape[1]}개, 새 {NEW["train"].shape[1]}개', flush=True)
    old_names = list(pd.read_csv(P_CACHE / 'work_model_image' / 'img_train.csv', nrows=1).columns[1:]) + T.RICH_FEATURES
    ytr, yva = fr['train'].label.values, fr['validation'].label.values
    g = fr['train'].experiment_id.values

    print('\n=== 1) 경계별: 새 피처 상위 3 vs 기존 77개 중 최고  (효과 = 실험 내 표준화 변화 중앙값, 일치 = 같은 방향 실험 비율)')
    r_new = shifts(NEW['train'], T.STAGE_FEATURES, ytr, g)
    r_old = shifts(OLD['train'], old_names, ytr, g)
    for k in range(1, 6):
        b = f'{k}→{k + 1}'
        o = r_old[r_old.경계 == b].nlargest(1, '점수').iloc[0]
        print(f'[{b}] 기존 최고: {o.피처} {o.효과:+.2f} ({o.일치 * 100:.0f}%)')
        for _, x in r_new[r_new.경계 == b].nlargest(3, '점수').iterrows():
            flag = '  ← 기존 최고보다 큼' if x.점수 > o.점수 else ''
            print(f'      새: {x.피처:<18} {x.효과:+.2f} ({x.일치 * 100:.0f}%){flag}')
    r_new.to_csv(P_OUT / 'stage_feature_shift_new.csv', index=False)

    folds = groupk(g)
    sets = {'기존 77 (퍼짐+분포)': OLD, '새 12만': NEW,
            '기존 77 + 새 12': {s: np.hstack([OLD[s], NEW[s]]) for s in SPLITS}}
    print('\n=== 2) 이미지만 (GroupKFold 5 + Validation)')
    oofs = {}
    for n, X in sets.items():
        oof = np.zeros((len(ytr), 6))
        for a, b in folds:
            oof[b] = img_fit_predict(X['train'][a], ytr[a], X['train'][b])
        pva = img_fit_predict(X['train'], ytr, X['validation'])
        oofs[n] = (oof, pva)
        sc = [macro(ytr[b], oof[b].argmax(1) + 1) for _, b in folds]
        per = f1_score(yva, pva.argmax(1) + 1, labels=LABELS, average=None, zero_division=0)
        print(f'{n:<22} | CV {np.mean(sc):.4f} ± {np.std(sc):.4f} | Val {macro(yva, pva.argmax(1) + 1):.4f} '
              f'| Val S1~6 {np.round(per, 2).tolist()} ({time.time() - t0:.0f}s)', flush=True)

    print('\n=== 3) 센서 + 이미지 확률 6개 (중첩 CV)  [참고: ConvNeXt 스태킹 CV 0.8731 / Val 0.7898, 센서만 CV 0.8678 / Val 0.7604]')
    _, Str, _, _ = load_sensor('train')
    _, Sva, _, _ = load_sensor('validation')
    for n in ['기존 77 (퍼짐+분포)', '기존 77 + 새 12']:
        X = sets[n]['train']
        sc = []
        for a, b in folds:
            inner = np.zeros((len(a), 6))
            for ia, ib in groupk(g[a]):
                inner[ib] = img_fit_predict(X[a[ia]], ytr[a[ia]], X[a[ib]])
            ev = img_fit_predict(X[a], ytr[a], X[b])
            sc.append(macro(ytr[b], build_lgbm().fit(np.hstack([Str[a], inner]), ytr[a]).predict(np.hstack([Str[b], ev]))))
        oof, pva = oofs[n]
        pv = build_lgbm().fit(np.hstack([Str, oof]), ytr).predict(np.hstack([Sva, pva]))
        per = f1_score(yva, pv, labels=LABELS, average=None, zero_division=0)
        print(f'센서 + {n} 확률{"":<4} | CV {np.mean(sc):.4f} ± {np.std(sc):.4f} | fold {" ".join(f"{x:.3f}" for x in sc)} '
              f'| Val {macro(yva, pv):.4f} | Val S1~6 {np.round(per, 2).tolist()} ({time.time() - t0:.0f}s)', flush=True)
    print(f'총 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
