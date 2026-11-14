"""단조 후처리 비교: 원래 예측 / 누적 최댓값 / 확률 기반 단조 경로(전체 시퀀스·과거만).

평가 데이터: ① Train GroupKFold(5) OOF 예측  ② Train으로 학습한 모델의 Validation 예측
지표: Macro F1, 실험별 F1 개선·악화 수, 과대 예측(pred>true, 너무 이른 상위 단계) / 과소 예측 비율, 역행 횟수.
원래 예측과 확률은 monotone_preds_<모듈>.csv 로 보관합니다.

사용법: python monotone_eval.py model_image_lgbm [model_lgbm ...]
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
DATA = DATA_ROOT
LABELS = [1, 2, 3, 4, 5, 6]
EPS = 1e-6


# ---------------------------------------------------------------- 후처리 (한 실험, 시간순 확률 P: n×6)
def cummax_path(P):
    return np.maximum.accumulate(P.argmax(1))


def viterbi_full(P):
    """전체 시퀀스에서 log-확률 합이 최대인 비감소 경로 (미래 정보 사용 → 분석용)."""
    L = np.log(P + EPS)
    n, k = L.shape
    score = L[0].copy()
    back = np.zeros((n, k), int)
    for t in range(1, n):
        best_prev = np.maximum.accumulate(score)                 # 상태 j로 올 수 있는 이전 상태 i<=j 중 최대
        arg_prev = np.array([np.argmax(score[:j + 1]) for j in range(k)])
        score = best_prev + L[t]
        back[t] = arg_prev
    path = np.zeros(n, int)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_causal(P):
    """시점 t마다 0..t 확률만으로 구한 최적 비감소 경로의 끝 상태 (과거만 사용 → 제출 가능 후보)."""
    L = np.log(P + EPS)
    score = L[0].copy()
    out = [score.argmax()]
    for t in range(1, len(L)):
        score = np.maximum.accumulate(score) + L[t]
        out.append(score.argmax())
    return np.array(out)


METHODS = {'원래 예측': lambda P: P.argmax(1), '누적 최댓값': cummax_path,
           '확률 경로(전체·분석용)': viterbi_full, '확률 경로(과거만)': viterbi_causal}


# ---------------------------------------------------------------- 평가
def macro(y, p):
    return f1_score(y, p, labels=LABELS, average='macro', zero_division=0)


def evaluate(df, P, title):
    """df: experiment_id, relative_time_s, label (P와 같은 순서)."""
    df = df.reset_index(drop=True)
    preds = {m: np.zeros(len(df), int) for m in METHODS}
    for _, idx in df.groupby('experiment_id').indices.items():
        idx = idx[np.argsort(df.relative_time_s.values[idx], kind='mergesort')]
        for m, fn in METHODS.items():
            preds[m][idx] = np.array(LABELS)[fn(P[idx])]
    y = df.label.values
    base_exp = {e: macro(y[i], preds['원래 예측'][i]) for e, i in df.groupby('experiment_id').indices.items()}
    rows = []
    for m, p in preds.items():
        exp_f1 = {e: macro(y[i], p[i]) for e, i in df.groupby('experiment_id').indices.items()}
        diff = np.array([exp_f1[e] - base_exp[e] for e in exp_f1])
        back = sum((np.diff(p[i[np.argsort(df.relative_time_s.values[i])]]) < 0).sum()
                   for i in df.groupby('experiment_id').indices.values())
        rows.append({'방식': m, 'Macro F1': macro(y, p),
                     '실험 개선/악화': f'{(diff > 0.005).sum()} / {(diff < -0.005).sum()}',
                     '최대 악화': diff.min(),
                     '과대예측%': (p > y).mean() * 100, '과소예측%': (p < y).mean() * 100, '역행': int(back),
                     **{f'S{l}': f1_score(y, p, labels=[l], average='macro', zero_division=0) for l in LABELS}})
    out = pd.DataFrame(rows).set_index('방식')
    print(f'\n=== {title} ({df.experiment_id.nunique()}개 실험, {len(df):,}행)')
    print(out.to_string(float_format=lambda x: f'{x:.3f}'))
    return preds


def run(mod_name):
    M = importlib.import_module(mod_name)
    cd = P_CACHE / f"work_{getattr(M, 'FEATURE_CACHE', mod_name)}"

    def load(s):
        f = pd.read_csv(DATA / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        img = pd.read_csv(cd / f'img_{s}.csv', dtype={'sample_id': 'string'},
                          float_precision='round_trip').set_index('sample_id')
        return f, M.build_features(f, img), f.label.astype(int)

    tr, Xtr, ytr = load('train')
    va, Xva, yva = load('validation')
    oof = np.zeros((len(tr), 6))
    for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups=tr.experiment_id):
        oof[b] = M.build_model(None).fit(Xtr.iloc[a], ytr.iloc[a]).predict_proba(Xtr.iloc[b])
    Pva = M.build_model(None).fit(Xtr, ytr).predict_proba(Xva)

    print(f'\n################ {mod_name}')
    p_tr = evaluate(tr, oof, 'Train OOF (GroupKFold 5)')
    p_va = evaluate(va, Pva, 'Validation')
    keep = []
    for split, df, P, pr in [('train_oof', tr, oof, p_tr), ('validation', va, Pva, p_va)]:
        k = df[['sample_id', 'experiment_id', 'relative_time_s', 'label']].reset_index(drop=True).assign(split=split)
        k[[f'p{l}' for l in LABELS]] = P
        for m, p in pr.items():
            k[m] = p
        keep.append(k)
    out = P_OUT / f'monotone_preds_{mod_name}.csv'
    pd.concat(keep).to_csv(out, index=False)
    print('예측 보관:', out)


if __name__ == '__main__':
    for name in sys.argv[1:] or ['model_image_lgbm']:
        run(name)
