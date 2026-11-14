"""경보 5개(순서형) 결과 시각화 → viz/ordinal_compare.png, viz/ordinal_alarms_validation.png

입력: ordinal_probs_<모듈>.csv (ordinal_eval.py), monotone_preds_<모듈>.csv (monotone_eval.py의 6클래스 원래 예측)
사용법: python viz_ordinal.py [model_image_lgbm]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from ordinal_eval import KS, LABELS, decode, macro, tune
from paths import DATA_ROOT  # 데이터 위치: 환경변수 DATA_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                 # ~/yeonwoo
P_CACHE = ROOT / 'cache'           # 이미지 피처·임베딩 캐시
P_OUT = ROOT / 'outputs'           # 그림·로그·예측 결과
DATA = DATA_ROOT
name = sys.argv[1] if len(sys.argv) > 1 else 'model_image_lgbm'
STAGE_C = ['#3d7cc9', '#1f9e8f', '#c99a12', '#e0731f', '#cf3b2b', '#7e3a99']
METH_C = {'6-class LGBM': '#7a8394', 'Alarms @0.5': '#1f77b4', 'Alarms tuned': '#e377c2'}

q = pd.read_csv(P_OUT / f'ordinal_probs_{name}.csv', dtype={'sample_id': 'string'})
mc = pd.read_csv(P_OUT / f'monotone_preds_{name}.csv', dtype={'sample_id': 'string'})[['sample_id', 'split', '원래 예측']]
meta = pd.concat([pd.read_csv(DATA / s / f'{s}.csv', usecols=['sample_id', 'experiment_id', 'relative_time_s'],
                              dtype={'sample_id': 'string'}) for s in ['train', 'validation']])
d = q.merge(mc, on=['sample_id', 'split']).merge(meta, on='sample_id')
Qc = [f'q_ge{k}' for k in KS]

tr = d[d.split == 'train_oof'].reset_index(drop=True)
va = d[d.split == 'validation'].reset_index(drop=True)
# 학습 때와 같은 분할을 재현하려면 train.csv 원래 행 순서가 필요 → 원본 순서로 정렬
order = pd.read_csv(DATA / 'train' / 'train.csv', usecols=['sample_id'], dtype={'sample_id': 'string'}).sample_id
tr = tr.set_index('sample_id').loc[order].reset_index()
folds = list(GroupKFold(n_splits=5).split(tr, tr.label, groups=tr.experiment_id))
Qtr, ytr = tr[Qc].values, tr.label.values

tuned = np.zeros(len(tr), int); fold_th = []
for i, (_, b) in enumerate(folds):
    rest = np.concatenate([folds[j][1] for j in range(5) if j != i])
    th = tune(Qtr[rest], ytr[rest]); fold_th.append(th); tuned[b] = decode(Qtr[b], th)
th_all = tune(Qtr, ytr)
P = {'6-class LGBM': (tr['원래 예측'].values, va['원래 예측'].values),
     'Alarms @0.5': (decode(Qtr, [.5] * 5), decode(va[Qc].values, [.5] * 5)),
     'Alarms tuned': (tuned, decode(va[Qc].values, th_all))}

# ---------------------------------------------------------------- 그림 1: 비교
fig = plt.figure(figsize=(17, 10)); gs = fig.add_gridspec(2, 3, hspace=.4, wspace=.28)
ax = fig.add_subplot(gs[0, 0])
for i, (m, (po, pv)) in enumerate(P.items()):
    fs = [macro(ytr[b], po[b]) for _, b in folds]
    for f, (_, b) in enumerate(folds):
        pass
    ax.plot(range(5), fs, '-o', color=METH_C[m], label=f'{m}  (mean {np.mean(fs):.3f})', ms=5)
ax.set_xticks(range(5), [f'fold {i + 1}' for i in range(5)]); ax.set_ylim(.78, .94); ax.grid(alpha=.3)
ax.set_title('CV Macro F1 per fold (same folds)', fontsize=10); ax.legend(fontsize=8, loc='lower left')

for j, (split, y, idx) in enumerate([('CV (Train OOF)', ytr, 0), ('Validation', va.label.values, 1)]):
    ax = fig.add_subplot(gs[0, 1 + j]); w = .27
    for i, (m, pp) in enumerate(P.items()):
        f = f1_score(y, pp[idx], labels=LABELS, average=None, zero_division=0)
        ax.bar(np.arange(6) + (i - 1) * w, f, w, color=METH_C[m], label=f'{m} ({macro(y, pp[idx]):.3f})')
    ax.set_xticks(range(6), [f'S{l}' for l in LABELS]); ax.set_ylim(.4, 1.02); ax.grid(axis='y', alpha=.3)
    ax.set_title(f'{split}: F1 by stage', fontsize=10); ax.legend(fontsize=8, loc='lower left')

ax = fig.add_subplot(gs[1, 0])
TH = np.array(fold_th)
for i in range(5):
    ax.plot([f'≥{k}' for k in KS], TH[i], '-o', alpha=.7, label=f'tuned w/o fold {i + 1}')
ax.plot([f'≥{k}' for k in KS], th_all, 'k--s', label='tuned on all OOF (used for Val)')
ax.axhline(.5, color='0.5', ls=':'); ax.set_ylim(0, 1); ax.grid(alpha=.3)
ax.set_title('Tuned thresholds swing across folds → overfitting', fontsize=10); ax.legend(fontsize=7)

ax = fig.add_subplot(gs[1, 1:])
y, Q = ytr, Qtr
for i, k in enumerate(KS):
    for s in LABELS:
        v = Q[y == s, i]
        pos = i * 7 + s - 1
        ax.boxplot(v, positions=[pos], widths=.7, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=STAGE_C[s - 1], alpha=.7), medianprops=dict(color='k'))
    ax.axvline(i * 7 + 5.5 + .25, color='0.8')
    ax.text(i * 7 + 2.5, 1.04, f'alarm "stage ≥{k}?"', ha='center', fontsize=9)
ax.set_xticks([i * 7 + s - 1 for i in range(5) for s in LABELS], [str(s) for _ in KS for s in LABELS], fontsize=7)
ax.set_xlabel('true stage (colors) within each alarm'); ax.set_ylabel('alarm probability (Train OOF)')
ax.set_ylim(-.03, 1.1); ax.axhline(.5, color='0.5', ls=':')
ax.set_title('Each alarm separates stages below vs at/above its boundary', fontsize=10)
fig.suptitle(f'{name}: 5 ordinal alarms vs 6-class classifier', fontsize=13)
fig.savefig(P_OUT / 'viz' / 'ordinal_compare.png', dpi=105, bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 그림 2: 실험별 경보 타임라인 (Validation)
exps = sorted(va.experiment_id.unique())
fig, axes = plt.subplots(len(exps), 1, figsize=(15, 1.55 * len(exps)), sharex=False)
for ax, eid in zip(axes, exps):
    e = va[va.experiment_id == eid].sort_values('relative_time_s')
    t = e.relative_time_s.values
    for i, k in enumerate(KS):
        ax.plot(t, e[f'q_ge{k}'], color=STAGE_C[k - 1], lw=1.1)
    ax2 = ax.twinx()
    ax2.step(t, e.label, where='post', color='k', lw=1.4)
    ax2.plot(t, decode(e[Qc].values, [.5] * 5), color='k', lw=.8, ls=':', alpha=.8)
    ax2.set_ylim(.5, 6.5); ax2.set_yticks([1, 3, 6]); ax2.tick_params(labelsize=7)
    ax.set_ylim(-.05, 1.05); ax.set_yticks([0, .5, 1]); ax.tick_params(labelsize=7)
    f = macro(e.label, decode(e[Qc].values, [.5] * 5))
    ax.text(.005, .85, f'{eid[2:10]}  alarms F1 {f:.2f} / 6-class {macro(e.label, e["원래 예측"]):.2f}',
            transform=ax.transAxes, fontsize=8, bbox=dict(fc='w', ec='none', alpha=.8))
handles = [plt.Line2D([], [], color=STAGE_C[k - 1], label=f'P(stage ≥{k})') for k in KS]
handles += [plt.Line2D([], [], color='k', label='true stage (right axis)'),
            plt.Line2D([], [], color='k', ls=':', label='alarms @0.5 → stage')]
fig.legend(handles=handles, loc='upper center', ncol=7, fontsize=9, bbox_to_anchor=(.5, 1.0))
axes[-1].set_xlabel('relative_time_s')
fig.suptitle('Validation: alarm probabilities over time (should fire one after another as the true stage steps up)',
             fontsize=11, y=1.015)
fig.tight_layout(rect=(0, 0, 1, .985))
fig.savefig(P_OUT / 'viz' / 'ordinal_alarms_validation.png', dpi=95, bbox_inches='tight'); plt.close(fig)
print('code', P_OUT / 'viz' / 'ordinal_compare.png')
print('code', P_OUT / 'viz' / 'ordinal_alarms_validation.png')
