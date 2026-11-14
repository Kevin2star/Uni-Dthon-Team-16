"""단계 전환(2→3, 4→5) 전후의 정답 vs 예측 비교.

입력: monotone_preds_<모듈>.csv (monotone_eval.py가 저장한 Train OOF + Validation 예측)
출력: viz/trans_summary_<모듈>.png        전환 시점 기준으로 맞춘 정답률·늦음·이름 비율 (모든 실험 평균)
      viz/trans_<23|45>_<split>_<모듈>.png 실험별 작은 그림 (검은 계단 = 정답, 점 = 예측)
      터미널 표: 실험별 '예측이 따라잡기까지 걸린 시간', 전환 후 구간별 정답률

사용법: python viz_transitions.py [model_image_lgbm] [--col 원래 예측]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRANS = {'23': dict(src=2, dst=3, before=300, after=600), '45': dict(src=4, dst=5, before=200, after=400)}
C_OK, C_LATE, C_EARLY = '#2e7d32', '#1565c0', '#c62828'   # 맞음 / 정답보다 낮음(늦음) / 정답보다 높음(이름)


def transitions(d, spec):
    """실험별 전환 시점 t0 = 처음으로 label >= dst 가 된 시각. 직전 단계가 src가 아니면 kind에 표시."""
    out = []
    for (split, eid), g in d.groupby(['split', 'experiment_id']):
        g = g.sort_values('relative_time_s')
        hit = np.nonzero(g.label.values >= spec['dst'])[0]
        if len(hit) == 0 or hit[0] == 0:
            continue
        prev = g.label.values[hit[0] - 1]
        out.append(dict(split=split, eid=eid, t0=g.relative_time_s.values[hit[0]], prev=int(prev), g=g))
    return out


def per_exp_table(trs, spec, col):
    rows = []
    for tr in trs:
        g, t0, dst = tr['g'], tr['t0'], spec['dst']
        rel = g.relative_time_s.values - t0
        y, p = g.label.values, g[col].values
        after = rel >= 0
        caught = rel[after & (p >= dst)]
        lag = caught.min() if len(caught) else np.inf                 # 예측이 dst 이상을 처음 낸 시각(전환 기준)
        early = ((rel < 0) & (rel >= -spec['before']) & (p >= dst)).sum()  # 전환 전인데 dst 이상 예측
        def acc(lo, hi):
            m = (rel >= lo) & (rel < hi) & (y == dst)
            return (p[m] == y[m]).mean() if m.any() else np.nan
        rows.append(dict(split=tr['split'], exp=tr['eid'][2:10], 직전단계=tr['prev'],
                         따라잡기_s=lag, 전환전_이른예측=int(early),
                         acc_0_60s=acc(0, 60), acc_60_300s=acc(60, 300), acc_300s_=acc(300, 1e9)))
    return pd.DataFrame(rows)


def summary_plot(d, col, name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharey=True)
    for c, (key, spec) in enumerate(TRANS.items()):
        trs = transitions(d, spec)
        for r, split in enumerate(['train_oof', 'validation']):
            ax = axes[r, c]
            rows = []
            for tr in [t for t in trs if t['split'] == split and t['prev'] == spec['src']]:
                g = tr['g']
                rel = g.relative_time_s.values - tr['t0']
                m = (rel >= -spec['before']) & (rel <= spec['after'])
                rows.append(pd.DataFrame({'rel': (rel[m] // 10) * 10, 'y': g.label.values[m], 'p': g[col].values[m]}))
            a = pd.concat(rows)
            s = a.groupby('rel').apply(lambda x: pd.Series({'맞음': (x.p == x.y).mean(), '늦음': (x.p < x.y).mean(),
                                                             '이름': (x.p > x.y).mean(), 'n': len(x)}), include_groups=False)
            s = s[s.n >= 3]
            ax.stackplot(s.index, s['맞음'], s['늦음'], s['이름'], colors=[C_OK, C_LATE, C_EARLY], alpha=.75,
                         labels=['correct', 'pred < true (late)', 'pred > true (early)'], step='post')
            ax.axvline(0, color='k', lw=1.2, ls='--')
            ax.set_xlim(-spec['before'], spec['after']); ax.set_ylim(0, 1)
            ax.set_title(f"{split}: stage {spec['src']}→{spec['dst']} ({len(rows)} experiments, aligned at transition)", fontsize=10)
            ax.set_xlabel('seconds from true transition'); ax.grid(alpha=.3)
            if c == 0: ax.set_ylabel('share of rows')
            if r == 0 and c == 0: ax.legend(fontsize=8, loc='lower left')
    col_en = {'원래 예측': 'raw', '누적 최댓값': 'cummax', '확률 경로(전체·분석용)': 'viterbi-full',
              '확률 경로(과거만)': 'viterbi-causal'}.get(col, 'custom')   # 서버에 한글 폰트가 없어 영어로 표기
    fig.suptitle(f'{name} [{col_en}]: is the model wrong only right after the transition, or persistently?', fontsize=12)
    path = HERE / 'viz' / f'trans_summary_{name}.png'
    fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)
    return path


def grid_plot(d, col, name, key, split):
    spec = TRANS[key]
    trs = sorted([t for t in transitions(d, spec) if t['split'] == split],
                 key=lambda t: (t['prev'] != spec['src'], t['eid']))
    n = len(trs); ncol = 6 if n > 24 else 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.6, nrow * 2.3), sharey=True, squeeze=False)
    for ax, tr in zip(axes.flat, trs):
        g = tr['g']
        rel = g.relative_time_s.values - tr['t0']
        m = (rel >= -spec['before']) & (rel <= spec['after'])
        x, y, p = rel[m], g.label.values[m], g[col].values[m]
        ax.step(x, y, where='post', color='k', lw=1.6)
        cols = np.where(p == y, C_OK, np.where(p < y, C_LATE, C_EARLY))
        ax.scatter(x, p, c=cols, s=7, zorder=3)
        ax.axvline(0, color='0.5', ls='--', lw=.8)
        acc = (p == y).mean()
        flag = '' if tr['prev'] == spec['src'] else f"  ({tr['prev']}→{spec['dst']}, no stage {spec['src']})"
        ax.set_title(f"{tr['eid'][2:10]}  acc {acc:.2f}{flag}", fontsize=8, color='k' if not flag else C_EARLY)
        ax.set_yticks(range(1, 7)); ax.set_ylim(.5, 6.5); ax.grid(alpha=.25); ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis('off')
    fig.suptitle(f"{split} | stage {spec['src']}→{spec['dst']} | black step = true stage, dots = prediction "
                 f"(green correct, blue lower=late, red higher=early) | x = seconds from transition", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = HERE / 'viz' / f'trans_{key}_{split}_{name}.png'
    fig.savefig(path, dpi=100); plt.close(fig)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default='model_image_lgbm')
    ap.add_argument('--col', default='원래 예측', help='비교할 예측 컬럼 (원래 예측 / 누적 최댓값 / 확률 경로(과거만) ...)')
    a = ap.parse_args()
    (HERE / 'viz').mkdir(exist_ok=True)
    d = pd.read_csv(HERE / f'monotone_preds_{a.name}.csv')
    pd.set_option('display.width', 200)
    paths = [summary_plot(d, a.col, a.name)]
    for key, spec in TRANS.items():
        t = per_exp_table(transitions(d, spec), spec, a.col)
        print(f"\n===== stage {spec['src']}→{spec['dst']}  (따라잡기_s: 전환 후 예측이 처음 {spec['dst']} 이상이 된 시각, inf=끝까지 못 따라잡음)")
        for split, s in t.groupby('split'):
            s = s[s.직전단계 == spec['src']]
            print(f"[{split}] {len(s)}개 실험 | 따라잡기 중앙값 {s.따라잡기_s.replace(np.inf, np.nan).median():.0f}s, "
                  f"못 따라잡음 {np.isinf(s.따라잡기_s).sum()}개 | 전환 후 정답률 0~60s {s.acc_0_60s.mean():.2f}, "
                  f"60~300s {s.acc_60_300s.mean():.2f}, 300s~ {s.acc_300s_.mean():.2f} | 전환 전 이른 예측 있는 실험 {(s.전환전_이른예측 > 0).sum()}개")
        print(t[t.split == 'validation'].drop(columns='split').round(2).to_string(index=False))
        for split in ['validation', 'train_oof']:
            paths.append(grid_plot(d, a.col, a.name, key, split))
    print('\n그림:')
    for p in paths:
        print('  code', p)
