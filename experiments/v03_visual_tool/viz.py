"""데이터 눈으로 보기용 그림 생성기. 결과 PNG는 ~/yeonwoo/viz/ 에 저장됩니다.

사용법 (터미널):
  python viz.py grid                 # 단계(label)별 열화상 이미지 샘플 격자
  python viz.py grid --n 10 --seed 1 # 열 개수·랜덤 시드 변경
  python viz.py exps                 # 실험 목록 (번호, id, 길이, 조건)
  python viz.py exp 0                # 0번 실험: 센서 시계열 + 단계 + 단계별 이미지
  python viz.py exp E_019ea6...      # experiment_id로 지정도 가능
  python viz.py exp 3 --split validation
  python viz.py sensors              # 단계별 센서 분포 박스플롯
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path('/home/ubuntu/data/battery_data_light')
OUT_DIR = Path(__file__).resolve().parent / 'viz'
LABEL_COLORS = {1: '#4caf50', 2: '#cddc39', 3: '#ffc107', 4: '#ff9800', 5: '#f44336', 6: '#7b1fa2'}
GASES = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
         'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']


def load(split):
    df = pd.read_csv(DATA_DIR / split / f'{split}.csv')
    df['img'] = df.image_path.map(lambda p: DATA_DIR / split / p)
    return df


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('저장:', path)
    print('열기:', f'code {path}')


def grid(args):
    df = load(args.split)
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(6, args.n, figsize=(args.n * 1.8, 6 * 1.7))
    for r, label in enumerate(range(1, 7)):
        sub = df[df.label == label]
        # 서로 다른 실험에서 고르게 뽑기
        exps = rng.permutation(sub.experiment_id.unique())
        picks = [sub[sub.experiment_id == exps[i % len(exps)]].sample(1, random_state=args.seed + i).iloc[0]
                 for i in range(args.n)]
        for c, row in enumerate(picks):
            ax = axes[r, c]
            ax.imshow(Image.open(row.img))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f't={row.relative_time_s}s', fontsize=7)
            if c == 0:
                ax.set_ylabel(f'stage {label}', fontsize=12, color=LABEL_COLORS[label], fontweight='bold')
    fig.suptitle(f'{args.split}: thermal images by stage (each row = one stage, random experiments)')
    save(fig, f'grid_{args.split}_seed{args.seed}.png')


def exps(args):
    df = load(args.split)
    g = df.groupby('experiment_id', sort=False).agg(
        rows=('sample_id', 'size'), t_max=('relative_time_s', 'max'),
        bat_class=('bat_class', 'first'), bat_cc=('bat_cc', 'first'),
        soc=('soc_status', 'first'), tr_method=('tr_method', 'first'),
        **({'stages': ('label', lambda s: ''.join(map(str, sorted(s.unique()))))} if 'label' in df else {}))
    g = g.reset_index()
    pd.set_option('display.width', 200)
    print(g.to_string())


def pick_exp(df, key):
    ids = df.experiment_id.unique()
    return ids[int(key)] if key.isdigit() else key


def exp(args):
    df = load(args.split)
    eid = pick_exp(df, args.key)
    e = df[df.experiment_id == eid].sort_values('relative_time_s')
    assert len(e), f'실험 없음: {eid}'
    t = e.relative_time_s.values
    has_label = 'label' in e

    panels = [('Voltage [V]', ['meas_v']), ('Temperature [C]', ['meas_t1', 'meas_t2', 'atmp_t', 'thermal_t', 'indu_t']),
              ('Pressure', ['meas_p']), ('Gas', GASES)]
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(len(panels) + 2, 1, height_ratios=[1] * len(panels) + [0.25, 1.6], hspace=0.35)
    ax0 = None
    for i, (title, cols) in enumerate(panels):
        ax = fig.add_subplot(gs[i], sharex=ax0)
        ax0 = ax0 or ax
        for col in cols:
            if e[col].notna().any():
                ax.plot(t, e[col], label=col, lw=1)
        if has_label:  # 단계 구간 배경색
            for label, seg in e.groupby((e.label != e.label.shift()).cumsum()):
                ax.axvspan(seg.relative_time_s.min(), seg.relative_time_s.max(),
                           color=LABEL_COLORS[seg.label.iloc[0]], alpha=0.12, lw=0)
        ax.set_ylabel(title); ax.legend(fontsize=7, ncol=5, loc='upper left')
        ax.tick_params(labelbottom=False)
    if has_label:  # 단계 띠
        ax = fig.add_subplot(gs[len(panels)], sharex=ax0)
        ax.scatter(t, np.zeros_like(t), c=[LABEL_COLORS[l] for l in e.label], marker='|', s=400)
        ax.set_yticks([]); ax.set_xlabel('relative_time_s')
        for l, c in LABEL_COLORS.items():
            ax.plot([], [], color=c, lw=6, label=f'stage {l}')
        ax.legend(ncol=6, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.8))

    # 이미지 띠: 단계가 있으면 각 단계의 시작·중간·끝, 없으면 균등 간격
    if has_label:
        rows = []
        for _, seg in e.groupby('label'):
            rows += [seg.iloc[0], seg.iloc[len(seg) // 2], seg.iloc[-1]]
    else:
        rows = [e.iloc[i] for i in np.linspace(0, len(e) - 1, 12).astype(int)]
    sub = gs[len(panels) + 1].subgridspec(1, len(rows), wspace=0.05)
    for j, row in enumerate(rows):
        ax = fig.add_subplot(sub[j])
        ax.imshow(Image.open(row.img)); ax.set_xticks([]); ax.set_yticks([])
        title = f'{row.relative_time_s}s'
        if has_label:
            title = f'S{row.label} {title}'
            for s in ax.spines.values():
                s.set_color(LABEL_COLORS[row.label]); s.set_linewidth(3)
        ax.set_title(title, fontsize=7)

    r = e.iloc[0]
    fig.suptitle(f'{args.split} | {eid} | bat_class={r.bat_class} bat_cc={r.bat_cc} '
                 f'soc={r.soc_status} tr_method={r.tr_method} | {len(e)} rows', fontsize=11)
    save(fig, f'exp_{args.split}_{eid[:10]}.png')


def sensors(args):
    df = load(args.split)
    cols = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'thermal_t', 'atmp_t'] + GASES
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    for ax, col in zip(axes.flat, cols):
        data = [df.loc[df.label == l, col].dropna() for l in range(1, 7)]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True)
        for patch, l in zip(bp['boxes'], range(1, 7)):
            patch.set_facecolor(LABEL_COLORS[l]); patch.set_alpha(0.6)
        ax.set_xticks(range(1, 7), [f'S{l}' for l in range(1, 7)])
        ax.set_title(col)
    fig.suptitle(f'{args.split}: sensor distribution by stage (outliers hidden)')
    save(fig, f'sensors_{args.split}.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['grid', 'exps', 'exp', 'sensors'])
    p.add_argument('key', nargs='?', default='0', help='exp: 실험 번호 또는 experiment_id')
    p.add_argument('--split', default='train', choices=['train', 'validation', 'test'])
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    {'grid': grid, 'exps': exps, 'exp': exp, 'sensors': sensors}[args.cmd](args)
