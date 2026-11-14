"""firstframe.log → viz/firstframe.png (결과 + 처음 프레임 기준이 안 된 이유 예시)"""
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import firstframe_eval as F
import thermal as T

EN = {'R0 임베딩 그대로': 'R0 embedding (as is)', 'R1 임베딩 차이': 'R1 embedding − first',
      'R2 직접 피처 차이': 'R2 hand features − first', 'R3 픽셀 차이 → ConvNeXt': 'R3 pixel diff → ConvNeXt'}
wrap = lambda n: n.replace(' (', '\n(').replace(' − ', '\n− ').replace(' → ', '\n→ ')

diag, img, stk = {}, {}, {}
for l in open('firstframe.log').read().splitlines():
    m = re.match(r'^(R\d .*?)\s+(\d+)%\s+(\d+)%\s+(\d+)%', l)
    if m: diag[EN[m[1].strip()]] = [int(m[2]), int(m[3]), int(m[4])]
    m = re.match(r'^(R\d .*?)\s+\| CV ([\d.]+) ± ([\d.]+) \| Val ([\d.]+)', l)
    if m: img[EN[m[1].strip()]] = (float(m[2]), float(m[3]), float(m[4]))
    m = re.match(r'^센서(.*?)\s+\| CV ([\d.]+) ± ([\d.]+) \| fold ([\d. ]+)\| Val ([\d.]+)', l)
    if m:
        k = m[1].replace(' + ', '').replace(' 확률', '').strip()
        stk[EN.get(k, 'Sensor only')] = (float(m[2]), float(m[3]), [float(x) for x in m[4].split()], float(m[5]))

names = list(EN.values()); C = ['#7a8394', '#1f77b4', '#2ca02c', '#d62728']
fig = plt.figure(figsize=(17, 12)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=.35, wspace=.28)

ax = fig.add_subplot(gs[0, 0]); w = .27
for j, (lab, c) in enumerate(zip(['experiment share', 'stage share', 'stage share within exp'], ['#999', '#e07b1f', '#2e7d32'])):
    ax.bar(np.arange(4) + (j - 1) * w, [diag[n][j] for n in names], w, color=c, label=lab)
ax.set_xticks(range(4), [wrap(n) for n in names], fontsize=8); ax.set_ylabel('% of variance')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3); ax.set_title('Variance diagnosis (want: experiment ↓, stage ↑)', fontsize=10)

ax = fig.add_subplot(gs[0, 1])
for i, n in enumerate(names):
    cv, sd, v = img[n]
    ax.errorbar(i, cv, yerr=sd, fmt='o', color=C[i], capsize=5, ms=8); ax.scatter(i, v, marker='D', color='k', s=30, zorder=3)
    ax.text(i + .12, cv, f'{cv:.3f}', fontsize=9, va='center')
ax.set_xticks(range(4), [wrap(n) for n in names], fontsize=8); ax.set_ylim(.45, .8); ax.grid(axis='y', alpha=.3)
ax.set_title('Image only: CV mean ± std (circle), Validation (diamond)', fontsize=10)

ax = fig.add_subplot(gs[0, 2]); order = ['Sensor only'] + names; CC = ['#bbb'] + C
for i, n in enumerate(order):
    cv, sd, fs, v = stk[n]
    ax.scatter([i] * 5, fs, color=CC[i], alpha=.45, s=25); ax.hlines(cv, i - .3, i + .3, color=CC[i], lw=3)
    ax.text(i, cv + .01, f'{cv:.3f}', ha='center', fontsize=9, fontweight='bold'); ax.scatter(i, v, marker='D', color='k', s=30, zorder=3)
ax.set_xticks(range(5), [wrap(n) for n in order], fontsize=7); ax.grid(axis='y', alpha=.3)
ax.set_title('Sensor + image probs (nested CV): folds, mean, Val', fontsize=10)

D = '/home/ubuntu/data/battery_data_light/train/'
f = pd.read_csv(D + 'train.csv')
sub = gs[1, :].subgridspec(2, 5, wspace=.08, hspace=.3)
for r, pre in enumerate(['E_019ea6', 'E_193138']):
    e = f[f.experiment_id.str.startswith(pre)].sort_values('relative_time_s')
    rows = [e.iloc[0]] + [e[e.label == l].iloc[len(e[e.label == l]) // 2] for l in [2, 4, 6]]
    ref = T.decode(D + rows[0].image_path)
    for c, row in enumerate(rows):
        ax = fig.add_subplot(sub[r, c]); ax.imshow(Image.open(D + row.image_path)); ax.axis('off')
        ax.set_title(('FIRST frame' if c == 0 else f'stage {row.label}') + f'  t={row.relative_time_s}s', fontsize=8)
    cur = T.decode(D + rows[2].image_path).astype(int)
    rr = ref if ref.shape == cur.shape else np.asarray(Image.fromarray(ref).resize(cur.shape[::-1]))
    ax = fig.add_subplot(sub[r, 4]); ax.imshow(cur - rr, cmap='RdBu_r', vmin=-255, vmax=255); ax.axis('off')
    ax.set_title('stage 4 − FIRST (index diff, red = higher)', fontsize=8)
fig.text(.5, .505,'Why it failed: each image is auto-scaled to its own min–max, so the cool, nearly uniform FIRST frame stretches tiny '
         'differences to the full color range.\n"Now − first" mixes real heating with a change of color scale, and the battery layout '
         '(cell edges) remains — it does not cancel out.', ha='center', fontsize=10, color='#333')
fig.suptitle('First-frame reference: did subtracting the first frame remove the experiment-specific look?  → No', fontsize=13)
fig.savefig('viz/firstframe.png', dpi=100, bbox_inches='tight')
print('code', 'viz/firstframe.png')
