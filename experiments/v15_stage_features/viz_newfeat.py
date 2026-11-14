"""outputs/newfeat.log → outputs/viz/newfeat.png (v15 단계 관찰 피처 결과)"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P_OUT = Path(__file__).resolve().parents[1] / 'outputs'
lines = (P_OUT / 'newfeat.log').read_text().splitlines()

# 1) 경계별: 기존 최고 vs 새 최고 (점수 = |효과| × 일치율)
bounds, old, new = [], [], []
cur = None
for l in lines:
    m = re.match(r'\[(\d→\d)\] 기존 최고: (\S+) ([+-][\d.]+) \((\d+)%\)', l)
    if m:
        cur = m[1]; bounds.append(cur); old.append((m[2], abs(float(m[3])) * int(m[4]) / 100)); new.append(None); continue
    m = re.match(r'\s+새: (\S+)\s+([+-][\d.]+) \((\d+)%\)', l)
    if m and cur and new[-1] is None:
        new[-1] = (m[1], abs(float(m[2])) * int(m[3]) / 100)
# 2) 이미지만 / 3) 스태킹
EN = {'기존 77 (퍼짐+분포)': 'Old 77 (spread+rich)', '새 12만': 'New 12 only', '기존 77 + 새 12': 'Old 77 + New 12'}
img, stk = {}, {}
for l in lines:
    m = re.match(r'^(기존 77 \(퍼짐\+분포\)|새 12만|기존 77 \+ 새 12)\s+\| CV ([\d.]+) ± ([\d.]+) \| Val ([\d.]+)', l)
    if m: img[EN[m[1]]] = (float(m[2]), float(m[3]), float(m[4]))
    m = re.match(r'^센서 \+ (.*?) 확률\s+\| CV ([\d.]+) ± ([\d.]+) \| fold ([\d. ]+)\| Val ([\d.]+)', l)
    if m: stk['Sensor + ' + EN[m[1].strip()]] = (float(m[2]), float(m[3]), [float(x) for x in m[4].split()], float(m[5]))
stk = {'Sensor only': (0.8678, 0.0306, [.892, .862, .896, .878, .811], 0.7604),
       'Sensor + ConvNeXt (v11)': (0.8731, 0.0360, [.897, .853, .899, .905, .812], 0.7898), **stk}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw=dict(wspace=.3))
ax = axes[0]; x = np.arange(len(bounds))
ax.bar(x - .2, [o[1] for o in old], .4, color='#9aa3b2', label='best of old 77')
ax.bar(x + .2, [n[1] for n in new], .4, color='#1f77b4', label='best of new 12')
for i, (o, n) in enumerate(zip(old, new)):
    ax.text(i - .2, o[1] + .02, o[0], rotation=90, ha='center', va='bottom', fontsize=7)
    ax.text(i + .2, n[1] + .02, n[0], rotation=90, ha='center', va='bottom', fontsize=7, color='#1f4e79')
ax.set_xticks(x, bounds); ax.set_ylim(0, 2.3); ax.legend(fontsize=8, loc='upper right'); ax.grid(axis='y', alpha=.3)
ax.set_title('Within-experiment change at each boundary\n(|effect| × direction agreement; higher = clearer signal)', fontsize=10)

ax = axes[1]; names = list(img)
for i, n in enumerate(names):
    cv, sd, v = img[n]
    ax.errorbar(i, cv, yerr=sd, fmt='o', ms=9, capsize=5, color=['#9aa3b2', '#1f77b4', '#2ca02c'][i])
    ax.scatter(i, v, marker='D', color='k', s=30, zorder=3); ax.text(i + .12, cv, f'{cv:.3f}', va='center', fontsize=9)
ax.set_xticks(range(len(names)), [n.replace(' (', '\n(') for n in names], fontsize=8); ax.set_ylim(.65, .82)
ax.grid(axis='y', alpha=.3); ax.set_title('Image only: CV mean ± std (circle), Validation (diamond)', fontsize=10)

ax = axes[2]; names = list(stk); C = ['#bbb', '#7a8394', '#9aa3b2', '#2ca02c']
for i, n in enumerate(names):
    cv, sd, fs, v = stk[n]
    ax.scatter([i] * 5, fs, color=C[i], alpha=.5, s=25); ax.hlines(cv, i - .3, i + .3, color=C[i], lw=3)
    ax.text(i, cv + .012, f'{cv:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.scatter(i, v, marker='D', color='k', s=30, zorder=3); ax.text(i + .15, v, f'{v:.3f}', fontsize=8, va='center', color='.3')
ax.set_xticks(range(len(names)), [n.replace(' + ', '\n+ ').replace(' (', '\n(') for n in names], fontsize=7)
ax.grid(axis='y', alpha=.3); ax.set_title('Sensor + image probs (nested CV): folds, mean, Val', fontsize=10)
fig.suptitle('v15: stage-observation image features (noise, holes, dark background, vertical position, shape)', fontsize=12)
fig.savefig(P_OUT / 'viz' / 'newfeat.png', dpi=105, bbox_inches='tight')
print('code', P_OUT / 'viz' / 'newfeat.png')
