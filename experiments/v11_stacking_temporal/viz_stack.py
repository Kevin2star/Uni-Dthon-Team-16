"""stack_temporal.py 출력 로그 → viz/stack_temporal.png

사용법: python stack_temporal.py | tee stack_temporal.log ; python viz_stack.py stack_temporal.log
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'stack_temporal.log'
EN = {'① 센서만': 'Sensor only', '② 센서 + 이미지 확률 6': 'Sensor + image probs (6)',
      '③ 센서 + 임베딩 시간 변화': 'Sensor + embedding change', '④ ②+③ (참고)': 'Sensor + both (ref)',
      '(참고) 센서+열퍼짐 현재 최고': 'Sensor + thermal spread (current best)', '(참고) 이미지 모델 단독': 'Image model alone'}
COL = {'Sensor only': '#7a8394', 'Sensor + image probs (6)': '#1f77b4', 'Sensor + embedding change': '#2ca02c',
       'Sensor + both (ref)': '#9467bd', 'Sensor + thermal spread (current best)': '#d62728', 'Image model alone': '#bcbd22'}

pat = re.compile(r'^(.*?)\s+\| CV ([\d.]+) ± ([\d.]+) \| fold ([\d. ]+) \| Val ([\d.]+) \| Val S1~6 \[(.*)\]')
R = []
for line in LOG.read_text().splitlines():
    m = pat.match(line.strip())
    if m:
        R.append(dict(name=EN.get(m[1].strip(), m[1].strip()), cv=float(m[2]), sd=float(m[3]),
                      folds=[float(x) for x in m[4].split()], val=float(m[5]),
                      per=[float(x) for x in m[6].split(',')]))
assert R, f'결과 줄을 찾지 못했습니다: {LOG}'

fig = plt.figure(figsize=(17, 10)); gs = fig.add_gridspec(2, 2, hspace=.38, wspace=.22)
base = next(r for r in R if r['name'] == 'Sensor only')

ax = fig.add_subplot(gs[0, 0])
for i, r in enumerate(R):
    c = COL.get(r['name'], 'k')
    ax.scatter([i] * len(r['folds']), r['folds'], color=c, alpha=.45, s=28)
    ax.hlines(r['cv'], i - .3, i + .3, color=c, lw=3)
    ax.text(i, r['cv'] + .012, f"{r['cv']:.3f}", ha='center', fontsize=9, fontweight='bold')
    ax.scatter(i, r['val'], marker='D', color='k', s=36, zorder=3)
    ax.text(i + .32, r['val'], f"{r['val']:.3f}", va='center', fontsize=8, color='.3')
ax.set_xticks(range(len(R)), [r['name'].replace(' + ', '\n+ ').replace(' (', '\n(') for r in R], fontsize=8)
ax.set_title('Macro F1: CV folds (dots), CV mean (bar), Validation (diamond)', fontsize=10); ax.grid(axis='y', alpha=.3)

ax = fig.add_subplot(gs[0, 1])
rows = [r for r in R if r['name'] not in ('Sensor only', 'Image model alone')]
x = np.arange(len(rows))
dcv = [r['cv'] - base['cv'] for r in rows]; dva = [r['val'] - base['val'] for r in rows]
ax.bar(x - .18, dcv, .36, color=[COL.get(r['name'], 'k') for r in rows], label='Δ CV mean')
ax.bar(x + .18, dva, .36, color=[COL.get(r['name'], 'k') for r in rows], alpha=.4, hatch='//', label='Δ Validation')
ax.errorbar(x - .18, dcv, yerr=[base['sd']] * len(rows), fmt='none', ecolor='k', capsize=4, lw=1)
for i, (a, b) in enumerate(zip(dcv, dva)):
    ax.text(i - .18, a + (.003 if a >= 0 else -.008), f'{a:+.3f}', ha='center', fontsize=8)
    ax.text(i + .18, b + (.003 if b >= 0 else -.008), f'{b:+.3f}', ha='center', fontsize=8)
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x, [r['name'].replace(' + ', '\n+ ').replace(' (', '\n(') for r in rows], fontsize=8)
ax.set_title(f"Gain over 'Sensor only' (error bar = CV std of sensor-only, ±{base['sd']:.3f})", fontsize=10)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

ax = fig.add_subplot(gs[1, :])
w = .8 / len(R)
for i, r in enumerate(R):
    ax.bar(np.arange(6) + (i - (len(R) - 1) / 2) * w, r['per'], w, color=COL.get(r['name'], 'k'), label=r['name'])
ax.set_xticks(range(6), [f'Stage {s}' for s in range(1, 7)]); ax.set_ylim(.3, 1.02)
ax.set_title('Validation F1 by stage', fontsize=10); ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8, ncol=3, loc='lower left')

fig.suptitle('Summarizing image info for the sensor model: stacked probabilities vs embedding change (nested CV, ConvNeXt-Tiny)',
             fontsize=12)
out = HERE / 'viz' / 'stack_temporal.png'
fig.savefig(out, dpi=105, bbox_inches='tight')
print('code', out)
