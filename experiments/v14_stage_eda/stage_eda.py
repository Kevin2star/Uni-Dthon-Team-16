"""단계별 이미지 특징 탐색 (Train).

1) stage_samples/stage{k}/c{bat_class}_{exp}_{t}s.png
   실험마다 단계별로 시작·중간·끝 3장씩 복사 → 탐색기에서 단계·배터리 종류별로 넘겨 보기
2) viz/stage_sheet_class{c}.png
   배터리 종류별 비교 시트: 행 = 실험, 열 = 1~6단계 (각 단계의 중간 시점)
3) 경계(1→2 … 5→6)마다 '같은 실험 안에서' 가장 일관되게 바뀌는 이미지 특징 순위
   - 피처: thermal.py의 퍼짐 13 + 분포 64 (모두 한 프레임 안에서만 계산 → 프레임 간 색 기준 문제 없음)
   - 실험마다 (k+1단계 평균 − k단계 평균) / 그 실험의 표준편차 → 실험들의 중앙값 = 효과 크기
   - 방향 일치율 = 같은 방향으로 바뀐 실험 비율
   결과: stage_feature_shift.csv, viz/stage_feature_shift.png

사용법: python stage_eda.py
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import thermal as T

HERE = Path(__file__).resolve().parent
DATA = Path('/home/ubuntu/data/battery_data_light/train')
OUT = HERE / 'stage_samples'
LABELS = [1, 2, 3, 4, 5, 6]

f = pd.read_csv(DATA / 'train.csv')
f['row'] = np.arange(len(f))

# ---------------------------------------------------------------- 1) 이미지 폴더
if OUT.exists():
    shutil.rmtree(OUT)
for l in LABELS:
    (OUT / f'stage{l}').mkdir(parents=True)
n = 0
for (eid, l), g in f.sort_values('relative_time_s').groupby(['experiment_id', 'label']):
    for i in sorted({0, len(g) // 2, len(g) - 1}):
        r = g.iloc[i]
        shutil.copy(DATA / r.image_path, OUT / f'stage{l}' / f'c{r.bat_class}_{eid[2:10]}_{int(r.relative_time_s):05d}s.png')
        n += 1
print(f'1) 이미지 {n}장 → {OUT}/stage1 ~ stage6  (파일명: c<배터리종류>_<실험>_<시각>.png)')

# ---------------------------------------------------------------- 2) 배터리 종류별 비교 시트
for c in [1, 2, 3]:
    exps = [e for e, g in f[f.bat_class == c].groupby('experiment_id') if g.label.nunique() == 6][:14]
    fig, axes = plt.subplots(len(exps), 6, figsize=(12, 1.9 * len(exps)))
    for r, eid in enumerate(exps):
        e = f[f.experiment_id == eid]
        for k, l in enumerate(LABELS):
            s = e[e.label == l].sort_values('relative_time_s')
            row = s.iloc[len(s) // 2]
            ax = axes[r, k]
            ax.imshow(Image.open(DATA / row.image_path)); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(f'stage {l}', fontsize=11, fontweight='bold')
            if k == 0: ax.set_ylabel(f'{eid[2:10]}\n{row.bat_cc}Ah', fontsize=7)
    fig.suptitle(f'bat_class {c}: each row = one experiment, columns = middle frame of each stage', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(HERE / 'viz' / f'stage_sheet_class{c}.png', dpi=80); plt.close(fig)
print('2) viz/stage_sheet_class1.png ~ class3.png')

# ---------------------------------------------------------------- 3) 경계별 특징 변화
spread = pd.read_csv(HERE / 'work_model_image' / 'img_train.csv').set_index('sample_id').loc[f.sample_id]
rich = pd.DataFrame(np.load(HERE / 'work_imgrep' / 'rich_train.npy'), columns=T.RICH_FEATURES, index=spread.index)
X = pd.concat([spread, rich], axis=1).reset_index(drop=True)
X['label'] = f.label.values; X['exp'] = f.experiment_id.values; X['bc'] = f.bat_class.values
feats = [c for c in X.columns if c not in ('label', 'exp', 'bc')]

rows = []
for k in range(1, 6):
    for c in ['all', 1, 2, 3]:
        sub = X if c == 'all' else X[X.bc == c]
        eff = []
        for _, e in sub.groupby('exp'):
            a, b = e[e.label == k], e[e.label == k + 1]
            if len(a) < 3 or len(b) < 3:
                continue
            sd = e[feats].std().replace(0, np.nan)
            eff.append(((b[feats].mean() - a[feats].mean()) / sd).values)
        eff = np.array(eff)
        for j, ft in enumerate(feats):
            v = eff[:, j][~np.isnan(eff[:, j])]
            if len(v) < 5:
                continue
            med = np.median(v)
            rows.append(dict(경계=f'{k}→{k + 1}', 배터리=c, 피처=ft, 효과크기=med,
                             방향일치율=(np.sign(v) == np.sign(med)).mean(), 실험수=len(v)))
R = pd.DataFrame(rows)
R['점수'] = R.효과크기.abs() * R.방향일치율
R.sort_values(['경계', '배터리', '점수'], ascending=[True, True, False]).to_csv(HERE / 'stage_feature_shift.csv', index=False)

pd.set_option('display.width', 200)
print('\n3) 경계별로 같은 실험 안에서 가장 일관되게 바뀌는 이미지 특징 (전체 실험, 상위 6)')
print('   효과크기 = (다음 단계 평균 − 이번 단계 평균)/실험 내 표준편차의 중앙값, +면 증가 / 방향일치율 = 같은 방향인 실험 비율')
for k in range(1, 6):
    t = R[(R.경계 == f'{k}→{k + 1}') & (R.배터리 == 'all')].nlargest(6, '점수')
    print(f'\n[{k}→{k + 1}]  (실험 {t.실험수.max()}개)')
    print(t[['피처', '효과크기', '방향일치율']].to_string(index=False, float_format=lambda x: f'{x:+.2f}'))

# 그림: 경계 × 피처 히트맵 (전체 실험, 각 경계 상위 8개 피처의 합집합)
top = sorted({ft for k in range(1, 6) for ft in
              R[(R.경계 == f'{k}→{k + 1}') & (R.배터리 == 'all')].nlargest(8, '점수').피처})
H = R[(R.배터리 == 'all') & R.피처.isin(top)].pivot(index='피처', columns='경계', values='효과크기').loc[top]
A = R[(R.배터리 == 'all') & R.피처.isin(top)].pivot(index='피처', columns='경계', values='방향일치율').loc[top]
fig, ax = plt.subplots(figsize=(8, .32 * len(top) + 1.5))
im = ax.imshow(H.values, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if not np.isnan(H.values[i, j]):
            ax.text(j, i, f'{H.values[i, j]:+.1f}\n{A.values[i, j] * 100:.0f}%', ha='center', va='center', fontsize=6.5)
ax.set_xticks(range(H.shape[1]), H.columns); ax.set_yticks(range(len(top)), top, fontsize=8)
ax.set_title('Within-experiment change of image features at each stage boundary\n'
             '(color/number = median effect size, red = increases; % = experiments agreeing in direction)', fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6)
fig.tight_layout(); fig.savefig(HERE / 'viz' / 'stage_feature_shift.png', dpi=110); plt.close(fig)
print('\n그림: viz/stage_feature_shift.png | 전체 표(배터리 종류별 포함): stage_feature_shift.csv')
