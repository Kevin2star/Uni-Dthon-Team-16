import hashlib
from pathlib import Path

import pandas as pd

DATA_DIR = Path('/home/ubuntu/data/battery_data_light')
SPLITS = ['train', 'validation', 'test']
frames = {s: pd.read_csv(DATA_DIR / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
          for s in SPLITS}
sub = pd.read_csv(DATA_DIR / 'sample_submission.csv', dtype={'sample_id': 'string'})

# 1) ID 중복: split 안 / split 간 sample_id 중복
print('== 1. ID 중복')
for s, f in frames.items():
    print(f'{s}: 행 {len(f):,} / sample_id 중복 {f.sample_id.duplicated().sum()} / 결측 {f.sample_id.isna().sum()}')
all_ids = pd.concat([f.sample_id for f in frames.values()])
print('split 간 sample_id 중복:', all_ids.duplicated().sum())
print('sample_submission ↔ test ID 일치:', set(sub.sample_id) == set(frames['test'].sample_id),
      '/ 순서 일치:', sub.sample_id.tolist() == frames['test'].sample_id.tolist())

# 2) 실험 간 분리: 같은 experiment_id가 여러 split에 있으면 누수
print('\n== 2. 실험 간 분리')
exps = {s: set(f.experiment_id) for s, f in frames.items()}
for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
    print(f'{a} ∩ {b}: {len(exps[a] & exps[b])}개 실험')
# 실험별 고정 조건(bat_class 등)이 실험 안에서 바뀌지 않는지
for s, f in frames.items():
    n = f.groupby('experiment_id')[['bat_class', 'bat_cc', 'soc_status', 'tr_method']].nunique().gt(1).sum()
    print(f'{s}: 실험 안에서 조건값이 바뀌는 실험 수', n.to_dict())

# 3) 이미지 누락·중복: 파일 존재 / 경로 중복 / 내용(md5) 중복 / 폴더에만 있는 고아 파일
print('\n== 3. 이미지 누락·중복')
hashes = []
for s, f in frames.items():
    paths = f.image_path.map(lambda p: DATA_DIR / s / p)
    missing = (~paths.map(Path.is_file)).sum()
    on_disk = {p.name for p in (DATA_DIR / s / 'images').iterdir()}
    orphan = len(on_disk - set(paths.map(lambda p: p.name)))
    print(f'{s}: 누락 {missing} / 경로 중복 {f.image_path.duplicated().sum()} / CSV에 없는 파일 {orphan}')
    hashes.append(pd.DataFrame({'split': s, 'sample_id': f.sample_id, 'experiment_id': f.experiment_id,
                                'md5': paths.map(lambda p: hashlib.md5(p.read_bytes()).hexdigest())}))
h = pd.concat(hashes)
dup = h[h.md5.duplicated(keep=False)].sort_values('md5')
print('내용이 같은 이미지:', len(dup), '장 /', dup.md5.nunique(), '그룹')
if len(dup):
    g = dup.groupby('md5').agg(splits=('split', lambda x: ','.join(sorted(set(x)))),
                               n_exp=('experiment_id', 'nunique'), n=('sample_id', 'size'))
    print(g.splits.value_counts().to_string())
    print('서로 다른 실험끼리 같은 이미지인 그룹:', (g.n_exp > 1).sum())

# 4) 결측률
print('\n== 4. 결측률 (0이 아닌 컬럼만, %)')
miss = pd.DataFrame({s: f.isna().mean() * 100 for s, f in frames.items()}).round(2)
print(miss[(miss > 0).any(axis=1)].to_string())

# 5) 시간 중복: 같은 실험 안에서 relative_time_s가 겹치는지 + 간격 분포
print('\n== 5. 시간 중복·간격')
for s, f in frames.items():
    d = f.duplicated(['experiment_id', 'relative_time_s']).sum()
    gap = f.sort_values(['experiment_id', 'relative_time_s']).groupby('experiment_id').relative_time_s.diff()
    print(f'{s}: (실험,시간) 중복 {d} / 간격 분포 {gap.value_counts().head(5).to_dict()}')
    if 'label' in f:  # 라벨이 시간 순으로 1→6 단조 증가하는지
        dec = (f.sort_values(['experiment_id', 'relative_time_s']).groupby('experiment_id').label.diff() < 0).sum()
        print(f'   라벨이 시간 순으로 감소하는 지점: {dec}')
