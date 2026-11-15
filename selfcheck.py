"""규정 자가 점검 (베이스라인의 B·C·D와 같은 취지).

  B  실험별로 나누어 따로 예측해도 전체를 한 번에 예측한 결과와 같다 (다른 실험의 행을 쓰지 않음)
  B2 실험마다 앞부분만 잘라 예측해도 그 행들의 예측이 같다 (미래 행을 쓰지 않음)
  C  입력 행 순서를 섞어도 같은 결과 (내부 정렬)
  D  이미지를 빈 이미지로 바꾸면 예측이 달라진다 (모델이 실제로 이미지를 사용)

예) python selfcheck.py --input /home/ubuntu/data/battery_data_light/validation/validation.csv \
                        --image_root /home/ubuntu/data/battery_data_light/validation --model weights/final_hold30f3r.joblib
"""
import argparse
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import model as M


def predict(frame, image_root, bundle, blank=None):
    if blank is None:
        paths = M.resolve_image_paths(frame, [image_root, image_root.parent])
    else:
        paths = pd.Series([blank] * len(frame), index=frame.index)
    th = M.extract_image_features(frame, paths)
    X, meta = M.make_dataset(frame, th, step=1)
    pred = M.decode(bundle['model'].predict_proba(X[bundle['features']]), meta, bundle['T'], bundle['mode'], bundle['gamma'])
    return pd.Series(pred, index=meta['sample_id'].to_numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--image_root', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('weights/final_hold30f3r.joblib'))
    p.add_argument('--n_exp', type=int, default=4, help='점검에 쓸 실험 수(빠르게 돌리려고 일부만)')
    a = p.parse_args()
    bundle = joblib.load(a.model)
    frame = pd.read_csv(a.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    frame = frame[frame.experiment_id.isin(sorted(frame.experiment_id.unique())[:a.n_exp])].reset_index(drop=True)
    full = predict(frame, a.image_root, bundle)

    per = pd.concat([predict(g.reset_index(drop=True), a.image_root, bundle) for _, g in frame.groupby('experiment_id')])
    print(f'B  실험별 따로 예측 = 전체 예측: {bool((per.loc[full.index] == full).all())}')

    heads = []
    for _, g in frame.groupby('experiment_id'):
        g = g.sort_values('relative_time_s')
        heads.append(g.iloc[: len(g) // 2])
    head = pd.concat(heads).reset_index(drop=True)
    ph = predict(head, a.image_root, bundle)
    print(f'B2 앞부분만 넣어도 같은 예측(미래 행 미사용): {bool((ph == full.loc[ph.index]).all())}')

    shuf = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    print(f'C  행 순서를 섞어도 같은 예측: {bool((predict(shuf, a.image_root, bundle).loc[full.index] == full).all())}')

    with tempfile.TemporaryDirectory() as d:
        blank = Path(d) / 'blank.png'
        Image.new('RGB', (151, 151), (0, 0, 0)).save(blank)
        pb = predict(frame, a.image_root, bundle, blank=blank)
    changed = int((pb.loc[full.index] != full).sum())
    print(f'D  빈 이미지로 바꾸면 예측이 바뀌는 행: {changed} / {len(full)} → 이미지 사용 {"확인" if changed > 0 else "안 됨"}')


if __name__ == '__main__':
    main()
