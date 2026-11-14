"""추론 스크립트: 저장된 가중치로 입력 CSV를 예측해 제출 파일을 만든다.

예) python test.py --input /home/ubuntu/data/battery_data_light/test/test.csv \
                   --image_root /home/ubuntu/data/battery_data_light/test \
                   --model weights/final_latch.joblib --output submissions/final_latch.csv

입력 CSV에 들어 있는 행만으로 피처를 만든다(실험마다 현재와 과거 행만 사용). 주최 측이 Test 일부만 넣거나
실험별로 나누어 실행해도 같은 샘플에는 같은 예측이 나온다.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('weights/final_latch.joblib'))
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame['sample_id'].is_unique
    paths = M.resolve_image_paths(frame, [args.image_root, args.image_root.parent])
    th = M.extract_image_features(frame, paths)
    X, meta = M.make_dataset(frame, th, step=1)

    bundle = joblib.load(args.model)
    X = X[bundle['features']]
    pred = M.decode(bundle['model'].predict_proba(X), meta, bundle['T'], bundle['mode'], bundle['gamma'])
    out = pd.Series(pred.astype(int), index=meta['sample_id'].to_numpy(), name='label')

    order = (pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
             if args.sample_submission is not None else frame['sample_id'])
    sub = pd.DataFrame({'sample_id': order.astype(str).to_numpy(), 'label': out.loc[order.astype(str)].to_numpy()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f'저장: {args.output} | {len(sub)}행 | 보정 {bundle["mode"]} | 단계 분포 {sub.label.value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
