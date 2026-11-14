"""추론 스크립트. 저장된 가중치로만 예측하며 학습하지 않습니다.

운영진 실행 형식:
  python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

import model as M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--model', type=Path, default=Path('weights/model.joblib'))
    parser.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    args = parser.parse_args()
    try:
        import torch  # 딥러닝 가중치를 복원하려면 불러오기 전에 필요합니다.
        del torch
    except Exception:
        pass
    if not args.model.is_file():
        raise SystemExit(
            f'가중치가 없습니다: {args.model}\n'
            'tmux에서 python train.py --data_dir <데이터 경로> 를 먼저 실행하십시오. 추론은 그때도 자동으로 시작되지 않습니다.'
        )
    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique
    prediction = M.predict_labels(bundle, frame, [args.image_root, args.image_root.parent])
    if args.sample_submission is not None:
        order = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id']
        assert set(order.astype(str)) == set(prediction.index), '입력 CSV와 제출 양식의 ID가 다릅니다.'
        ids = order.astype(str).tolist()
    else:
        ids = frame['sample_id'].astype(str).tolist()
    submission = pd.DataFrame({'sample_id': ids, 'label': prediction.loc[ids].astype(int).to_numpy()})
    assert submission['label'].isin(M.LABELS).all()
    assert submission['sample_id'].is_unique and len(submission) == len(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission)}행)')


if __name__ == '__main__':
    main()
