"""추론 스크립트. 재학습 없이 저장된 가중치로 제출 파일을 만든다.

예) python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로>
    [--sample_submission <sample_submission.csv>]   # 주면 제출 양식의 행 순서로 저장

입력 CSV에 들어 있는 행만으로 피처를 만든다. 행 순서를 섞거나, 실험별로 나누거나,
이후 시점 행을 빼고 실행해도 같은 sample_id에는 같은 예측이 나온다.
가중치는 이 파일과 같은 폴더의 model.joblib을 쓰므로 어느 폴더에서 실행해도 된다.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

import model as M

HERE = Path(__file__).resolve().parent


def predict_frame(frame: pd.DataFrame, image_roots: list[Path], bundle: dict) -> pd.Series:
    """sample_id를 인덱스로 하는 1~6 예측."""
    paths = M.resolve_image_paths(frame, image_roots)
    img = M.extract_image_features(frame, paths)
    X, meta = M.make_dataset(frame, img)
    prob = M.predict_proba(bundle['model'], X[bundle['features']])
    labels = M.decode(prob, meta, bundle['T'], bundle['mode'], bundle['gamma'])
    return pd.Series(labels, index=meta['sample_id'].to_numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    parser.add_argument('--model', type=Path, default=HERE / 'model.joblib')
    parser.add_argument('--sample_submission', type=Path, default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame['sample_id'].is_unique, 'sample_id가 겹칩니다.'
    pred = predict_frame(frame, [args.image_root, args.image_root.parent], joblib.load(args.model))

    if args.sample_submission is not None:
        ids = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id'].astype(str).tolist()
        assert set(ids) == set(pred.index), '입력 CSV와 제출 양식의 sample_id가 다릅니다.'
    else:
        ids = frame['sample_id'].astype(str).tolist()
    submission = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].to_numpy().astype(int)})
    assert submission['label'].isin(M.LABELS).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f'저장: {args.output} ({len(submission):,}행) | 단계별 예측 수: {submission["label"].value_counts().sort_index().to_dict()}')


if __name__ == '__main__':
    main()
