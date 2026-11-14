"""규정 자가 점검: 운영진이 코드 검증에서 하는 네 가지를 test.py와 같은 함수로 미리 돌려 본다.

예) python selfcheck.py --input /home/ubuntu/data/battery_data_light/test/test.csv \
                        --image_root /home/ubuntu/data/battery_data_light/test

  ① 행 순서를 섞어서     → 예측이 그대로여야 한다
  ② 실험별로 나눠서      → 예측이 그대로여야 한다
  ③ 이후 시점 행을 빼고  → 남은 행의 예측이 그대로여야 한다
  ④ 이미지를 빈 이미지로 → 예측이 달라져야 한다
모델이나 피처를 바꿀 때마다 실행한다. 출력은 비율뿐이다.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image

import model as M
from test import predict_frame

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=HERE / 'model.joblib')
    parser.add_argument('--work_dir', type=Path, default=HERE / 'work')
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    bundle = joblib.load(args.model)
    roots = [args.image_root, args.image_root.parent]

    def run(part):
        return predict_frame(part, roots, bundle) if len(part) else pd.Series(dtype=int)

    base = run(frame)

    def diff(pred):
        return float((base.loc[pred.index] != pred).mean())

    exps = sorted(frame['experiment_id'].unique())
    half = frame[frame['relative_time_s'] <= frame.groupby('experiment_id')['relative_time_s'].transform('median')]
    same = {
        '① 행 순서 섞기': run(frame.sample(frac=1, random_state=0)),
        '② 실험별로 나누기': pd.concat([run(frame[frame['experiment_id'].isin(exps[0::2])]),
                                  run(frame[frame['experiment_id'].isin(exps[1::2])])]),
        '③ 이후 시점 행 빼기': run(half),
    }
    failed = []
    for name, pred in same.items():
        rate = diff(pred)
        print(f'{name}: 불일치 {rate:.2%} ({len(pred):,}행)')
        if rate > 0:
            failed.append(name)

    first = M.resolve_image_paths(frame.head(1), roots).iloc[0]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    blank = (args.work_dir / 'blank.png').resolve()
    with Image.open(first) as im:
        Image.new(im.mode, im.size).save(blank)
    rate = diff(run(frame.assign(image_path=str(blank))))
    print(f'④ 빈 이미지: 예측 변화 {rate:.2%}')
    if rate == 0:
        failed.append('④ 빈 이미지(이미지가 예측에 쓰이지 않음)')

    if failed:
        print('자가 점검 실패:', failed)
        sys.exit(1)
    print('자가 점검 통과')


if __name__ == '__main__':
    main()
