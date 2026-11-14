"""덩어리·중심의 과거 변화와 전압/표면온도 비율로 ExtraTrees를 다시 학습합니다.

기준 피처는 공개 점수가 가장 높았던 상승 유지 시간 ExtraTrees입니다.
같은 역행 보정을 붙입니다. 문턱은 새로 고르지 않습니다.

  python -u run_spread.py --data_dir ~/data/battery_data_light
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
os.environ.setdefault('UNID_N_JOBS', '8')

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import model as M
from train import labels_of, macro_f1, per_class_f1, prepare, read_split

TUNED = {
    'n_estimators': 400,
    'max_depth': None,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'class_weight': None,
}


def self_check() -> None:
    """미래 행을 쓰지 않는지, 온도 변화가 작으면 비율이 0인지 확인합니다."""
    times = np.array([0.0, 10.0, 40.0, 70.0, 100.0])
    base = pd.DataFrame({
        'experiment_id': ['e'] * 5,
        'sample_id': [str(i) for i in range(5)],
        'relative_time_s': times,
        'meas_v': [4.0, 4.0, 3.5, 3.0, 2.0],
        'meas_t1': [20.0, 20.2, 25.0, 30.0, 40.0],
        'img_blob_max': [0.0, 0.0, 0.1, 0.2, 0.4],
        'img_cent_x': [np.nan, np.nan, 0.4, 0.5, 0.7],
        'img_cent_y': [np.nan, np.nan, 0.4, 0.4, 0.5],
    })
    signals = ['meas_v', 'meas_t1', 'img_blob_max', 'img_cent_x', 'img_cent_y']
    slopes = [signal for signal in signals if M._wants_slope(signal)]
    causal = M.add_causal_features(base, signals, slopes)
    ratio = M.voltage_per_surface(causal)
    names = M.shape_feature_names()
    assert set(names).issubset(set(causal.columns) | set(ratio.columns))
    # 40초 행: 30초 창의 비교 시점은 10초. 전압 -0.5V, 표면 온도 +4.8°C.
    assert np.isclose(ratio.loc[2, 'meas_v_per_t1_30s'], -0.5 / 4.8)
    # 10초 행은 30초 전 관측이 없고 온도 변화도 1°C 미만이라 비율은 0입니다.
    assert ratio.loc[1, 'meas_v_per_t1_30s'] == 0.0
    assert np.isclose(causal.loc[4, 'img_blob_max_from_start'], 0.4)
    assert np.isclose(causal.loc[3, 'img_cent_x_delta30s'], 0.1)
    # 시작점에 뜨거운 영역이 없으면 중심의 시작점 대비 값은 비워 둡니다.
    assert np.isnan(causal.loc[4, 'img_cent_x_from_start'])


def report(name: str, y, prediction) -> dict:
    score = macro_f1(y, prediction)
    class_f1 = per_class_f1(y, prediction)
    print(f'{name:<32} {score:6.4f}  4단계 {class_f1[3]:.4f}  5단계 {class_f1[4]:.4f}')
    return {'name': name, 'val_f1': score, 'f1_4': float(class_f1[3]), 'f1_5': float(class_f1[4])}


def write_submission(ids, labels_by_id, output: Path) -> None:
    submission = pd.DataFrame({'sample_id': ids, 'label': labels_by_id.loc[ids].astype(int).to_numpy()})
    assert submission['label'].isin(M.LABELS).all()
    assert submission['sample_id'].is_unique
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f'저장: {output} ({len(submission)}행)')


def main():
    self_check()
    parser = argparse.ArgumentParser(description='열 퍼짐 속도와 전압/표면온도 비율을 비교합니다.')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--base_model', type=Path, default=Path('weights/extratrees_duration.joblib'))
    parser.add_argument('--model_out', type=Path, default=Path('weights/extratrees_spread.joblib'))
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=M.SEED)
    args = parser.parse_args()
    M.set_seed(args.seed)

    train = read_split(args.data_dir, 'train')
    validation = read_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(set(validation.experiment_id))
    print(f'Train {len(train):,}행 / {train.experiment_id.nunique()}실험, Validation {len(validation):,}행 / {validation.experiment_id.nunique()}실험')
    _, train_x = prepare(train, args.data_dir, 'train')
    _, val_x = prepare(validation, args.data_dir, 'validation')
    base = joblib.load(args.base_model)
    extra = M.shape_feature_names()
    columns = list(base['features']) + extra
    missing = [column for column in columns if column not in train_x.columns]
    assert not missing, missing
    assert set(base['features']).isdisjoint(extra)
    train_x = train_x.loc[:, columns]
    val_x = val_x.reindex(columns=columns)
    print(f'피처 수: {train_x.shape[1]} (기존 {len(base["features"])} + 열 퍼짐 {len(extra)})')
    y_val = labels_of(validation)
    print(f"{'model':<32} {'val_f1':>6}")

    base_raw = (np.argmax(base['estimator'].predict_proba(val_x.reindex(columns=base['features'])), axis=1) + 1).astype(int)
    base_hard = M.enforce_causal_monotone(validation, base_raw)
    rows = [
        report('duration', y_val, base_raw),
        report('duration+hard', y_val, base_hard),
    ]

    print('덩어리·중심 변화와 전압/표면온도 비율로 ExtraTrees를 학습합니다.', flush=True)
    model = M.ExtraTreesModel(TUNED, args.seed, args.n_jobs)
    model.fit(train_x, labels_of(train))
    bundle = {
        'kind': 'tabular',
        'model_name': 'extratrees',
        'estimator': model,
        'features': columns,
        'seed': args.seed,
        'labels': M.LABELS,
        'params': {key: ('none' if value is None else value) for key, value in TUNED.items()},
    }
    raw = model.predict(val_x)
    hard = M.enforce_causal_monotone(validation, raw)
    rows.extend([
        report('spread', y_val, raw),
        report('spread+hard', y_val, hard),
    ])
    print(f'  새 모델 역행 보정 {int((raw != hard).sum())}행')
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print('모델 저장:', args.model_out)
    pd.DataFrame(rows).to_csv(args.model_out.with_name('spread_val.csv'), index=False)

    print('Test 피처를 만드는 중입니다.', flush=True)
    test = pd.read_csv(args.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    _, test_x = prepare(test, args.data_dir, 'test')
    test_x = test_x.reindex(columns=columns)
    ids = pd.read_csv(args.data_dir / 'sample_submission.csv', dtype={'sample_id': 'string'})['sample_id'].astype(str).tolist()

    def as_series(prediction):
        return pd.Series(prediction, index=test['sample_id'].astype(str).to_numpy())

    outputs = {
        'submission_extratrees_spread.csv': as_series(model.predict(test_x)),
        'submission_extratrees_spread_monotone.csv': as_series(M.enforce_causal_monotone(test, model.predict(test_x))),
    }
    for name, values in outputs.items():
        write_submission(ids, values, Path(name))


if __name__ == '__main__':
    main()
