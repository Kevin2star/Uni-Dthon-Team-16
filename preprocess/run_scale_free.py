"""용량에 묶인 절대 기준 대신, 현재까지의 최고점 대비와 프레임 안 열 퍼짐으로 학습합니다.

경과 시간과 '몇 초 전에 임계값을 넘었는가' 열은 이 모델에 넣지 않습니다.
역행 보정은 학습에 넣지 않고, 비교용 파일로만 따로 저장합니다.

  python -u run_scale_free.py --data_dir ~/data/battery_data_light
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
    """열 퍼짐이 절대 밝기가 아닌지, 최고점 비율이 미래 행을 보지 않는지 확인합니다."""
    corner = np.zeros((40, 40), dtype=np.float32)
    corner[2:8, 2:8] = 1.0
    spread_x, spread_y, _share, second, edge = M._relative_shape(corner)
    assert spread_x < 0.2 and spread_y < 0.2
    assert second == 0.0
    assert edge > 0.8
    center = np.zeros((40, 40), dtype=np.float32)
    center[16:24, 16:24] = 1.0
    assert M._relative_shape(center)[4] < 0.2
    wide = np.tile(np.linspace(0.0, 1.0, 40, dtype=np.float32), (40, 1))
    assert M._relative_shape(wide)[0] > spread_x
    flat = np.full((40, 40), 0.4, dtype=np.float32)
    assert M._relative_shape(flat)[0] == 0.0

    base = pd.DataFrame({
        'experiment_id': ['e'] * 4,
        'sample_id': ['a', 'b', 'c', 'd'],
        'relative_time_s': [0.0, 10.0, 20.0, 30.0],
        'meas_v': [10.0, 10.0, 5.0, 8.0],
        'meas_t1': [20.0, 25.0, 50.0, 40.0],
        'meas_t2': [20.0, 21.0, 30.0, 28.0],
    })
    shuffled = base.sample(frac=1, random_state=0)
    got = M.add_scale_free(shuffled).assign(sample_id=shuffled['sample_id'].to_numpy()).set_index('sample_id')
    assert np.isclose(got.loc['c', 'meas_v_frac_peak'], 0.5)
    assert np.isclose(got.loc['d', 'meas_v_frac_peak'], 0.8)
    assert np.isclose(got.loc['d', 'meas_t1_below_peak'], 10.0)
    assert np.isclose(got.loc['c', 'meas_t1_rise_frac'], 1.0)
    assert np.isclose(got.loc['d', 'meas_t1_rise_frac'], 20.0 / 30.0)
    prefix = M.add_scale_free(base.iloc[:3])
    assert np.allclose(prefix.to_numpy(), M.add_scale_free(base).iloc[:3].to_numpy(), equal_nan=True)

    rows = []
    for index, second_mark in enumerate((0.0, 30.0, 60.0, 90.0, 120.0, 150.0)):
        rows.append({
            'experiment_id': 'e',
            'sample_id': str(index),
            'relative_time_s': second_mark,
            'meas_v': 4.0 - 0.4 * index,
            'meas_a': 1.0,
            'meas_t1': 20.0 + 5.0 * index,
            'meas_t2': 20.0 + 2.0 * index,
            'meas_p': 1.0,
        })
    frame = pd.DataFrame(rows)
    image = pd.DataFrame(
        np.full((len(frame), len(M.IMAGE_STATS)), 0.2),
        columns=M.IMAGE_STATS,
        index=pd.Index(frame['sample_id'].to_numpy(), name='sample_id'),
    )
    assert M.future_cut_matches(frame, image)
    columns = M.corrected_feature_columns(M.build_features(frame, image).columns)
    assert 'relative_time_s' not in columns
    assert not any('_since' in column for column in columns)
    assert 'meas_v_frac_peak' in columns and 'img_rel_spread_x' in columns


def report(name: str, y, prediction) -> dict:
    score = macro_f1(y, prediction)
    class_f1 = per_class_f1(y, prediction)
    print(
        f'{name:<28} {score:6.4f}  '
        + '  '.join(f'{label}단계 {class_f1[label - 1]:.4f}' for label in M.LABELS)
    )
    return {
        'name': name,
        'val_f1': score,
        **{f'f1_{label}': float(class_f1[label - 1]) for label in M.LABELS},
    }


def write_submission(ids, labels_by_id, output: Path) -> None:
    submission = pd.DataFrame({'sample_id': ids, 'label': labels_by_id.loc[ids].astype(int).to_numpy()})
    assert submission['label'].isin(M.LABELS).all()
    assert submission['sample_id'].is_unique
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f'저장: {output} ({len(submission)}행)')


def fit_and_score(name, model, columns, train_x, y_train, val_x, validation, y_val, model_out: Path):
    print(f'{name} 학습을 시작합니다.', flush=True)
    model.fit(train_x, y_train)
    bundle = {
        'kind': 'tabular',
        'model_name': name,
        'estimator': model,
        'features': columns,
        'seed': model.seed,
        'labels': M.LABELS,
    }
    raw = model.predict(val_x)
    hard = M.enforce_causal_monotone(validation, raw)
    rows = [report(name, y_val, raw), report(f'{name}+hard', y_val, hard)]
    print(f'  역행 보정 {int((raw != hard).sum())}행')
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    print('모델 저장:', model_out)
    return rows, model


def main():
    self_check()
    parser = argparse.ArgumentParser(description='용량 무관 척도와 프레임 안 열 퍼짐으로 학습합니다.')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('weights/extratrees_duration.joblib'))
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
    columns = M.corrected_feature_columns(train_x.columns)
    missing = [column for column in ('meas_v_frac_peak', 'img_rel_spread_x', 'meas_t_gap') if column not in columns]
    assert not missing, missing
    print(f'피처 수: {len(columns)} (전체 {train_x.shape[1]}에서 경과시간·임계 경과초 제외)')
    y_val = labels_of(validation)
    y_train = labels_of(train)

    reference = joblib.load(args.reference)
    ref_raw = reference['estimator'].predict(val_x.reindex(columns=reference['features']))
    ref_hard = M.enforce_causal_monotone(validation, ref_raw)
    rows = [
        report('duration', y_val, ref_raw),
        report('duration+hard', y_val, ref_hard),
    ]

    train_fit = train_x.loc[:, columns]
    val_fit = val_x.reindex(columns=columns)
    et_rows, et_model = fit_and_score(
        'extratrees',
        M.ExtraTreesModel(TUNED, args.seed, args.n_jobs),
        columns, train_fit, y_train, val_fit, validation, y_val,
        Path('weights/extratrees_scale.joblib'),
    )
    lgb_rows, lgb_model = fit_and_score(
        'lightgbm',
        M.TreeModel('lightgbm', None, args.seed, args.n_jobs),
        columns, train_fit, y_train, val_fit, validation, y_val,
        Path('weights/lightgbm_scale.joblib'),
    )
    rows.extend(et_rows + lgb_rows)
    pd.DataFrame(rows).to_csv(Path('weights/scale_free_val.csv'), index=False)

    print('Test 피처를 만드는 중입니다.', flush=True)
    test = pd.read_csv(args.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    _, test_x = prepare(test, args.data_dir, 'test')
    test_x = test_x.reindex(columns=columns)
    ids = pd.read_csv(args.data_dir / 'sample_submission.csv', dtype={'sample_id': 'string'})['sample_id'].astype(str).tolist()

    def as_series(prediction):
        return pd.Series(prediction, index=test['sample_id'].astype(str).to_numpy())

    outputs = {
        'submission_extratrees_scale.csv': as_series(et_model.predict(test_x)),
        'submission_extratrees_scale_monotone.csv': as_series(M.enforce_causal_monotone(test, et_model.predict(test_x))),
        'submission_lightgbm_scale.csv': as_series(lgb_model.predict(test_x)),
        'submission_lightgbm_scale_monotone.csv': as_series(M.enforce_causal_monotone(test, lgb_model.predict(test_x))),
    }
    for name, values in outputs.items():
        write_submission(ids, values, Path(name))


if __name__ == '__main__':
    main()
