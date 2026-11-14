"""완화된 역행 보정과 전압·가스·양극 온도의 유지 시간을 비교합니다.

기준은 공개 점수가 가장 높았던 상승 유지 시간 ExtraTrees입니다.
낮은 단계 확률이 직전 단계보다 0.05 이상 클 때만 단계를 내립니다.

  python -u run_soft_duration.py --data_dir ~/data/battery_data_light
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


def class_proba(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    return bundle['estimator'].predict_proba(features.reindex(columns=bundle['features']))


def labels_from(proba: np.ndarray) -> np.ndarray:
    return (np.argmax(proba, axis=1) + 1).astype(int)


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
    parser = argparse.ArgumentParser(description='완화된 역행 보정과 추가 유지 시간 피처를 비교합니다.')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--base_model', type=Path, default=Path('weights/extratrees_duration.joblib'))
    parser.add_argument('--model_out', type=Path, default=Path('weights/extratrees_duration_v2.joblib'))
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
    val_x = val_x.reindex(columns=train_x.columns)
    print(f'피처 수: {train_x.shape[1]}')
    y_val = labels_of(validation)
    print(f"{'model':<32} {'val_f1':>6}")

    base = joblib.load(args.base_model)
    base_proba = class_proba(base, val_x)
    base_raw = labels_from(base_proba)
    base_hard = M.enforce_causal_monotone(validation, base_raw)
    base_soft = M.enforce_soft_monotone(validation, base_proba)
    rows = [
        report('duration', y_val, base_raw),
        report('duration+hard', y_val, base_hard),
        report('duration+soft', y_val, base_soft),
    ]
    print(f'  기존 모델 하드 보정 {int((base_raw != base_hard).sum())}행, 완화 보정 {int((base_raw != base_soft).sum())}행')

    print('전압 하락, 양극 온도, 가스 유지 시간으로 ExtraTrees를 학습합니다.', flush=True)
    model = M.ExtraTreesModel(TUNED, args.seed, args.n_jobs)
    model.fit(train_x, labels_of(train))
    bundle = {
        'kind': 'tabular',
        'model_name': 'extratrees',
        'estimator': model,
        'features': list(train_x.columns),
        'seed': args.seed,
        'labels': M.LABELS,
        'params': {key: ('none' if value is None else value) for key, value in TUNED.items()},
    }
    proba = class_proba(bundle, val_x)
    raw = labels_from(proba)
    hard = M.enforce_causal_monotone(validation, raw)
    soft = M.enforce_soft_monotone(validation, proba)
    rows.extend([
        report('v2', y_val, raw),
        report('v2+hard', y_val, hard),
        report('v2+soft', y_val, soft),
    ])
    print(f'  새 모델 하드 보정 {int((raw != hard).sum())}행, 완화 보정 {int((raw != soft).sum())}행')
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print('모델 저장:', args.model_out)
    pd.DataFrame(rows).to_csv(args.model_out.with_name('duration_v2_val.csv'), index=False)

    print('Test 피처를 만드는 중입니다.', flush=True)
    test = pd.read_csv(args.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    _, test_x = prepare(test, args.data_dir, 'test')
    ids = pd.read_csv(args.data_dir / 'sample_submission.csv', dtype={'sample_id': 'string'})['sample_id'].astype(str).tolist()

    def as_series(prediction):
        return pd.Series(prediction, index=test['sample_id'].astype(str).to_numpy())

    base_test = class_proba(base, test_x)
    new_test = class_proba(bundle, test_x)
    outputs = {
        'submission_extratrees_duration_soft.csv': as_series(M.enforce_soft_monotone(test, base_test)),
        'submission_extratrees_duration_v2.csv': as_series(labels_from(new_test)),
        'submission_extratrees_duration_v2_hard.csv': as_series(M.enforce_causal_monotone(test, labels_from(new_test))),
        'submission_extratrees_duration_v2_soft.csv': as_series(M.enforce_soft_monotone(test, new_test)),
    }
    for name, values in outputs.items():
        write_submission(ids, values, Path(name))


if __name__ == '__main__':
    main()
