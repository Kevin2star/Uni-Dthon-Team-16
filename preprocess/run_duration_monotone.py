"""단계가 뒤로 가는 예측을 올리고, 상승이 유지된 시간으로 ExtraTrees를 다시 학습합니다.

Validation Macro F1과 4·5단계 F1을 비교합니다. Test는 제출 파일을 만들 때만 읽습니다.

  python -u run_duration_monotone.py --data_dir ~/data/battery_data_light
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


def predict_bundle(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    if bundle.get('kind') == 'blend':
        weights = np.asarray(bundle['weights'], dtype=np.float64)
        weights = weights / weights.sum()
        mixed = None
        for child, weight in zip(bundle['bundles'], weights):
            proba = child['estimator'].predict_proba(features.reindex(columns=child['features']))
            mixed = weight * proba if mixed is None else mixed + weight * proba
        return (np.argmax(mixed, axis=1) + 1).astype(int)
    proba = bundle['estimator'].predict_proba(features.reindex(columns=bundle['features']))
    return (np.argmax(proba, axis=1) + 1).astype(int)


def report(name: str, y, prediction) -> dict:
    score = macro_f1(y, prediction)
    class_f1 = per_class_f1(y, prediction)
    print(f'{name:<28} {score:6.4f}  4단계 {class_f1[3]:.4f}  5단계 {class_f1[4]:.4f}')
    return {'name': name, 'val_f1': score, 'f1_4': float(class_f1[3]), 'f1_5': float(class_f1[4])}


def write_submission(ids, labels_by_id, output: Path) -> None:
    submission = pd.DataFrame({'sample_id': ids, 'label': labels_by_id.loc[ids].astype(int).to_numpy()})
    assert submission['label'].isin(M.LABELS).all()
    assert submission['sample_id'].is_unique
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f'저장: {output} ({len(submission)}행)')


def main():
    parser = argparse.ArgumentParser(description='단조 보정과 상승 유지 시간 피처를 비교합니다.')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--base_model', type=Path, default=Path('weights/extratrees_tuned.joblib'))
    parser.add_argument('--ensemble_model', type=Path, default=Path('weights/ensemble_extratrees_catboost.joblib'))
    parser.add_argument('--model_out', type=Path, default=Path('weights/extratrees_duration.joblib'))
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
    y_train = labels_of(train)
    y_val = labels_of(validation)

    print(f"{'model':<28} {'val_f1':>6}")
    rows = []
    base = joblib.load(args.base_model)
    base_pred = predict_bundle(base, val_x)
    base_mono = M.enforce_causal_monotone(validation, base_pred)
    rows.append(report('extratrees', y_val, base_pred))
    rows.append(report('extratrees+monotone', y_val, base_mono))
    print(f'  ExtraTrees 역행 보정 {int((base_pred != base_mono).sum())}행')

    if args.ensemble_model.is_file():
        ensemble = joblib.load(args.ensemble_model)
        ensemble_pred = predict_bundle(ensemble, val_x)
        ensemble_mono = M.enforce_causal_monotone(validation, ensemble_pred)
        rows.append(report('extratrees+catboost', y_val, ensemble_pred))
        rows.append(report('extratrees+catboost+monotone', y_val, ensemble_mono))
        print(f'  앙상블 역행 보정 {int((ensemble_pred != ensemble_mono).sum())}행')
    else:
        ensemble = ensemble_pred = ensemble_mono = None

    print('상승 유지 시간 피처로 ExtraTrees를 학습합니다.', flush=True)
    duration = M.ExtraTreesModel(TUNED, args.seed, args.n_jobs)
    duration.fit(train_x, y_train)
    duration_bundle = {
        'kind': 'tabular',
        'model_name': 'extratrees',
        'estimator': duration,
        'features': list(train_x.columns),
        'seed': args.seed,
        'labels': M.LABELS,
        'params': {key: ('none' if value is None else value) for key, value in TUNED.items()},
    }
    duration_pred = predict_bundle(duration_bundle, val_x)
    duration_mono = M.enforce_causal_monotone(validation, duration_pred)
    rows.append(report('duration', y_val, duration_pred))
    rows.append(report('duration+monotone', y_val, duration_mono))
    print(f'  새 모델 역행 보정 {int((duration_pred != duration_mono).sum())}행')
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(duration_bundle, args.model_out)
    print('모델 저장:', args.model_out)
    pd.DataFrame(rows).to_csv(args.model_out.with_name('duration_monotone_val.csv'), index=False)

    print('Test 피처를 만드는 중입니다.', flush=True)
    test = pd.read_csv(args.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    _, test_x = prepare(test, args.data_dir, 'test')
    test_x = test_x.reindex(columns=train_x.columns)
    sample = pd.read_csv(args.data_dir / 'sample_submission.csv', dtype={'sample_id': 'string'})
    ids = sample['sample_id'].astype(str).tolist()

    def series_of(prediction):
        return pd.Series(prediction, index=test['sample_id'].astype(str).to_numpy())

    outputs = {
        'submission_extratrees_tuned_monotone.csv': series_of(M.enforce_causal_monotone(test, predict_bundle(base, test_x))),
        'submission_extratrees_duration.csv': series_of(predict_bundle(duration_bundle, test_x)),
        'submission_extratrees_duration_monotone.csv': series_of(M.enforce_causal_monotone(test, predict_bundle(duration_bundle, test_x))),
    }
    if ensemble is not None:
        outputs['submission_ensemble_extratrees_catboost_monotone.csv'] = series_of(
            M.enforce_causal_monotone(test, predict_bundle(ensemble, test_x))
        )
    for name, values in outputs.items():
        write_submission(ids, values, Path(name))


if __name__ == '__main__':
    main()
