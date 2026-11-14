"""ExtraTrees 하이퍼파라미터 탐색.

선택 기준은 Train 실험 단위 GroupKFold Macro F1입니다.
Validation은 가장 높은 후보를 고른 뒤에 한 번만 확인합니다. Test는 읽지 않습니다.

예)
  python -u tune_extratrees.py --data_dir ~/data/battery_data_light

끝난 가중치는 test.py 가 그대로 읽습니다.
  python test.py --input $D/test/test.csv --image_root $D/test \\
      --model weights/extratrees_tuned.joblib --output submission_extratrees_tuned.csv
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
os.environ.setdefault('UNID_N_JOBS', '2')

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import model as M
from train import labels_of, macro_f1, per_class_f1, prepare, read_split

# 현재 train.py 기본값(트리 300, 깊이 제한 없음, 잎 1, sqrt, 가중치 없음)을 첫 줄에 둡니다.
CANDIDATES = [
    {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
    {'n_estimators': 600, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
    {'n_estimators': 400, 'max_depth': 24, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': None},
    {'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': None},
    {'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'class_weight': None},
    {'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 0.5, 'class_weight': None},
    {'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 2, 'max_features': 0.3, 'class_weight': None},
    {'n_estimators': 400, 'max_depth': None, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'n_estimators': 400, 'max_depth': 32, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'n_estimators': 600, 'max_depth': None, 'min_samples_leaf': 2, 'max_features': 0.5, 'class_weight': None},
]


def describe(params: dict) -> str:
    depth = 'None' if params['max_depth'] is None else str(params['max_depth'])
    weight = 'none' if params['class_weight'] is None else str(params['class_weight'])
    return (
        f"trees={params['n_estimators']} depth={depth} leaf={params['min_samples_leaf']} "
        f"features={params['max_features']} weight={weight}"
    )


def cross_validate(params: dict, features, y, groups, seed: int, folds: int, n_jobs: int):
    n_splits = min(max(2, folds), int(groups.nunique()))
    scores, class_scores = [], []
    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(features, y, groups):
        estimator = M.ExtraTreesModel(params, seed, n_jobs)
        estimator.fit(features.iloc[train_idx], y.iloc[train_idx])
        prediction = estimator.predict(features.iloc[valid_idx])
        scores.append(macro_f1(y.iloc[valid_idx], prediction))
        class_scores.append(per_class_f1(y.iloc[valid_idx], prediction))
    return float(np.mean(scores)), float(np.std(scores)), np.mean(class_scores, axis=0)


def main():
    parser = argparse.ArgumentParser(description='ExtraTrees를 Train GroupKFold Macro F1로 고릅니다.')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--model_out', type=Path, default=Path('weights/extratrees_tuned.joblib'))
    parser.add_argument('--log_out', type=Path, default=Path('weights/extratrees_tune.csv'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=M.SEED)
    parser.add_argument('--n_jobs', type=int, default=4, help='트리 학습에 쓸 프로세스 수. 전체를 쓰려면 -1')
    args = parser.parse_args()
    M.set_seed(args.seed)

    train = read_split(args.data_dir, 'train')
    validation = read_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(set(validation.experiment_id))
    print(f'Train {len(train):,}행 / {train.experiment_id.nunique()}실험, Validation {len(validation):,}행 / {validation.experiment_id.nunique()}실험')
    _, features = prepare(train, args.data_dir, 'train')
    print(f'피처 수: {features.shape[1]}')
    y = labels_of(train)
    groups = train.experiment_id.astype(str)
    print(f"{'trial':>5} {'cv_f1':>6} {'cv_std':>6} {'f1_4':>6} {'f1_5':>6}  settings")

    rows = []
    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    for trial, params in enumerate(CANDIDATES, start=1):
        cv_mean, cv_std, class_f1 = cross_validate(params, features, y, groups, args.seed, args.folds, args.n_jobs)
        row = {
            'trial': trial,
            'cv_f1': cv_mean,
            'cv_std': cv_std,
            'f1_4': float(class_f1[3]),
            'f1_5': float(class_f1[4]),
            **{key: ('none' if value is None else value) for key, value in params.items()},
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.log_out, index=False)
        print(f'{trial:5d} {cv_mean:6.4f} {cv_std:6.4f} {class_f1[3]:6.4f} {class_f1[4]:6.4f}  {describe(params)}')

    best_row = max(rows, key=lambda row: (row['cv_f1'], -row['cv_std']))
    best_params = CANDIDATES[best_row['trial'] - 1]
    print(f"선택: trial {best_row['trial']}  Train 교차검증 Macro F1 {best_row['cv_f1']:.4f}. Validation 점수는 선택에 쓰지 않았습니다.")
    print('설정:', describe(best_params))

    estimator = M.ExtraTreesModel(best_params, args.seed, args.n_jobs)
    estimator.fit(features, y)
    _, val_features = prepare(validation, args.data_dir, 'validation')
    val_pred = estimator.predict(val_features.reindex(columns=features.columns))
    val_f1 = macro_f1(labels_of(validation), val_pred)
    val_class = per_class_f1(labels_of(validation), val_pred)
    print(f'Validation Macro F1 {val_f1:.4f}  4단계 {val_class[3]:.4f}  5단계 {val_class[4]:.4f}')

    bundle = {
        'kind': 'tabular',
        'model_name': 'extratrees',
        'estimator': estimator,
        'features': list(features.columns),
        'seed': args.seed,
        'labels': M.LABELS,
        'params': {key: ('none' if value is None else value) for key, value in best_params.items()},
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    print('모델 저장:', args.model_out)
    print('후보 기록:', args.log_out)
    print('추론은 시작하지 않았습니다.')


if __name__ == '__main__':
    main()
