"""ExtraTrees와 다른 표 모델의 클래스 확률을 섞습니다.

각 모델은 Train 실험 단위 GroupKFold로 따로 학습하고, 조합마다 홀드아웃 확률의
비중을 같은 폴드의 Macro F1로 고릅니다. 조합은 교차검증 순위와 관계없이 모두 저장합니다.
Validation은 저장된 조합마다 한 번만 확인하며 저장 대상을 바꾸지 않습니다.
Test는 --test_csv 를 줄 때만 읽고, 그때 조합마다 제출 파일을 만듭니다.

ExtraTrees는 탐색에서 고른 설정(트리 400, max_features 0.5)을 씁니다.
LightGBM, XGBoost, CatBoost, HistGradientBoosting은 train.py와 같은 설정입니다.
이미 있는 홀드아웃 확률과 전체 학습 가중치가 현재 피처와 맞으면 다시 학습하지 않습니다.

예)
  python -u train_ensemble.py --data_dir ~/data/battery_data_light

ExtraTrees와 LightGBM, XGBoost만 섞으려면
  python -u train_ensemble.py --data_dir ~/data/battery_data_light \\
      --partners lightgbm xgboost

이미 있는 ExtraTrees 쌍을 제외한 조합은 --set new 로 만듭니다.
  python -u train_ensemble.py --data_dir ~/data/battery_data_light --set new

조합마다 weights/ensemble_<모델이름>.joblib 이 생깁니다.
교차검증이 가장 높은 조합은 --model_out 에도 한 번 더 저장됩니다.
  python test.py --input $D/test/test.csv --image_root $D/test \\
      --model weights/ensemble_extratrees_lightgbm.joblib \\
      --sample_submission $D/sample_submission.csv \\
      --output submission_ensemble_extratrees_lightgbm.csv
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
os.environ.setdefault('UNID_N_JOBS', '2')

import argparse
import shutil
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import model as M
from train import labels_of, macro_f1, per_class_f1, prepare, read_split

PARTNERS = ['lightgbm', 'xgboost', 'catboost', 'histgb']
ANCHOR = 'extratrees'
# tune_extratrees.py trial 6. Train 교차검증 Macro F1 0.8915.
TUNED_EXTRATREES = {
    'n_estimators': 400,
    'max_depth': None,
    'min_samples_leaf': 1,
    'max_features': 0.5,
    'class_weight': None,
}


def make_estimator(name: str, seed: int, n_jobs: int):
    if name == ANCHOR:
        return M.ExtraTreesModel(dict(TUNED_EXTRATREES), seed, n_jobs)
    return M.TreeModel(name, None, seed, n_jobs=n_jobs)


def weight_grid(n_models: int, step: float) -> list[tuple[float, ...]]:
    ticks = int(round(1.0 / step))
    if abs(ticks * step - 1.0) > 1e-8:
        raise ValueError(f'비중 간격이 1을 나누지 못합니다: {step}')
    found: list[tuple[float, ...]] = []

    def walk(remaining: int, slots: int, prefix: tuple[float, ...]) -> None:
        if slots == 1:
            found.append(prefix + (remaining / ticks,))
            return
        for count in range(remaining + 1):
            walk(remaining - count, slots - 1, prefix + (count / ticks,))

    walk(ticks, n_models, ())
    equal = tuple(1.0 / n_models for _ in range(n_models))
    if not any(np.allclose(weights, equal) for weights in found):
        found.append(equal)
    return found


# 이미 저장해 제출 파일을 만든 ExtraTrees 쌍입니다. --set new 에서는 다시 만들지 않습니다.
SAVED_PAIRS = {
    (ANCHOR, 'lightgbm'),
    (ANCHOR, 'xgboost'),
    (ANCHOR, 'catboost'),
    (ANCHOR, 'histgb'),
}


def combo_list(partners: list[str], mode: str) -> list[tuple[str, ...]]:
    members = [ANCHOR, *partners]
    if mode == 'anchor':
        combos = [(ANCHOR, partner) for partner in partners]
        if len(partners) >= 2:
            combos.append((ANCHOR, partners[0], partners[1]))
        if len(partners) > 2:
            combos.append(tuple(members))
    else:
        combos = []
        for size in range(2, len(members) + 1):
            for combo in combinations(members, size):
                if combo in SAVED_PAIRS:
                    continue
                combos.append(combo)
    unique = []
    for combo in combos:
        if combo not in unique:
            unique.append(combo)
    return unique


def score_mixture(probas: list[np.ndarray], weights: tuple[float, ...], y: np.ndarray, folds: list[np.ndarray]):
    mixed = np.zeros((probas[0].shape[0], probas[0].shape[1]), dtype=np.float64)
    for weight, proba in zip(weights, probas):
        if weight > 0:
            mixed += weight * proba
    scores, class_scores = [], []
    for valid_idx in folds:
        prediction = np.argmax(mixed[valid_idx], axis=1) + 1
        scores.append(macro_f1(y[valid_idx], prediction))
        class_scores.append(per_class_f1(y[valid_idx], prediction))
    return float(np.mean(scores)), float(np.std(scores)), np.mean(class_scores, axis=0)


def load_oof(path: Path, sample_ids: np.ndarray, seed: int | None, folds: int | None):
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    ids = np.asarray(data['sample_id']).astype(str)
    if len(ids) != len(sample_ids) or not np.array_equal(ids, sample_ids):
        print(f'홀드아웃 행이 현재 Train과 다릅니다: {path}')
        return None
    if seed is not None and 'seed' in data.files and int(np.asarray(data['seed']).reshape(-1)[0]) != seed:
        return None
    if folds is not None and 'folds' in data.files and int(np.asarray(data['folds']).reshape(-1)[0]) != folds:
        return None
    proba = np.asarray(data['proba'], dtype=np.float64)
    if proba.shape != (len(sample_ids), len(M.LABELS)) or not np.isfinite(proba).all():
        print(f'홀드아웃 확률이 비어 있거나 크기가 다릅니다: {path}')
        return None
    return proba


def save_oof(path: Path, proba: np.ndarray, frame: pd.DataFrame, y: np.ndarray, seed: int, folds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        proba=proba,
        sample_id=M.as_text(frame['sample_id']),
        label=y,
        experiment_id=M.as_text(frame['experiment_id']),
        seed=np.array([seed]),
        folds=np.array([folds]),
    )


def cross_validate_member(name: str, features, y: pd.Series, splits, seed: int, n_jobs: int) -> np.ndarray:
    oof = np.full((len(y), len(M.LABELS)), np.nan, dtype=np.float64)
    scores = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        estimator = make_estimator(name, seed, n_jobs)
        estimator.fit(features.iloc[train_idx], y.iloc[train_idx])
        proba = estimator.predict_proba(features.iloc[valid_idx])
        oof[np.asarray(valid_idx)] = proba
        score = macro_f1(y.iloc[valid_idx], np.argmax(proba, axis=1) + 1)
        scores.append(score)
        print(f'  fold {fold}: Macro F1 {score:.4f}', flush=True)
    print(f'  평균 {float(np.mean(scores)):.4f}  표준편차 {float(np.std(scores)):.4f}', flush=True)
    if not np.isfinite(oof).all():
        raise RuntimeError(f'{name} 홀드아웃 확률에 빈 행이 있습니다.')
    return oof


def params_match(params: dict) -> bool:
    for key, value in TUNED_EXTRATREES.items():
        if params.get(key) != value:
            return False
    return True


def full_bundle(name: str, features, y: pd.Series, seed: int, n_jobs: int, weight_dir: Path, reuse: bool) -> dict:
    path = weight_dir / ('extratrees_tuned.joblib' if name == ANCHOR else f'{name}.joblib')
    if reuse and path.is_file():
        bundle = joblib.load(path)
        estimator = bundle.get('estimator')
        columns_ok = features is None or list(bundle.get('features', [])) == list(features.columns)
        params_ok = name != ANCHOR or params_match(getattr(estimator, 'params', {}))
        if columns_ok and params_ok and hasattr(estimator, 'predict_proba'):
            print(f'{name}: 저장된 전체 학습 가중치를 사용합니다. {path}')
            return bundle
        if features is None:
            raise SystemExit(f'{name}: 저장된 가중치가 현재 설정과 다릅니다. Train 피처 없이 다시 학습할 수 없습니다.')
        print(f'{name}: 저장된 가중치가 현재 피처 또는 설정과 달라 다시 학습합니다.')
    if features is None:
        raise SystemExit(f'{name}: 저장된 가중치가 없습니다. Train 피처 없이 학습할 수 없습니다.')
    print(f'{name}: Train 전체로 학습합니다.', flush=True)
    estimator = make_estimator(name, seed, n_jobs)
    estimator.fit(features, y)
    return {
        'kind': 'tabular',
        'model_name': name,
        'estimator': estimator,
        'features': list(features.columns),
        'seed': seed,
        'labels': M.LABELS,
    }


def save_blend(path: Path, names: list[str], weights: list[float], bundles: dict, cv_f1: float,
               probas: dict, frame: pd.DataFrame, y: np.ndarray, seed: int, folds: int) -> dict:
    blend = {
        'kind': 'blend',
        'bundles': [bundles[name] for name in names],
        'weights': list(weights),
        'stack': None,
        'members': list(names),
        'cv_f1': cv_f1,
        'seed': seed,
        'labels': M.LABELS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(blend, path)
    mixed = np.zeros((len(y), len(M.LABELS)), dtype=np.float64)
    for name, weight in zip(names, weights):
        mixed += weight * probas[name]
    save_oof(path.with_name(f'{path.stem}_oof.npz'), mixed, frame, y, seed, folds)
    return blend


def blend_labels(bundles: list[dict], weights: list[float], features) -> np.ndarray:
    mixed = None
    for bundle, weight in zip(bundles, weights):
        proba = bundle['estimator'].predict_proba(features.reindex(columns=bundle['features']))
        mixed = weight * proba if mixed is None else mixed + weight * proba
    return (np.argmax(mixed, axis=1) + 1).astype(int)


def positive_weights(names: tuple[str, ...], weights: tuple[float, ...]):
    kept = [(name, weight) for name, weight in zip(names, weights) if weight > 1e-8]
    total = sum(weight for _, weight in kept)
    return [name for name, _ in kept], [weight / total for _, weight in kept]


def main():
    parser = argparse.ArgumentParser(description='ExtraTrees와 다른 표 모델의 확률을 Train 홀드아웃으로 섞습니다.')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--partners', nargs='+', default=PARTNERS, choices=PARTNERS)
    parser.add_argument('--set', choices=['anchor', 'new'], default='anchor',
                        help='anchor는 ExtraTrees와 각 모델의 쌍입니다. new는 그 쌍을 제외한 나머지 조합입니다.')
    parser.add_argument('--min_weight', type=float, default=0.1,
                        help='세 모델 이상일 때 각 모델의 최소 비중. 0이 되면 이미 있는 쌍과 같아집니다.')
    parser.add_argument('--test_csv', type=Path, default=None, help='주면 저장한 조합마다 제출 파일을 만듭니다.')
    parser.add_argument('--sample_submission', type=Path, default=None)
    parser.add_argument('--model_out', type=Path, default=Path('weights/ensemble_extratrees.joblib'))
    parser.add_argument('--log_out', type=Path, default=Path('weights/ensemble_search.csv'))
    parser.add_argument('--oof_dir', type=Path, default=Path('weights/ensemble_oof'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=M.SEED)
    parser.add_argument('--n_jobs', type=int, default=4, help='한 모델이 쓰는 프로세스 수. 모델은 한 번에 하나만 학습합니다.')
    parser.add_argument('--no_reuse', action='store_true', help='저장된 홀드아웃 확률과 전체 학습 가중치를 쓰지 않고 다시 학습합니다.')
    args = parser.parse_args()
    partners = list(dict.fromkeys(args.partners))
    M.set_seed(args.seed)

    train = read_split(args.data_dir, 'train')
    validation = read_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(set(validation.experiment_id))
    print(f'Train {len(train):,}행 / {train.experiment_id.nunique()}실험, Validation {len(validation):,}행 / {validation.experiment_id.nunique()}실험')
    y_series = labels_of(train)
    y = y_series.to_numpy()
    groups = train.experiment_id.astype(str)
    n_splits = min(max(2, args.folds), int(groups.nunique()))
    sample_ids = np.asarray(train['sample_id']).astype(str)
    members = [ANCHOR, *partners]
    combos = combo_list(partners, args.set)
    print('구성:', ' + '.join(members))
    print(f"조합 {len(combos)}개 ({'기존 ExtraTrees 쌍 제외' if args.set == 'new' else 'ExtraTrees 기준'})")
    print(f'폴드 {n_splits}, 프로세스 {args.n_jobs}. 모델은 한 번에 하나만 학습합니다.')

    probas = {}
    missing = []
    reuse_external = (not args.no_reuse) and args.seed == M.SEED and n_splits == 5
    for name in members:
        cached = None if args.no_reuse else load_oof(args.oof_dir / f'{name}.npz', sample_ids, args.seed, n_splits)
        source = '이 스크립트'
        if cached is None and reuse_external and name != ANCHOR:
            cached = load_oof(args.model_out.parent / f'{name}_oof.npz', sample_ids, None, None)
            source = 'train.py'
        if cached is None:
            missing.append(name)
        else:
            probas[name] = cached
            print(f'{name}: {source}가 저장한 홀드아웃 확률을 사용합니다.')
    if missing:
        _, features = prepare(train, args.data_dir, 'train')
        print(f'피처 수: {features.shape[1]}')
        splits = list(GroupKFold(n_splits=n_splits).split(features, y, groups))
        for name in missing:
            print(f'{name}: 교차검증 학습을 시작합니다.', flush=True)
            cached = cross_validate_member(name, features, y_series, splits, args.seed, args.n_jobs)
            save_oof(args.oof_dir / f'{name}.npz', cached, train, y, args.seed, n_splits)
            probas[name] = cached
    else:
        features = None
        print('Train 피처는 다시 만들지 않습니다.')
        splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(train)), y, groups))
    fold_indices = [np.asarray(valid_idx) for _, valid_idx in splits]

    print(f"{'model':<56} {'cv_f1':>6} {'cv_std':>6} {'f1_4':>6} {'f1_5':>6}  weights")
    rows = []
    singles = {}
    for name in members:
        mean, std, class_f1 = score_mixture([probas[name]], (1.0,), y, fold_indices)
        singles[name] = mean
        print(f'{name:<56} {mean:6.4f} {std:6.4f} {class_f1[3]:6.4f} {class_f1[4]:6.4f}  1')
        rows.append({
            'combo': name, 'n_models': 1, 'weights': '1', 'cv_f1': mean, 'cv_std': std,
            'f1_4': float(class_f1[3]), 'f1_5': float(class_f1[4]), 'selected': 0,
            'path': '', 'val_f1': np.nan,
        })

    best = None
    chosen = []
    for combo in combos:
        step = 0.05 if len(combo) <= 3 else 0.1
        floor = 0.0 if len(combo) == 2 else args.min_weight
        combo_best = None
        for weights in weight_grid(len(combo), step):
            if min(weights) < floor - 1e-8 or sum(weight > 1e-8 for weight in weights) < len(combo):
                continue
            mean, std, class_f1 = score_mixture([probas[name] for name in combo], weights, y, fold_indices)
            row = {
                'combo': '+'.join(combo),
                'n_models': int(sum(weight > 1e-8 for weight in weights)),
                'weights': '/'.join(f'{weight:.2f}' for weight in weights),
                'cv_f1': mean,
                'cv_std': std,
                'f1_4': float(class_f1[3]),
                'f1_5': float(class_f1[4]),
                'selected': 0,
                'path': '',
                'val_f1': np.nan,
            }
            rows.append(row)
            rank = (mean, -std, -row['n_models'])
            if combo_best is None or rank > combo_best[0]:
                combo_best = (rank, row, combo, weights)
        assert combo_best is not None
        _, row, combo_names, weights = combo_best
        print(
            f"{row['combo']:<56} {row['cv_f1']:6.4f} {row['cv_std']:6.4f} "
            f"{row['f1_4']:6.4f} {row['f1_5']:6.4f}  {row['weights']}"
        )
        chosen.append((combo_names, weights, row))
        if best is None or combo_best[0] > best[0]:
            best = combo_best

    assert best is not None
    _, best_row, _, _ = best
    best_row['selected'] = 1
    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.log_out, index=False)
    print(f"교차검증이 가장 높은 조합: {best_row['combo']}  비중 {best_row['weights']}  Macro F1 {best_row['cv_f1']:.4f}")
    print(f'ExtraTrees 단독 교차검증 Macro F1 {singles[ANCHOR]:.4f}. 이 점수로 저장 대상을 지우지 않습니다.')
    print('후보 기록:', args.log_out)

    member_bundles = {
        name: full_bundle(name, features, y_series, args.seed, args.n_jobs, args.model_out.parent, not args.no_reuse)
        for name in members
    }
    saved = []
    seen_paths = set()
    for combo_names, weights, row in chosen:
        kept_names, kept_weights = positive_weights(combo_names, weights)
        path = args.model_out.parent / f"ensemble_{'_'.join(kept_names)}.joblib"
        if path in seen_paths or (args.set == 'new' and path.is_file()):
            row['path'] = str(path)
            print('이미 있어 건너뜁니다:', path)
            continue
        seen_paths.add(path)
        blend = save_blend(path, kept_names, kept_weights, member_bundles, row['cv_f1'], probas, train, y, args.seed, n_splits)
        row['path'] = str(path)
        saved.append((path, blend, kept_weights, row))
        print('모델 저장:', path)
    best_path = Path(best_row['path']) if best_row['path'] else None
    if args.set == 'anchor' and best_path is not None and best_path.resolve() != args.model_out.resolve():
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, args.model_out)
        oof_src = best_path.with_name(f'{best_path.stem}_oof.npz')
        if oof_src.is_file():
            shutil.copy2(oof_src, args.model_out.with_name(f'{args.model_out.stem}_oof.npz'))
        print('교차검증 1위 복사:', args.model_out)

    print('Validation 피처를 만드는 중입니다.', flush=True)
    _, val_features = prepare(validation, args.data_dir, 'validation')
    val_y = labels_of(validation)
    print(f"{'model':<56} {'cv_f1':>6} {'val_f1':>6} {'val_4':>6} {'val_5':>6}")
    for path, blend, weights, row in saved:
        val_pred = blend_labels(blend['bundles'], weights, val_features)
        val_f1 = macro_f1(val_y, val_pred)
        val_class = per_class_f1(val_y, val_pred)
        row['val_f1'] = val_f1
        print(f"{row['combo']:<56} {row['cv_f1']:6.4f} {val_f1:6.4f} {val_class[3]:6.4f} {val_class[4]:6.4f}")
    pd.DataFrame(rows).to_csv(args.log_out, index=False)
    if args.test_csv is None:
        print('Test 제출 파일은 만들지 않았습니다. --test_csv 를 주면 한 번에 만듭니다.')
        return
    test = pd.read_csv(args.test_csv, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    print('Test 피처를 만드는 중입니다.', flush=True)
    _, test_features = prepare(test, args.test_csv.parent.parent, 'test')
    if args.sample_submission is not None:
        ids = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id'].astype(str).tolist()
    else:
        ids = test['sample_id'].astype(str).tolist()
    for path, blend, weights, _row in saved:
        prediction = blend_labels(blend['bundles'], weights, test_features)
        series = pd.Series(prediction, index=test['sample_id'].astype(str).to_numpy())
        output = Path(f'submission_{path.stem}.csv')
        submission = pd.DataFrame({'sample_id': ids, 'label': series.loc[ids].astype(int).to_numpy()})
        assert submission['label'].isin(M.LABELS).all()
        assert submission['sample_id'].is_unique and len(submission) == len(test)
        submission.to_csv(output, index=False)
        print(f'저장: {output} ({len(submission)}행)')


if __name__ == '__main__':
    main()
