"""후보 모델 학습 스크립트.

노트북에서 돌리지 않습니다. tmux 세션 안에서 실행합니다.

예)
  python -u train.py --data_dir ~/data/battery_data_light --model lightgbm
  python -u train.py --data_dir ~/data/battery_data_light --compare tabular

모델은 Train 실험 단위 GroupKFold Macro F1로 고릅니다.
Validation은 후보를 고른 뒤 한 번만 확인하며, 선택 기준이 아닙니다.
Test는 읽지 않습니다.
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
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import model as M


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0))


def per_class_f1(y_true, y_pred) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=M.LABELS, average=None, zero_division=0)


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    return frame.reset_index(drop=True)


def prepare(frame: pd.DataFrame, data_dir: Path, split: str):
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    print(f'이미지 통계 계산 중 ({len(frame):,}장, 스레드 {M.cpu_jobs()}개)', flush=True)
    image = M.extract_image_features(frame, paths)
    print('이미지 통계 완료', flush=True)
    features = M.build_features(frame, image)
    return paths, features


def labels_of(frame: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(frame['label'], errors='raise').astype(int)
    assert set(y).issubset(M.LABELS)
    return y


def cross_validate(name: str, features: pd.DataFrame, y: pd.Series, groups: pd.Series, class_weight, seed: int, folds: int):
    n_splits = min(max(2, folds), int(groups.nunique()))
    if n_splits < 2:
        raise ValueError('GroupKFold에는 실험이 2개 이상 필요합니다.')
    scores, class_scores = [], []
    oof = np.full((len(y), 6), np.nan, dtype=np.float32)
    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(features, y, groups):
        estimator = M.make_tabular(name, class_weight, seed)
        estimator.fit(features.iloc[train_idx], y.iloc[train_idx])
        proba = estimator.predict_proba(features.iloc[valid_idx])
        oof[valid_idx] = proba
        prediction = (np.argmax(proba, axis=1) + 1).astype(int)
        scores.append(macro_f1(y.iloc[valid_idx], prediction))
        class_scores.append(per_class_f1(y.iloc[valid_idx], prediction))
    return float(np.mean(scores)), float(np.std(scores)), np.mean(class_scores, axis=0), oof


def save_bundle(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print('모델 저장:', path)


def tabular_bundle(name: str, estimator, features: pd.DataFrame, seed: int) -> dict:
    return {
        'kind': 'tabular',
        'model_name': name,
        'estimator': estimator,
        'features': list(features.columns),
        'seed': seed,
        'labels': M.LABELS,
    }


def fit_torch(name: str, frame: pd.DataFrame, paths: pd.Series, features: pd.DataFrame, y: pd.Series,
              groups: pd.Series, args, seed: int):
    import torch
    from torch import nn

    M.set_seed(seed)
    spec = M.deep_config(name)
    config = {
        'model_name': name,
        'backbone': spec['backbone'],
        'sequence': spec['sequence'],
        'use_gru': spec['sequence'] == 'gru',
        'use_image': spec['image'],
        'frames': spec['frames'],
        'head': args.head,
        'tabular_dim': 0 if not spec['image'] else int(features.shape[1]),
        'sensor_dim': 0,
        'freeze_backbone': spec['backbone'] not in (None, 'smallcnn') and not args.unfreeze,
        'image_size': M.IMAGE_SIZE,
    }
    sequence_cols = M.sequence_columns(frame) if spec['sequence'] != 'none' else []
    config['sensor_dim'] = len(sequence_cols)
    folds = args.folds if args.folds is not None else 1
    splits = list(GroupKFold(n_splits=min(5, int(groups.nunique()))).split(features, y, groups))
    chosen = splits[:min(folds, len(splits))]
    holdout_scores, chosen_epochs, class_scores = [], [], []
    oof = np.full((len(frame), 6), np.nan, dtype=np.float32)
    for fold, (train_idx, holdout_idx) in enumerate(chosen, start=1):
        epochs, score, class_f1, holdout_proba = _train_one_torch(
            config, frame, paths, features, y, sequence_cols, train_idx, holdout_idx, args, seed + fold,
        )
        holdout_scores.append(score)
        chosen_epochs.append(epochs)
        class_scores.append(class_f1)
        oof[np.asarray(holdout_idx)] = holdout_proba
        print(f'  fold {fold}: holdout Macro F1 {score:.4f}, 선택 epoch {epochs}')
    final_epochs = max(1, int(np.median(chosen_epochs)))
    network, tabular_prep, sequence_prep = _train_final_torch(
        config, frame, paths, features, y, sequence_cols, final_epochs, args, seed,
    )
    bundle = {
        'kind': 'torch',
        'model_name': name,
        'features': [] if not spec['image'] else list(features.columns),
        'sequence_cols': sequence_cols,
        'tabular_prep': tabular_prep,
        'sequence_prep': sequence_prep,
        'state_dict': {key: value.detach().cpu() for key, value in network.state_dict().items()},
        'config': config,
        'seed': seed,
        'labels': M.LABELS,
        'pretrained': M.PRETRAINED_MODELS.get(name, '없음'),
    }
    del torch, nn
    std = float(np.std(holdout_scores)) if len(holdout_scores) > 1 else float('nan')
    return bundle, float(np.mean(holdout_scores)), std, np.mean(class_scores, axis=0), oof


_IMAGE_LOADER = None


def _image_loader() -> ThreadPoolExecutor:
    global _IMAGE_LOADER
    if _IMAGE_LOADER is None:
        _IMAGE_LOADER = ThreadPoolExecutor(max_workers=M.cpu_jobs())
    return _IMAGE_LOADER


def _batch_images(paths: pd.Series, indices, normalize: str, augment: bool, frame_index=None):
    import torch
    jobs = []
    width = 1 if frame_index is None else frame_index.shape[1]
    for index in indices:
        flip = bool(augment and np.random.random() < 0.5)
        for slot in range(width):
            source = int(index) if frame_index is None else int(frame_index[int(index), slot])
            jobs.append((source, flip))
    tensors = list(_image_loader().map(
        lambda job: M.load_image_tensor(paths.iloc[job[0]], normalize=normalize, flip=job[1]),
        jobs,
    ))
    if frame_index is None:
        return torch.stack(tensors)
    return torch.stack([torch.stack(tensors[i:i + width]) for i in range(0, len(tensors), width)])


def _run_epoch(network, optimizer, indices, paths, tabular, sequences, mask, y, config, batch_size, device, train_mode, frame_index=None):
    import torch
    from torch import nn
    network.train(train_mode)
    if train_mode and getattr(network, 'freeze_backbone', False) and getattr(network, 'backbone', None) is not None:
        network.backbone.eval()
    total_loss, seen = 0.0, 0
    order = np.array(indices)
    if train_mode:
        order = np.random.permutation(order)
    predictions = np.zeros(len(y), dtype=int)
    proba = np.zeros((len(y), 6), dtype=np.float32)
    use_image = bool(config.get('use_image', True))
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        images = table = seq = mask_batch = None
        if use_image:
            images = _batch_images(paths, batch, config['normalize'], train_mode, frame_index).to(device)
            table = torch.from_numpy(tabular[batch]).to(device)
        target = torch.from_numpy(y[batch].astype(np.int64)).to(device)
        if sequences is not None:
            seq = torch.from_numpy(sequences[batch]).to(device)
            mask_batch = torch.from_numpy(mask[batch]).to(device)
        logits = network(images, table, seq, mask_batch)
        if config['head'] == 'coral':
            levels = torch.stack([(target > level).float() for level in range(1, 6)], dim=1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, levels)
        else:
            loss = nn.functional.cross_entropy(logits, target - 1)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * len(batch)
        seen += len(batch)
        batch_proba = M.logits_to_proba(logits.detach(), config['head']).cpu().numpy()
        proba[batch] = batch_proba
        predictions[batch] = batch_proba.argmax(axis=1) + 1
    return total_loss / max(seen, 1), predictions, proba


def _train_one_torch(config, frame, paths, features, y, sequence_cols, train_idx, holdout_idx, args, seed):
    import torch
    M.limit_torch_threads()
    M.set_seed(seed)
    network, normalize = M.build_fusion(config, pretrained=True)
    config['normalize'] = normalize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    if config.get('use_image', True):
        tabular_prep = M.TabularPrep().fit(features.iloc[train_idx])
        tabular = tabular_prep.transform(features).astype(np.float32)
    else:
        tabular_prep = None
        tabular = None
    sequences = mask = None
    if config.get('sequence', 'none') != 'none':
        sequence_prep = M.TabularPrep().fit(M._sensor_matrix(frame.iloc[train_idx], sequence_cols))
        sensor = sequence_prep.transform(M._sensor_matrix(frame, sequence_cols)).astype(np.float32)
        sequences, mask = M.build_sequences(frame, sensor)
    frame_index = M.causal_frame_rows(frame) if int(config.get('frames', 1)) > 1 else None
    head_lr = 1e-4 if args.unfreeze else 1e-3
    groups = [{'params': [p for n, p in network.named_parameters() if p.requires_grad and not n.startswith('backbone.')], 'lr': head_lr}]
    backbone_params = [p for n, p in network.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    y_np = y.to_numpy()
    best_score, best_epoch, wait, best_state = -1.0, 1, 0, None
    best_class = np.zeros(6)
    best_proba = None
    for epoch in range(1, args.epochs + 1):
        print(f'  epoch {epoch}/{args.epochs} 학습', flush=True)
        _run_epoch(network, optimizer, train_idx, paths, tabular, sequences, mask, y_np, config, args.batch_size, device, True, frame_index)
        _, prediction, proba = _run_epoch(
            network, optimizer, holdout_idx, paths, tabular, sequences, mask, y_np, config, args.batch_size, device, False, frame_index,
        )
        holdout_pred = prediction[np.asarray(holdout_idx)]
        score = macro_f1(y.iloc[holdout_idx], holdout_pred)
        if score > best_score:
            best_score, best_epoch, wait = score, epoch, 0
            best_class = per_class_f1(y.iloc[holdout_idx], holdout_pred)
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            best_proba = proba[np.asarray(holdout_idx)].copy()
        else:
            wait += 1
            if wait >= 2:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    return best_epoch, best_score, best_class, best_proba


def _train_final_torch(config, frame, paths, features, y, sequence_cols, epochs, args, seed):
    import torch
    M.limit_torch_threads()
    M.set_seed(seed)
    network, normalize = M.build_fusion(config, pretrained=True)
    config['normalize'] = normalize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    if config.get('use_image', True):
        tabular_prep = M.TabularPrep().fit(features)
        tabular = tabular_prep.transform(features).astype(np.float32)
    else:
        tabular_prep = None
        tabular = None
    sequence_prep = None
    sequences = mask = None
    if config.get('sequence', 'none') != 'none':
        sequence_prep = M.TabularPrep().fit(M._sensor_matrix(frame, sequence_cols))
        sensor = sequence_prep.transform(M._sensor_matrix(frame, sequence_cols)).astype(np.float32)
        sequences, mask = M.build_sequences(frame, sensor)
    frame_index = M.causal_frame_rows(frame) if int(config.get('frames', 1)) > 1 else None
    head_lr = 1e-4 if args.unfreeze else 1e-3
    groups = [{'params': [p for n, p in network.named_parameters() if p.requires_grad and not n.startswith('backbone.')], 'lr': head_lr}]
    backbone_params = [p for n, p in network.named_parameters() if p.requires_grad and n.startswith('backbone.')]
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    indices = np.arange(len(frame))
    print(f'전체 재학습 {epochs} epoch', flush=True)
    for epoch in range(1, epochs + 1):
        print(f'  재학습 {epoch}/{epochs}', flush=True)
        _run_epoch(network, optimizer, indices, paths, tabular, sequences, mask, y.to_numpy(), config, args.batch_size, device, True, frame_index)
    network.eval()
    return network, tabular_prep, sequence_prep


def score_validation(bundle, frame, data_dir: Path) -> tuple[float, np.ndarray]:
    paths_root = [data_dir / 'validation', data_dir]
    prediction = M.predict_labels(bundle, frame, paths_root)
    y = labels_of(frame)
    ordered = prediction.loc[frame['sample_id'].astype(str)].to_numpy()
    return macro_f1(y, ordered), per_class_f1(y, ordered)


def format_row(name, cv_mean, cv_std, class_f1, val_f1) -> str:
    std = '   -  ' if cv_std != cv_std else f'{cv_std:6.4f}'
    val = '   -  ' if val_f1 != val_f1 else f'{val_f1:6.4f}'
    return f'{name:<20} {cv_mean:6.4f} {std} {class_f1[3]:6.4f} {class_f1[4]:6.4f} {val}'


def train_tabular(name, features, y, groups, class_weight, seed, folds):
    cv_mean, cv_std, class_f1, oof = cross_validate(name, features, y, groups, class_weight, seed, folds)
    estimator = M.make_tabular(name, class_weight, seed)
    estimator.fit(features, y)
    return tabular_bundle(name, estimator, features, seed), cv_mean, cv_std, class_f1, oof


def main():
    parser = argparse.ArgumentParser(description='tmux에서 실행하는 후보 모델 학습')
    parser.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 이 있는 battery_data_light 경로')
    parser.add_argument('--model', choices=M.ALL_MODELS, default='lightgbm')
    parser.add_argument('--compare', choices=['tabular', 'deep'], default=None, help='후보를 교차검증으로 비교하고 CV가 가장 높은 모델을 저장')
    parser.add_argument('--model_out', type=Path, default=Path('weights/model.joblib'))
    parser.add_argument('--refit_train_val', action='store_true', help='선택 후 Train+Validation으로 다시 학습. 이때 Validation 점수는 성능 숫자가 아님')
    parser.add_argument('--class_weight', choices=['none', 'balanced'], default='none')
    parser.add_argument('--folds', type=int, default=None, help='표 모델 기본 5, 딥러닝 기본 1')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--head', choices=['ce', 'coral'], default='ce')
    parser.add_argument('--unfreeze', action='store_true', help='사전학습 합성곱망의 마지막이 아닌 전체 가중치를 학습')
    parser.add_argument('--seed', type=int, default=M.SEED)
    parser.add_argument('--oof_only', action='store_true', help='표 모델의 Train 교차검증 확률만 저장하고 기존 가중치는 덮어쓰지 않음')
    args = parser.parse_args()
    class_weight = None if args.class_weight == 'none' else 'balanced'
    if args.oof_only and (args.compare is not None or args.model in M.DEEP_MODELS):
        raise SystemExit('--oof_only 는 표 모델 하나에만 씁니다. 예: --model lightgbm --oof_only')
    M.set_seed(args.seed)

    train = read_split(args.data_dir, 'train')
    validation = read_split(args.data_dir, 'validation')
    assert set(train.experiment_id).isdisjoint(set(validation.experiment_id))
    print(f'Train {len(train):,}행 / {train.experiment_id.nunique()}실험, Validation {len(validation):,}행 / {validation.experiment_id.nunique()}실험')
    image_free = args.model in M.DEEP_MODELS and not M.deep_config(args.model)['image'] and args.compare is None
    if image_free:
        features = pd.DataFrame(np.zeros((len(train), 1)), columns=['unused'])
        print('피처 수: 0. 과거 300초 센서 시퀀스만 사용합니다.')
    else:
        _, features = prepare(train, args.data_dir, 'train')
        print(f'피처 수: {features.shape[1]}')
    y = labels_of(train)
    groups = train.experiment_id.astype(str)

    names = M.TABULAR_MODELS if args.compare == 'tabular' else M.COMPARE_DEEP_MODELS if args.compare == 'deep' else [args.model]
    print(f"{'model':<20} {'cv_f1':>6} {'cv_std':>6} {'f1_4':>6} {'f1_5':>6} {'val_f1':>6}")
    rows = []
    weight_dir = args.model_out.parent
    for name in names:
        try:
            train_paths = None
            if name in M.DEEP_MODELS and M.deep_config(name)['image']:
                train_paths = M.resolve_image_paths(train, [args.data_dir / 'train', args.data_dir])
            if name in M.DEEP_MODELS:
                bundle, cv_mean, cv_std, class_f1, oof = fit_torch(name, train, train_paths, features, y, groups, args, args.seed)
            else:
                folds = args.folds if args.folds is not None else 5
                if args.oof_only:
                    cv_mean, cv_std, class_f1, oof = cross_validate(name, features, y, groups, class_weight, args.seed, folds)
                    oof_path = weight_dir / f'{name}_oof.npz'
                    np.savez_compressed(
                        oof_path, proba=oof, sample_id=train['sample_id'].astype(str).to_numpy(),
                        label=y.to_numpy(), experiment_id=train['experiment_id'].astype(str).to_numpy(),
                    )
                    print('홀드아웃 확률:', oof_path)
                    print(format_row(name, cv_mean, cv_std, class_f1, float('nan')))
                    continue
                bundle, cv_mean, cv_std, class_f1, oof = train_tabular(name, features, y, groups, class_weight, args.seed, folds)
            stem = name
            if name in M.DEEP_MODELS and args.head == 'coral':
                stem += '_coral'
            if args.seed != M.SEED:
                stem += f'_seed{args.seed}'
            candidate = weight_dir / f'{stem}.joblib'
            save_bundle(candidate, bundle)
            if oof is not None:
                oof_path = candidate.with_name(f'{stem}_oof.npz')
                np.savez_compressed(
                    oof_path, proba=oof, sample_id=train['sample_id'].astype(str).to_numpy(),
                    label=y.to_numpy(), experiment_id=train['experiment_id'].astype(str).to_numpy(),
                )
                print('홀드아웃 확률:', oof_path)
            val_f1, _ = score_validation(bundle, validation, args.data_dir)
            print(format_row(name, cv_mean, cv_std, class_f1, val_f1))
            rows.append({'name': name, 'cv': cv_mean, 'path': candidate, 'bundle': bundle, 'val': val_f1})
        except Exception as error:
            print(f'{name:<20} 건너뜀: {type(error).__name__}: {error}')
            traceback.print_exc()
    if args.oof_only:
        print('기존 가중치는 그대로 두었습니다. 추론은 시작하지 않았습니다.')
        return
    if not rows:
        raise SystemExit('저장된 모델이 없습니다. 위 오류를 확인하십시오.')
    best = max(rows, key=lambda row: row['cv'])
    print(f"선택: {best['name']} (Train 교차검증 Macro F1 {best['cv']:.4f}). Validation 점수는 선택에 쓰지 않았습니다.")
    if args.refit_train_val:
        if best['name'] in M.DEEP_MODELS:
            raise SystemExit('딥러닝은 Train 홀드아웃으로만 학습합니다. --refit_train_val 은 표 모델에 사용하십시오.')
        print('Train+Validation으로 다시 학습합니다. 이 재학습 모델의 Validation 점수는 성능 숫자가 아닙니다.')
        feature_parts, label_parts = [], []
        for split, part in (('train', train), ('validation', validation)):
            _, part_features = prepare(part, args.data_dir, split)
            feature_parts.append(part_features)
            label_parts.append(labels_of(part))
        both_x = pd.concat(feature_parts, ignore_index=True).reindex(columns=features.columns)
        estimator = M.make_tabular(best['name'], class_weight, args.seed)
        estimator.fit(both_x, pd.concat(label_parts, ignore_index=True))
        best['bundle'] = tabular_bundle(best['name'], estimator, both_x, args.seed)
        save_bundle(args.model_out, best['bundle'])
    elif args.model_out != best['path']:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best['path'], args.model_out)
        print('최종 가중치:', args.model_out)
    print('추론은 시작하지 않았습니다. tmux에서 python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로> 를 실행하십시오.')


if __name__ == '__main__':
    main()
