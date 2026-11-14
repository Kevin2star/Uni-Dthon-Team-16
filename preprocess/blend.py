"""이미 저장된 가중치의 클래스 확률을 평균합니다. 학습은 하지 않습니다.

같은 비중은 바로 실행할 수 있습니다. 비중 탐색과 스택은 Train 홀드아웃 확률
파일(`*_oof.npz`)이 있을 때만 하고, Test 예측으로 비중을 맞추지 않습니다.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
os.environ.setdefault('UNID_N_JOBS', '2')

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

import model as M


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0))


def load_oof(path: Path) -> pd.DataFrame:
    npz_path = path.with_name(f'{path.stem}_oof.npz')
    if not npz_path.is_file():
        raise SystemExit(
            f'홀드아웃 확률이 없습니다: {npz_path}\n'
            '표 모델은 python train.py --model <이름> --oof_only 로 만들고, '
            '딥러닝은 일반 학습이 같은 이름의 npz를 저장합니다. Test로 비중을 맞추지 않습니다.'
        )
    data = np.load(npz_path, allow_pickle=True)
    frame = pd.DataFrame(np.asarray(data['proba'], dtype=np.float64))
    frame['sample_id'] = M.as_text(data['sample_id'])
    frame['label'] = np.asarray(data['label']).astype(int)
    if 'experiment_id' not in data.files:
        raise SystemExit(f'{npz_path}에 experiment_id가 없습니다. 홀드아웃 확률을 다시 저장해야 스택을 확인할 수 있습니다.')
    frame['experiment_id'] = M.as_text(data['experiment_id'])
    return frame


def align_oof(paths: list[Path]) -> tuple[list[np.ndarray], np.ndarray]:
    tables = []
    for index, path in enumerate(paths):
        table = load_oof(path).rename(columns={column: f'{index}_{column}' for column in range(6)})
        tables.append(table)
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table.drop(columns=['label', 'experiment_id']), on='sample_id', how='inner')
    probas = []
    finite = np.ones(len(merged), dtype=bool)
    for index in range(len(paths)):
        values = merged[[f'{index}_{column}' for column in range(6)]].to_numpy(dtype=np.float64)
        finite &= np.isfinite(values).all(axis=1)
        probas.append(values)
    if not finite.any():
        raise SystemExit('홀드아웃 확률에서 모든 모델이 동시에 예측한 행이 없습니다.')
    return [values[finite] for values in probas], merged['label'].to_numpy()[finite], merged['experiment_id'].to_numpy()[finite]


def best_pair_weights(left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    best_weight, best_score = 0.5, -1.0
    for step in range(11):
        weight = step / 10
        prediction = np.argmax(weight * left + (1.0 - weight) * right, axis=1) + 1
        score = macro_f1(labels, prediction)
        if score > best_score:
            best_weight, best_score = weight, score
    return np.array([best_weight, 1.0 - best_weight]), best_score


def mix_score(probas: list[np.ndarray], weights: np.ndarray, labels: np.ndarray) -> float:
    mixed = sum(weight * proba for weight, proba in zip(weights, probas))
    return macro_f1(labels, np.argmax(mixed, axis=1) + 1)


def write_submission(bundle: dict, frame: pd.DataFrame, image_root: Path, output: Path) -> None:
    prediction = M.predict_labels(bundle, frame, [image_root, image_root.parent])
    submission = pd.DataFrame({
        'sample_id': frame['sample_id'].astype(str),
        'label': prediction.loc[frame['sample_id'].astype(str)].astype(int).to_numpy(),
    })
    assert submission['label'].isin(M.LABELS).all()
    assert submission['sample_id'].is_unique and len(submission) == len(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f'저장: {output} ({len(submission)}행)')


def main():
    parser = argparse.ArgumentParser(description='저장된 모델의 클래스 확률을 평균합니다. 학습하지 않습니다.')
    parser.add_argument('--models', type=Path, nargs='+', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--image_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model_out', type=Path, required=True)
    parser.add_argument('--weights', type=float, nargs='*', default=None)
    parser.add_argument('--fit-weights', action='store_true', help='모델이 둘일 때 Train 홀드아웃 확률로 비중을 고릅니다.')
    parser.add_argument('--stack', action='store_true', help='홀드아웃 확률 위에 로지스틱 회귀를 올립니다. 평균보다 오를 때만 저장합니다.')
    args = parser.parse_args()
    if args.fit_weights and args.stack:
        raise SystemExit('--fit-weights 와 --stack 은 한 번에 하나만 지정합니다.')
    bundles = [joblib.load(path) for path in args.models]
    weights = np.ones(len(bundles), dtype=np.float64)
    stack = None
    if args.weights:
        if len(args.weights) != len(bundles):
            raise SystemExit('--weights 개수는 --models 개수와 같아야 합니다.')
        weights = np.asarray(args.weights, dtype=np.float64)
    elif args.fit_weights:
        if len(bundles) != 2:
            raise SystemExit('비중 탐색은 모델 둘에만 씁니다. 먼저 같은 비중으로 묶어 두 개로 만든 뒤 실행하십시오.')
        probas, labels, _groups = align_oof(args.models)
        weights, score = best_pair_weights(probas[0], probas[1], labels)
        print(f'홀드아웃 비중: {weights[0]:.1f} / {weights[1]:.1f}, Macro F1 {score:.4f}')
    if args.stack:
        probas, labels, groups = align_oof(args.models)
        baseline = mix_score(probas, np.ones(len(probas)), labels)
        if len(probas) == 2:
            _pair_weights, pair_score = best_pair_weights(probas[0], probas[1], labels)
            baseline = max(baseline, pair_score)
        design = np.concatenate(probas, axis=1)
        n_splits = min(5, len(np.unique(groups)))
        if n_splits < 2:
            raise SystemExit('홀드아웃 실험이 하나뿐이라 스택 교차검증을 할 수 없습니다.')
        held = cross_val_predict(
            LogisticRegression(max_iter=1000), design, labels,
            cv=GroupKFold(n_splits=n_splits), groups=groups,
        )
        stacked = macro_f1(labels, held)
        print(f'홀드아웃 평균 Macro F1 {baseline:.4f}, 스택 교차검증 Macro F1 {stacked:.4f}')
        if stacked <= baseline:
            print('스택이 확률 평균보다 높지 않아 제출 파일을 만들지 않았습니다.')
            return
        stack = LogisticRegression(max_iter=1000)
        stack.fit(design, labels)
    blend = {'kind': 'blend', 'bundles': bundles, 'weights': weights.tolist(), 'stack': stack}
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(blend, args.model_out)
    print('조합 가중치:', args.model_out)
    frame = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    assert frame.sample_id.is_unique
    write_submission(blend, frame, args.image_root, args.output)


if __name__ == '__main__':
    main()
