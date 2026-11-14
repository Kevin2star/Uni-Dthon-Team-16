"""학습 스크립트: Train으로 학습 → (선택) 실험 단위 교차검증 → Validation 확인 → 가중치 저장.

최종 제출 두 개는 같은 모델(XGBoost, Train만 학습, 시드 42)이고 순서 보정만 다르다.
  weights/final_latch.joblib      순서 보정 filter_latch
  weights/final_hold30f3r.joblib  순서 보정 filter_hold30+f3+r

예) python train.py --data_dir /home/ubuntu/data/battery_data_light            # 학습 + Validation 확인 + 저장 (GPU 약 1분)
    python train.py --data_dir /home/ubuntu/data/battery_data_light --cv       # Train 실험 단위 5-fold OOF 점수도 출력
화면에는 점수·개수만 출력한다(데이터 행·ID는 출력하지 않음).
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

import model as M

HERE = Path(__file__).resolve().parent
FINALS = {'final_latch': 'filter_latch', 'final_hold30f3r': 'filter_hold30+f3+r'}
STEP_NAME = {1: '5초', 2: '10초', 3: '15초'}


def macro_f1(y, p):
    return f1_score(y, p, labels=M.LABELS, average='macro', zero_division=0)


def image_features(frame, data_dir, split, work_dir):
    """이미지 온도 지수는 계산이 오래 걸려 work/에 저장해 둔다."""
    cache = work_dir / f'th_{split}.joblib'
    ids = set(frame['sample_id'].astype(str))
    if cache.is_file():
        th = joblib.load(cache)
        if set(th.index) == ids:
            return th
    th = M.extract_image_features(frame, M.resolve_image_paths(frame, [data_dir / split, data_dir]))
    work_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(th, cache)
    return th


def load_split(data_dir, split, work_dir):
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    th = image_features(frame, data_dir, split, work_dir)
    return {s: M.make_dataset(frame, th, step=s) for s in M.STEPS}


def stack(data, masks=None):
    Xs, ys = [], []
    for s, (X, meta) in data.items():
        keep = np.ones(len(X), bool) if masks is None else masks[s]
        Xs.append(X[keep])
        ys.append(meta['label'].to_numpy()[keep])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def cross_validate(train, device):
    meta1 = train[1][1]
    exp1 = meta1['experiment_id'].to_numpy()
    fold_of = {}
    for k, (_, b) in enumerate(GroupKFold(n_splits=5).split(exp1, groups=exp1)):
        fold_of.update({e: k for e in np.unique(exp1[b])})
    folds = {s: m['experiment_id'].map(fold_of).to_numpy() for s, (_, m) in train.items()}
    oof = {s: np.zeros((len(m), M.N_STAGES)) for s, (_, m) in train.items()}
    T_of = {}
    for k in range(5):
        X_tr, y_tr = stack(train, {s: folds[s] != k for s in M.STEPS})
        fitted = M.StageXGB(device=device).fit(X_tr, y_tr)
        for s, (X, _) in train.items():
            oof[s][folds[s] == k] = fitted.predict_proba(X[folds[s] == k])
        T = M.transition_matrix(meta1[folds[1] != k])
        T_of.update({e: T for e, f in fold_of.items() if f == k})
        print(f'  fold {k + 1}/5 완료', flush=True)
    for name, mode in FINALS.items():
        sc = {s: macro_f1(m['label'], M.decode(oof[s], m, T_of, mode)) for s, (_, m) in train.items()}
        print(f'[교차검증 OOF] {name:16s} ({mode}) | ' + ' / '.join(f'{STEP_NAME[s]} {v:.4f}' for s, v in sc.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    parser.add_argument('--work_dir', type=Path, default=HERE / 'work')
    parser.add_argument('--out_dir', type=Path, default=HERE / 'weights')
    parser.add_argument('--device', default='cuda', help="XGBoost 학습 장치: cuda(제출에 쓴 설정) 또는 cpu")
    parser.add_argument('--cv', action='store_true', help='Train 실험 단위 5-fold 교차검증 OOF 점수 출력')
    args = parser.parse_args()
    start = time.time()

    train = load_split(args.data_dir, 'train', args.work_dir)
    val = load_split(args.data_dir, 'validation', args.work_dir)
    print(f'train {len(train[1][0]):,}행 / validation {len(val[1][0]):,}행 / 피처 {train[1][0].shape[1]}개 ({time.time() - start:.0f}초)')
    if args.cv:
        cross_validate(train, args.device)

    X_tr, y_tr = stack(train)
    fitted = M.StageXGB(device=args.device).fit(X_tr, y_tr)
    T = M.transition_matrix(train[1][1])
    Xv, mv = val[1]
    pv = fitted.predict_proba(Xv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, mode in FINALS.items():
        pred = M.decode(pv, mv, T, mode)
        per = f1_score(mv['label'], pred, labels=M.LABELS, average=None, zero_division=0)
        score = macro_f1(mv['label'], pred)
        print(f'[Validation] {name:16s} ({mode}) | Macro F1 {score:.4f} | 단계별 {" ".join(f"{v:.3f}" for v in per)}')
        joblib.dump({'model': fitted, 'features': list(X_tr.columns), 'T': T, 'mode': mode, 'gamma': 1.0,
                     'val_macro_f1': float(score), 'trained_on': 'train'}, args.out_dir / f'{name}.joblib')
    print(f'가중치 저장: {args.out_dir} ({time.time() - start:.0f}초)')


if __name__ == '__main__':
    main()
