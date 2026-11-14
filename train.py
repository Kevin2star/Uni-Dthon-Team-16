"""학습 스크립트: 실험 단위 교차검증(OOF) → Validation 확인 → 가중치 저장.

예) python train.py --data_dir /home/ubuntu/data/battery_data_light --debug   # 일부 실험으로 실행 확인
    python train.py --data_dir /home/ubuntu/data/battery_data_light           # 본 학습
    python train.py --data_dir /home/ubuntu/data/battery_data_light --refit_train_val   # 최종: Train+Validation

화면에는 점수·개수·피처 이름만 출력한다(데이터 행·ID는 출력하지 않음).
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

import model as M

HERE = Path(__file__).resolve().parent
STEPS = (1, 2, 3)                              # 1 = 원본(5초 간격), 2·3 = 2행·3행마다 하나만 남긴 복사본(10초·15초)
STEP_NAME = {1: '5초', 2: '10초', 3: '15초'}
STEP_WEIGHT = {1: 0.70, 2: 0.18, 3: 0.12}      # test 행의 관측 간격 비율
CANDIDATES = [('raw', 0.0)] + [(m, g) for m in ('filter', 'filter_latch') for g in (0.25, 0.5, 1.0)]

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=M.LABELS, average='macro', zero_division=0)


def image_features(frame, data_dir, split, cache):
    """이미지 통계는 계산이 오래 걸려서 work/에 저장해 둔다(값을 그대로 보존하는 joblib 형식)."""
    ids = set(frame['sample_id'].astype(str))
    if cache.is_file():
        img = joblib.load(cache)
        if list(img.columns) == M.IMAGE_STATS and set(img.index) == ids:
            return img
    paths = M.resolve_image_paths(frame, [data_dir / split, data_dir])
    img = M.extract_image_features(frame, paths)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(img, cache)
    return img


def load_split(data_dir, split, work_dir, n_exp=None):
    frame = pd.read_csv(data_dir / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    tag = ''
    if n_exp:
        keep = sorted(frame['experiment_id'].unique())[:n_exp]
        frame = frame[frame['experiment_id'].isin(keep)].reset_index(drop=True)
        tag = '_debug'
    img = image_features(frame, data_dir, split, work_dir / f'img_{split}{tag}.joblib')
    return {s: M.make_dataset(frame, img, step=s) for s in STEPS}


def stack(datasets, masks=None):
    """여러 데이터(원본 + 간격을 넓힌 복사본)를 이어 붙여 학습용 X, y를 만든다."""
    Xs, ys = [], []
    for data in datasets:
        for s, (X, meta) in data.items():
            keep = np.ones(len(X), dtype=bool) if masks is None else masks[s]
            Xs.append(X[keep])
            ys.append(meta['label'].to_numpy()[keep])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def cross_validate(data, n_estimators):
    meta1 = data[1][1]
    exp1 = meta1['experiment_id'].to_numpy()
    n_splits = min(5, len(np.unique(exp1)))
    fold_of = {}
    for k, (_, va) in enumerate(GroupKFold(n_splits=n_splits).split(exp1, groups=exp1)):
        fold_of.update({e: k for e in np.unique(exp1[va])})
    folds = {s: meta['experiment_id'].map(fold_of).to_numpy() for s, (_, meta) in data.items()}
    oof = {s: np.zeros((len(meta), M.N_STAGES)) for s, (_, meta) in data.items()}
    T_of = {}
    for k in range(n_splits):
        X_tr, y_tr = stack([data], {s: folds[s] != k for s in STEPS})
        fitted = M.build_model(n_estimators).fit(X_tr, y_tr)
        for s, (X, _) in data.items():
            oof[s][folds[s] == k] = M.predict_proba(fitted, X[folds[s] == k])
        T = M.transition_matrix(meta1[folds[1] != k])   # 전이 확률도 그 fold의 학습 실험에서만 구한다
        T_of.update({e: T for e, f in fold_of.items() if f == k})
        print(f'  fold {k + 1}/{n_splits} 완료', flush=True)
    return oof, T_of


def score_candidates(prob, data, T):
    rows = []
    for mode, gamma in CANDIDATES:
        f1 = {s: macro_f1(meta['label'], M.decode(prob[s], meta, T, mode, gamma)) for s, (_, meta) in data.items()}
        rows.append({'보정': mode, 'gamma': gamma, **{STEP_NAME[s]: f1[s] for s in STEPS},
                     '가중 평균': sum(STEP_WEIGHT[s] * f1[s] for s in STEPS)})
    return pd.DataFrame(rows)


def report(title, meta, pred):
    y = meta['label'].to_numpy()
    print(f'\n[{title}] Macro F1 {macro_f1(y, pred):.4f}')
    print(classification_report(y, pred, labels=M.LABELS, digits=4, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y, pred, labels=M.LABELS),
                       index=[f'실제 {l}' for l in M.LABELS], columns=[f'예측 {l}' for l in M.LABELS]).to_string())
    wrong = pd.Series(y != pred).groupby(meta['experiment_id'].to_numpy()).agg(['sum', 'mean'])
    wrong = wrong.sort_values('sum', ascending=False)
    total = max(int(wrong['sum'].sum()), 1)
    print(f'실험별 정확도: 최소 {1 - wrong["mean"].max():.3f} / 중앙 {1 - wrong["mean"].median():.3f} / 최대 {1 - wrong["mean"].min():.3f}'
          f' | 오답 상위 3개 실험이 전체 오답의 {wrong["sum"].head(3).sum() / total:.1%}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help='train/, validation/ 폴더가 있는 경로')
    parser.add_argument('--model_out', type=Path, default=HERE / 'model.joblib')
    parser.add_argument('--work_dir', type=Path, default=HERE / 'work')
    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--refit_train_val', action='store_true', help='최종 제출용: Train+Validation으로 다시 학습해 저장')
    parser.add_argument('--debug', action='store_true', help='실험 일부 + 적은 트리로 에러 없이 도는지만 확인')
    args = parser.parse_args()
    if args.debug:
        args.n_estimators = 30
        args.model_out = args.model_out.with_name('model_debug.joblib')
    start = time.time()

    train = load_split(args.data_dir, 'train', args.work_dir, 12 if args.debug else None)
    val = load_split(args.data_dir, 'validation', args.work_dir, 4 if args.debug else None)
    features = list(train[1][0].columns)
    print(f'train {len(train[1][0]):,}행 / validation {len(val[1][0]):,}행 / 피처 {len(features)}개 ({time.time() - start:.0f}초)')

    print('\n===== 1. Train 실험 단위 교차검증 (같은 실험은 학습·검증 한쪽에만) =====')
    oof, T_of = cross_validate(train, args.n_estimators)
    table = score_candidates(oof, train, T_of)
    print(table.round(4).to_string(index=False))
    best = table.loc[table['가중 평균'].idxmax()]
    mode, gamma = best['보정'], float(best['gamma'])
    print(f'선택: 보정={mode}, gamma={gamma} (교차검증 가중 평균 기준)')
    report('교차검증 OOF, 5초 간격', train[1][1], M.decode(oof[1], train[1][1], T_of, mode, gamma))

    print('\n===== 2. Train 전체로 학습 → Validation 확인 (선택에는 쓰지 않음) =====')
    X_tr, y_tr = stack([train])
    fitted = M.build_model(args.n_estimators).fit(X_tr, y_tr)
    T = M.transition_matrix(train[1][1])
    val_prob = {s: M.predict_proba(fitted, X) for s, (X, _) in val.items()}
    val_table = score_candidates(val_prob, val, T)
    print(val_table.round(4).to_string(index=False))
    val_pred = M.decode(val_prob[1], val[1][1], T, mode, gamma)
    report('Validation, 5초 간격', val[1][1], val_pred)

    gain = pd.Series(fitted.booster_.feature_importance(importance_type='gain'), index=features)
    gain = gain / gain.sum()
    print(f'\n피처 중요도 합계: 이미지에서 온 피처 {gain[gain.index.str.startswith("img_")].sum():.1%}'
          f' / 그 밖(센서·조건·시간) {gain[~gain.index.str.startswith("img_")].sum():.1%}')
    print('중요도 상위 30개')
    print(gain.sort_values(ascending=False).head(30).round(4).to_string())

    if args.refit_train_val:
        print('\n===== 3. 최종: Train+Validation으로 다시 학습 =====')
        X_all, y_all = stack([train, val])
        fitted = M.build_model(args.n_estimators).fit(X_all, y_all)
        T = M.transition_matrix(pd.concat([train[1][1], val[1][1]], ignore_index=True))
    joblib.dump({
        'model': fitted, 'features': features, 'T': T, 'mode': mode, 'gamma': gamma,
        'oof_macro_f1': float(best[STEP_NAME[1]]), 'val_macro_f1': float(macro_f1(val[1][1]['label'], val_pred)),
        'trained_on': 'train+validation' if args.refit_train_val else 'train',
    }, args.model_out)
    print(f'\n모델 저장: {args.model_out} ({time.time() - start:.0f}초)')


if __name__ == '__main__':
    main()
