"""남은 제출 횟수 안에서 올릴 파일을 고릅니다.

지금 올릴 파일은 서로 다른 3개입니다. 이미 만든 다른 앙상블 CSV는
구성이 비슷해서 이 목록에 넣지 않습니다. 슬롯은 공개 점수를 본 뒤에 씁니다.

cross_attn 은 Validation Macro F1이 0.80 이상일 때만 제출 파일을 만듭니다.
그 미만이면 파일을 만들지 않으므로 제출 횟수를 쓰지 않습니다.

예)
  python plan_submissions.py --remaining 15

학습이 끝난 뒤에만 확인합니다.
  python plan_submissions.py --remaining 15 --data_dir $D \\
      --cross_model weights/cross_attn.joblib \\
      --test_csv $D/test/test.csv --sample_submission $D/sample_submission.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VAL_GATE = 0.80

# 공개 점수가 이미 있는 ExtraTrees를 기준으로, 성격이 다른 파일만 올립니다.
UPLOADS = [
    {
        'file': 'submission_ensemble_lightgbm_xgboost.csv',
        'role': '부스팅 조합',
        'reason': 'Validation 0.8272. LightGBM 0.60 + XGBoost 0.40',
    },
    {
        'file': 'submission_ensemble_extratrees_catboost.csv',
        'role': '교차검증 1위',
        'reason': 'Train Macro F1 0.8929. ExtraTrees 0.75 + CatBoost 0.25',
    },
    {
        'file': 'submission_extratrees_tuned.csv',
        'role': '공개 점수 기준',
        'reason': '튜닝 ExtraTrees. 이미 제출했다면 이 슬롯은 건너뜁니다.',
    },
]


def existing_uploads(root: Path) -> list[dict]:
    rows = []
    for item in UPLOADS:
        path = root / item['file']
        rows.append({**item, 'path': path, 'exists': path.is_file()})
    return rows


def maybe_cross_submission(args, remaining_after: int) -> dict | None:
    if args.cross_model is None:
        return None
    if remaining_after < 1:
        print('남은 제출 횟수가 없어 cross_attn 제출 파일을 만들지 않습니다.')
        return None
    if not args.cross_model.is_file():
        print(f'cross_attn 가중치가 없습니다: {args.cross_model}')
        print('학습은 아래 명령으로 하고, 11:00 전에 Validation이 나올 때만 이 스크립트를 다시 실행합니다.')
        print('python -u train.py --data_dir', args.data_dir, '--model cross_attn --folds 1 --epochs 6 --model_out weights/cross_attn.joblib')
        return None
    if args.data_dir is None:
        raise SystemExit('cross_attn 확인에는 --data_dir 가 필요합니다.')
    import joblib

    import model as M
    from train import labels_of, macro_f1, per_class_f1, read_split

    bundle = joblib.load(args.cross_model)
    validation = read_split(args.data_dir, 'validation')
    print('cross_attn Validation을 확인하는 중입니다.')
    prediction = M.predict_labels(bundle, validation, [args.data_dir / 'validation', args.data_dir])
    y = labels_of(validation)
    ordered = prediction.loc[validation['sample_id'].astype(str)].to_numpy()
    score = macro_f1(y, ordered)
    class_f1 = per_class_f1(y, ordered)
    record = {
        'val_f1': score,
        'f1_4': float(class_f1[3]),
        'f1_5': float(class_f1[4]),
        'gate': VAL_GATE,
        'passed': bool(score >= VAL_GATE),
    }
    args.cross_model.with_name('cross_attn_val.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f"cross_attn Validation Macro F1 {score:.4f}  4단계 {class_f1[3]:.4f}  5단계 {class_f1[4]:.4f}")
    if score < VAL_GATE:
        print(f'기준 {VAL_GATE:.2f}보다 낮아서 제출 파일을 만들지 않습니다. 남은 횟수를 쓰지 않습니다.')
        return record
    if args.test_csv is None:
        print(f'기준 {VAL_GATE:.2f} 이상입니다. --test_csv 를 주면 제출 파일 1개를 만듭니다.')
        return record
    test = pd.read_csv(args.test_csv, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    test_prediction = M.predict_labels(bundle, test, [args.test_csv.parent, args.test_csv.parent.parent])
    if args.sample_submission is not None:
        ids = pd.read_csv(args.sample_submission, dtype={'sample_id': 'string'})['sample_id'].astype(str).tolist()
    else:
        ids = test['sample_id'].astype(str).tolist()
    output = Path('submission_cross_attn.csv')
    submission = pd.DataFrame({'sample_id': ids, 'label': test_prediction.loc[ids].astype(int).to_numpy()})
    assert submission['label'].isin(M.LABELS).all()
    assert submission['sample_id'].is_unique and len(submission) == len(test)
    submission.to_csv(output, index=False)
    print(f'저장: {output} ({len(submission)}행). 이 파일만 추가로 제출하면 횟수 1회를 씁니다.')
    record['file'] = str(output)
    return record


def main():
    parser = argparse.ArgumentParser(description='남은 제출 횟수 안에서 올릴 파일을 고릅니다.')
    parser.add_argument('--remaining', type=int, default=15)
    parser.add_argument('--data_dir', type=Path, default=None)
    parser.add_argument('--cross_model', type=Path, default=None)
    parser.add_argument('--test_csv', type=Path, default=None)
    parser.add_argument('--sample_submission', type=Path, default=None)
    parser.add_argument('--plan_out', type=Path, default=Path('weights/submit_plan.csv'))
    args = parser.parse_args()
    if args.remaining < 1:
        raise SystemExit('남은 제출 횟수가 없습니다.')

    root = Path('.')
    rows = existing_uploads(root)
    print(f'남은 제출 횟수 {args.remaining}회. 지금은 {min(3, args.remaining)}개만 올리는 구성입니다.')
    print(f"{'순서':>4} {'파일':<48} {'상태':<8} 내용")
    planned = []
    for order, row in enumerate(rows, start=1):
        if order > args.remaining:
            break
        status = '있음' if row['exists'] else '없음'
        print(f'{order:4d} {row["file"]:<48} {status:<8} {row["reason"]}')
        planned.append({'order': order, 'file': row['file'], 'exists': row['exists'], 'role': row['role'], 'reason': row['reason']})
    missing = [row['file'] for row in planned if not row['exists']]
    if missing:
        print('없는 파일이 있습니다:', ', '.join(missing))
    held = max(args.remaining - len(planned), 0)
    print(f'지금 목록에 쓰지 않고 남겨 두는 횟수: {held}회.')
    print('다른 앙상블 CSV는 예측이 비슷하므로 이 횟수로 올리지 않습니다.')
    print('12:30에 공개 점수가 더 높은 파일 1개를 최종 선택합니다. 차이가 0.005 안이면 Validation이 높은 파일을 고릅니다.')

    cross = maybe_cross_submission(args, held)
    if cross is not None:
        planned.append({
            'order': len(planned) + 1,
            'file': cross.get('file', ''),
            'exists': bool(cross.get('passed')),
            'role': 'cross_attn',
            'reason': f"Validation {cross['val_f1']:.4f}, 기준 {VAL_GATE:.2f}, 통과 {cross['passed']}",
        })
    elif args.cross_model is None:
        print('cross_attn 은 아직 제출 목록에 없습니다. Validation 0.80 이상일 때만 횟수 1회를 씁니다.')
        print('python -u train.py --data_dir $D --model cross_attn --folds 1 --epochs 6 --model_out weights/cross_attn.joblib')

    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(planned).to_csv(args.plan_out, index=False)
    print('계획 저장:', args.plan_out)


if __name__ == '__main__':
    main()
