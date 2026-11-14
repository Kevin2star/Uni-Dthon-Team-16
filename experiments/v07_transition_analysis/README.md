# v07_transition_analysis — 단계 전환(2→3, 4→5) 전후 분석

## 목적
정답이 2→3, 4→5로 바뀌는 지점 전후로 정답·예측 겹쳐 보기: 전환 직후에만 틀리나, 계속 틀리나.

## 실행
`python viz_transitions.py [--col "확률 경로(과거만)"]`

## 결과
- Train OOF: 전환 ±100초에서만 틀림 (2→3 전환 후 정답률 0~60s 0.69 → 300s~ 0.98)
- Validation: 늦게 따라가는 쪽으로 치우침 (2→3 직후 약 30%, 따라잡기 중앙값 35초)
- 계속 틀리는 실험: `e5bcddc7`, `31e1aaff`, `8623a013`, `f91086ae`
- **4단계를 건너뛰는 실험**(Val 2개)이 4→5 최하위, Train엔 1개뿐

## 이 폴더의 파일
- `viz_transitions.py`
- `trans_23_train_oof_model_image_lgbm.png`
- `trans_23_validation_model_image_lgbm.png`
- `trans_45_train_oof_model_image_lgbm.png`
- `trans_45_validation_model_image_lgbm.png`
- `trans_summary_model_image_lgbm.png`
