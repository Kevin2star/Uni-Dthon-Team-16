# v14_stage_eda — 단계별 이미지 특징 탐색

## 목적
단계별 이미지를 뽑아 보고, 같은 실험 안에서 k→k+1 단계로 갈 때 일관되게 바뀌는 이미지 특징을 순위로.

## 실행
`python stage_eda.py` (6초). 이미지 1,310장: `stage_samples/` (원본 `~/yeonwoo/stage_samples` 링크)

## 결과 (경계별 최상위)
| 경계 | 바뀌는 것 | 효과 / 일치 |
|---|---|---|
| 1→2 | 엔트로피·표준편차 ↑ (노이즈 화면 → 매끈) | +1.5 / 100% |
| 2→3 | 중간 온도 면적 ↑ (천천히 넓어짐) | +0.9 / 96% |
| 3→4 | 고온 면적 ↑ (빨강 새로 생김) | +1.2 / 86% |
| 4→5 | 매우 고온 면적·덩어리 ↑ | +0.7 / 73% (**가장 약함**) |
| 5→6 | 열이 아래로, 형태 사라짐 | +0.7 / 85% |

## 결론
이미지 신호가 약한 2→3, 4→5 = 모델이 틀리던 경계. 새 피처 후보 5가지 도출 → v15.

## 이 폴더의 파일
- `stage_eda.py`
- `stage_feature_shift.csv`
- `stage_feature_shift.png`
- `stage_sheet_class1.png`
- `stage_sheet_class2.png`
- `stage_sheet_class3.png`
- `stage_samples/ (링크)`
