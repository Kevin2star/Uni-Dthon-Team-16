# v01_baseline — 베이스라인 v7 실행 + 결과 리포트

## 목적
주최 측 베이스라인(센서 5개 + 열화상 밝기 통계 6개 + 과거 30초 시계열 → 로지스틱 회귀)을 그대로 돌려 기준점 확보.

## 실행
노트북 `uni_d_participant_baseline_v7.ipynb`의 0번 셀 `DATA_DIR`만 `/home/ubuntu/data/battery_data_light`로 바꾸고 Run All (약 3분).
리포트: `report_data.py`로 결과 JSON 추출 → `template.html`에 넣어 `baseline_report.html` 생성.

## 결과
| 지표 | 값 |
|---|---|
| Train 실험 단위 5-fold CV Macro F1 | 0.803 ± 0.022 |
| Validation Macro F1 / 정확도 | 0.678 / 0.836 |
| 규정 자가 점검 B·C·D | 모두 통과 |

- 단계별 Val F1: S1 0.99, S2 0.67, S3 0.87, **S4 0.44, S5 0.50, S6 0.60**
- 오분류 대부분이 인접 단계: 5→4 632건, 3→2 630건
- 실험 하나(`E_e5bcddc7`, Val의 15%)가 5→4 오류 564건을 차지. 이 실험을 빼면 Val 0.802 ≈ CV
- 예측이 시간상 거꾸로 가는 곳 91번

## 결론
4·5단계가 약점. Validation은 실험 하나에 크게 흔들리므로 모델 선택은 CV 기준으로.

## 이 폴더의 파일
- `uni_d_participant_baseline_v7.ipynb`
- `model.py`
- `train.py`
- `test.py`
- `submission.csv`
- `baseline_report.html`
- `report_data.py`
- `template.html`
