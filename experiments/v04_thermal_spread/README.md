# v04_thermal_spread — 컬러맵 역변환 + 열 퍼짐 피처

## 목적
이미지가 256색 컬러맵이라는 점을 이용해 색 → 0~255 온도 지수로 복원하고, 고온 면적·덩어리·중심·퍼짐 피처 13개를 만들어 베이스라인 이미지 통계를 교체.

## 코드
- `common/thermal.py`: `decode()`, `spread_features()`, `SPREAD_FEATURES`, `to_model_input()` (이 버전에서 추가)
- `common/thermal_palette.npy`: 복원한 컬러맵 (검정=차가움 → 흰색=뜨거움)
- `model_image.py`: 베이스라인 model.py에 위 피처 연결, `WINDOW_SIGNALS`에 `hot200_ratio`, `hot200_spread`, `th_p99` 추가
- `compare_cv.py`: 모듈 간 CV/Val 비교 (`python compare_cv.py model_image`)

## 결과 (로지스틱 회귀)
| | 피처 | CV | Val | Val S4 / S5 |
|---|---|---|---|---|
| 베이스라인 | 33 | 0.803 ± 0.022 | 0.678 | 0.44 / 0.50 |
| 열 퍼짐 | 43 | 0.800 ± 0.019 | **0.771** | **0.66 / 0.78** |

## 결론
Val은 크게 올랐지만 CV는 그대로 → 일부 실험에서만 효과.

## 이 폴더의 파일
- `model_image.py`
- `compare_cv.py`
