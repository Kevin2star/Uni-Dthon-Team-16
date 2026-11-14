# v05_lightgbm — LightGBM 교체

## 목적
로지스틱 회귀 → LightGBM (시드·스레드·deterministic 고정).

## 실행
`python compare_cv.py model_lgbm model_image model_image_lgbm` (2~3분). 그림: `cmp_dump.py` → `model_compare.png`

## 결과
| 피처 | 모델 | CV | Val |
|---|---|---|---|
| 베이스라인 | 로지스틱 | 0.803 ± 0.022 | 0.678 |
| 베이스라인 | **LightGBM** | **0.874** ± 0.028 | 0.757 |
| 열 퍼짐 | 로지스틱 | 0.800 ± 0.019 | 0.771 |
| 열 퍼짐 | **LightGBM** | **0.876** ± 0.033 | **0.775** |

## 결론
모델 교체 효과 확실 (+0.07, 5개 fold 모두 상승, 특히 S6 0.60→0.85). 열 퍼짐 피처는 LightGBM 위에서 차이 없음. **이후 기본 모델 = 열 퍼짐 + LightGBM.**

## 이 폴더의 파일
- `lgbm_common.py`
- `model_lgbm.py`
- `model_image_lgbm.py`
- `cmp_dump.py`
- `cmp.json`
- `model_compare.png`
