# common
여러 버전이 함께 쓰는 모듈의 **현재 버전** 사본입니다.

- `thermal.py`: v04 decode/spread → v09 rich → v13 diff 입력 → v15 stage 피처 순으로 함수가 추가됨 (기존 함수는 바뀌지 않음)
- `thermal_palette.npy`: 복원한 256색 컬러맵
- `lgbm_common.py`: LightGBM 설정 (v05~)
- `model_image.py`, `model_lgbm.py`, `model_image_lgbm.py`: v04~v05 모델 모듈
