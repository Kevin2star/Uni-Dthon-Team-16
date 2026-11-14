# v13_first_frame_reference — 처음 프레임 기준 차분 (실패)

## 목적
각 실험의 0초 프레임을 빼서 실험 고유의 모습을 제거. R1 임베딩 차이 / R2 직접 피처 차이 / R3 픽셀 차이 이미지 → ConvNeXt.

## 실행
`python -u firstframe_eval.py | tee firstframe.log` (약 18분), `python viz_firstframe.py`
`common/thermal.py`: `to_model_input_diff()` (이 버전에서 추가)

## 결과
| | 실험 차이 | 이미지만 CV | 센서+확률 CV | Val |
|---|---|---|---|---|
| R0 그대로 | 61% | **0.704** | **0.873** | **0.790** |
| R1 임베딩 차이 | 74% | 0.593 | 0.859 | 0.758 |
| R2 피처 차이 | 66% | 0.634 | 0.860 | 0.738 |
| R3 픽셀 차이 | 69% | 0.523 | 0.858 | 0.761 |

## 결론
모두 R0보다 나쁨, 센서만(0.868)보다도 낮음. 원인: **이미지마다 색 범위를 따로 늘려서** 0초 프레임과 지금 프레임의 같은 색이 다른 온도 → 차분이 의미 없음. 프레임 간 비교는 절대 온도 정보(`thermal_t` 정의, 프레임별 표시 온도 범위) 확인 전까지 보류.

## 이 폴더의 파일
- `firstframe_eval.py`
- `firstframe.log`
- `viz_firstframe.py`
- `firstframe.png`
