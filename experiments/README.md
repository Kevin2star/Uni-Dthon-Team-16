# 실험 기록 (이미지 담당)

각 폴더 = 실험 하나의 **기록용 사본**(코드 + 결과 + README). 폴더 안 코드는 당시 경로 기준이라 그대로는 실행되지 않습니다.

**다시 실행은 `~/yeonwoo/src/`에서** 하세요 (README의 `python xxx.py` 명령을 그대로 사용):
```
cd ~/yeonwoo/src && source ~/venv/bin/activate
python compare_cv.py model_image_lgbm
```
- 코드: `~/yeonwoo/src/`  ·  캐시: `~/yeonwoo/cache/` (work_imgrep 385MB 등)  ·  결과: `~/yeonwoo/outputs/` (그림은 `outputs/viz/`, 로그·CSV)
- `common/`: 여러 버전이 함께 쓰는 모듈의 현재 버전 (src와 같음)

평가 규칙(전 버전 공통): Train 실험 단위 GroupKFold(5) CV로 선택, Validation(19개 실험)은 확인용. Macro F1.

| 버전 | 내용 |
|---|---|
| [v01_baseline](v01_baseline/README.md) | 베이스라인 v7 실행 + 결과 리포트 |
| [v02_data_check](v02_data_check/README.md) | 데이터 연결·분리 점검 |
| [v03_visual_tool](v03_visual_tool/README.md) | 데이터 눈으로 보기 도구 |
| [v04_thermal_spread](v04_thermal_spread/README.md) | 컬러맵 역변환 + 열 퍼짐 피처 |
| [v05_lightgbm](v05_lightgbm/README.md) | LightGBM 교체 |
| [v06_monotone_postproc](v06_monotone_postproc/README.md) | 단조 후처리 비교 |
| [v07_transition_analysis](v07_transition_analysis/README.md) | 단계 전환(2→3, 4→5) 전후 분석 |
| [v08_ordinal_alarms](v08_ordinal_alarms/README.md) | "k단계 이상인가?" 경보 5개 (순서형) |
| [v09_image_representations](v09_image_representations/README.md) | 이미지 표현 비교 (분포 피처 vs CNN) |
| [v10_convnext_tiny](v10_convnext_tiny/README.md) | ConvNeXt-Tiny 백본 |
| [v11_stacking_temporal](v11_stacking_temporal/README.md) | 스태킹(이미지 확률 6개) + 임베딩 시간 변화 |
| [v12_experiment_look_diagnosis](v12_experiment_look_diagnosis/README.md) | "실험 고유의 모습" 진단 |
| [v13_first_frame_reference](v13_first_frame_reference/README.md) | 처음 프레임 기준 차분 (실패) |
| [v14_stage_eda](v14_stage_eda/README.md) | 단계별 이미지 특징 탐색 |
| [v15_stage_features](v15_stage_features/README.md) | 단계 관찰 피처 12개 추가 |

## 현재까지 요약
- 기본 모델: **열 퍼짐 피처 + LightGBM** (CV 0.876, Val 0.775)
- 이미지 쪽 유일한 개선: **스태킹(ConvNeXt 이미지 모델의 6단계 확률)** — Val +0.029, 4·5단계 개선, CV는 오차 범위
- 효과 없음: 백본 교체, 경보 5개, 단조 후처리, 임베딩 시간 변화, 처음 프레임 차분
- v15: 이미지만 0.767로 최고지만 센서와 합치면 이득 없음 (개선이 센서가 이미 완벽한 1→2에 몰림)
- 핵심 난제: 2→3, 4→5 경계 지연 / 이미지 색이 프레임마다 상대 온도
