# Uni-DTHON 배터리 열폭주 6단계 분류 — 이미지 파트 (v1)

열화상 이미지 + 센서 시계열로 각 관측 시점의 단계(1~6)를 분류합니다. 이 저장소는 **이미지 쪽 분석·실험** 기록입니다.

> 대회 데이터(이미지·CSV)와 그 파생물(캐시, 예측 CSV, 원본 이미지가 들어간 그림)은 저장소에 없습니다. 아래 "데이터 준비"대로 각자 받아 주세요.

## 현재까지 결과 (Macro F1)
평가: Train **실험 단위 GroupKFold(5)** CV로 선택, Validation(19개 실험)은 확인용.

| 모델 | CV | Validation | 비고 |
|---|---|---|---|
| 주최 측 베이스라인 (로지스틱 회귀) | 0.803 | 0.678 | v01 |
| **센서 + 열 퍼짐 피처 + LightGBM** | **0.876** | 0.775 | v05, 현재 기본 모델 |
| 센서 + ConvNeXt 이미지 확률 6개 (스태킹) | 0.873 | **0.790** | v11, 4·5단계 개선 |
| 이미지만 (직접 만든 피처 89개) | 0.767 | 0.714 | v15, 이미지 단독 최고 |

핵심 발견
- 열화상 PNG는 **8비트 값 + 256색 컬러맵** → 색을 0~255 온도 지수로 정확히 복원 (`src/thermal_palette.npy`)
- 하지만 **이미지마다 색 범위를 따로 늘린 상대 온도**라 프레임 간 비교(차분)는 의미가 없음 (v13)
- 사전학습 CNN 임베딩은 분산의 약 60%가 '어떤 실험(배터리·구도)인가' (v12)
- 남은 난제: 2→3, 4→5 경계를 늦게 잡는 문제 (v07, v14)

실험별 상세: [`experiments/README.md`](experiments/README.md)

## 폴더 구조
```
src/            코드 (실행은 여기서)
  paths.py        데이터 위치 (환경변수 DATA_DIR)
  thermal.py      컬러맵 역변환 + 이미지 피처 (퍼짐·분포·단계 관찰)
  model_*.py      피처 조립 + 모델 (베이스라인 model.py 기반)
  *_eval.py       실험 스크립트,  viz*.py  그림
experiments/    v01~v15 실험 기록 (목적·명령·수치·결론 README + 로그 + 그래프)
outputs/viz/    결과 그래프
uni_d_baseline_v7/  주최 측 베이스라인 코드 (model.py 등, src에서 import)
cache/, outputs/*.csv, stage_samples/   ← 실행하면 생성 (저장소에 없음)
```

## 환경
```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
CNN 실험(v09~v13)은 처음 실행 때 torchvision 사전학습 가중치(ResNet18, ConvNeXt-Tiny, BSD-3)를 내려받습니다.

## 데이터 준비
대회에서 받은 `battery_data_light/`(train/ validation/ test/ sample_submission.csv)를 두고 경로를 지정합니다.
```bash
export DATA_DIR=/path/to/battery_data_light     # 기본값: /home/ubuntu/data/battery_data_light
```

## 실행 순서 (처음부터)
```bash
cd src
python data_check.py                          # 데이터 점검 (3초)
python compare_cv.py model_image_lgbm         # 이미지 피처 캐시 생성 + 베이스라인 vs 현재 기본 모델 (약 5분)
python image_repr.py --arch convnext_tiny     # CNN 임베딩 캐시 (GPU) + 비교
python -u stack_temporal.py | tee ../outputs/stack_temporal.log     # 스태킹 (약 8분)
python -u newfeat_eval.py  | tee ../outputs/newfeat.log             # 단계 관찰 피처 (약 6분)
```
각 실험의 정확한 명령은 `experiments/vXX_*/README.md`에 있습니다.

## 규정 관련 메모
- 모든 시계열 피처는 같은 실험 안의 **과거 관측만** 사용 (미래·실험 전체 통계 없음)
- 이미지 피처는 이미지마다 독립 계산, 정규화·PCA는 학습 데이터로만 fit
- LightGBM은 시드·스레드·`deterministic=True` 고정
- 확인 필요: `thermal_t`의 정의, '부분 입력' 검사의 범위 (앞부분 유지인지 임의 부분집합인지)
