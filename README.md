# Uni-D 배터리 열폭주 6단계 분류 — 최종 제출

## 팀 정보
- 팀명: 열받은사람들
- 최종 선택 제출 파일: `submissions/final_hold30f3r.csv` (가중치 `weights/final_hold30f3r.joblib`), Private 복원 대상
- 모델: XGBoost + 순서 보정 `filter_hold30+f3+r`

| 제출 | 가중치 | 순서 보정 | Train 교차검증 OOF Macro F1 (5초) | Validation Macro F1 | Public |
|---|---|---|---|---|---|
| final_hold30f3r | `weights/final_hold30f3r.joblib` | `filter_hold30+f3+r` | 0.9073 | 0.9218 | 0.8783 |

## 개발 환경
- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- GPU: 대회 제공 GPU 서버 (NVIDIA GeForce RTX 5080). XGBoost 학습에 GPU hist를 씁니다.
- 라이브러리: `requirements.txt` 참고 (확인한 버전도 적어 두었습니다)

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 피처 생성(센서·실험 조건·과거 방향 시계열·열화상 온도 지수), 모델(XGBoost), 순서 보정 |
| `thermal_palette.npy` | 열화상 256색 → 0~255 온도 지수 역변환표 (`model.py`가 읽음) |
| `train.py` | Train으로 학습, (선택) 실험 단위 교차검증, Validation 평가, 가중치 저장 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성 |
| `check_submission.py` | 운영진 제공 제출 전 점검 (`test.py`를 7번 실행해 형식·반복 실행·최종 파일 재현·행 순서·실험 분리·미래 행·이미지 사용 확인) |
| `selfcheck.py` | 규정 자가 점검 (실험 분리·미래 행 미사용·행 순서·이미지 사용) |
| `weights/final_hold30f3r.joblib` | 학습된 가중치 (3.7MB). 재학습 없이 `test.py`만으로 결과를 재현합니다. |
| `submissions/final_hold30f3r.csv` | 위 가중치로 만든 Test 제출 파일 (최종 선택) |

## 실행 방법
```bash
pip install -r requirements.txt
D=<battery_data_light 경로>

# 학습 (Train만 사용, GPU 약 30초. --cv 를 주면 Train 실험 단위 5-fold OOF 점수도 출력)
python train.py --data_dir $D --cv

# 추론 (--model 을 빼도 weights/final_hold30f3r.joblib 사용)
python test.py --input $D/test/test.csv --image_root $D/test --model weights/final_hold30f3r.joblib --output submissions/final_hold30f3r.csv
# --sample_submission $D/sample_submission.csv 를 주면 제출 양식의 행 순서로 저장

# 제출 전 점검 (운영진 제공)
python check_submission.py --data_dir $D --final submissions/final_hold30f3r.csv

# 규정 자가 점검
python selfcheck.py --input $D/validation/validation.csv --image_root $D/validation --model weights/final_hold30f3r.joblib
```
재현 확인: 이 폴더에서 `train.py` → `test.py`를 다시 실행한 결과가 리더보드에 올린 최종 파일과 3,451행 모두 같습니다.

## 학습·검증 구성
- 학습 데이터: **Train만** (78개 실험). 원본(5초)에 2행·3행마다 하나만 남긴 복사본(10초·15초)을 더해 학습합니다. Test에 10초·15초 간격 실험이 섞여 있기 때문입니다.
- 검증 방법: Train 실험 단위 GroupKFold(5) + 제공 Validation(19개 실험). 같은 실험의 행은 학습·검증 한쪽에만 들어갑니다.
- 모델 선택: 실험 단위 검증(Train 교차검증 + Validation)을 주 기준으로, Public 점수는 참고로 썼습니다.
- 검증 점수: 위 표 참고. 단계별 Validation F1 (final_hold30f3r): 1.000 / 0.815 / 0.928 / 0.926 / 0.956 / 0.905

## 피처 구성 (총 269개)
- 현재 값: 센서 17개(전압·온도·압력·`thermal_t`·`atmp_t`·`indu_t`·가스 10종), 실험 조건 4개(`bat_class`, `bat_cc`, `soc_status`, `tr_method`), 파생 3개(가스 합, 두 온도 중 높은 값, 두 온도 차)
- 열화상: 컬러맵을 온도 순서로 되돌린 **온도 지수의 프레임 평균(th_mean) 하나**. PNG는 정확히 256색 컬러맵이고, 프레임마다 범위가 자동으로 맞춰져 있어 상대 온도입니다. 이미지 통계를 여럿 넣는 것보다 의미가 분명한 신호 하나만 쓰는 쪽이 교차검증·Validation 모두에서 나았습니다.
- 과거 방향 시계열 (신호 9개 × 27종): 같은 `experiment_id` 안에서 `relative_time_s` 기준 현재와 과거 행만 사용. 첫 관측 대비 변화, 지금까지 최고·최저값과 그로부터의 거리, 최고점 이후 경과 시간, 직전 대비 초당 변화, 과거 30·60·120·300·600초 창의 평균·최대·최소·초당 변화.
- 미래 시점 관측과 실험 전체 기준 통계(전체 평균·최댓값·길이·순위)는 사용하지 않습니다.

## 모델과 순서 보정
- 모델: XGBoost (트리 300, 학습률 0.05, 깊이 6, subsample 0.8, colsample 0.5, 시드 42). 단계별 행 수 차이는 행 가중치(`balanced`)로 맞춥니다. Macro F1은 6단계를 같은 비중으로 보기 때문입니다.
- 순서 보정: 단계는 앞으로만 진행하므로, 학습 라벨로 구한 단계 전이 확률과 지금까지의 예측으로 현재 단계를 고칩니다(순방향 필터, 과거만 사용).
  - `filter_hold30+f3+r`: 30초 이상 이어진 단계만 확정, 4단계 이상 전환은 빠른 필터로 앞당김(`+f3`), 온도·전압 규칙으로 가능한 단계 범위 제한(`+r`: `meas_t1` 45도 전 1단계, 50도 전 2단계 이하, 가열 실험의 전압 급락 후 4단계 이상)
- 순서 보정 코드는 팀원(jangminjun) 구현을 그대로 사용했습니다.

## 규정 자가 점검 (`selfcheck.py`, Validation 4개 실험)
| 항목 | 결과 |
|---|---|
| 실험별로 따로 예측해도 같은 결과 | 통과 |
| 실험 앞부분만 넣어도 같은 예측 (미래 행 미사용) | 통과 |
| 행 순서를 섞어도 같은 결과 | 통과 |
| 빈 이미지로 바꾸면 예측이 바뀜 (이미지 사용) | 84 / 2,190행 변경 |

## 제출 전 점검 (`check_submission.py`, Test 전체, final_hold30f3r)
| 항목 | 결과 |
|---|---|
| 형식 (sample_id·label, 3,451행, 라벨 1~6) | ✓ |
| 같은 입력 다시 실행 | ✓ 0.00% 다름 |
| 최종 선택 파일 재현 | ✓ 0.00% 다름 |
| 행 순서 섞기 | ✓ 0.00% 다름 |
| 실험 나눠 넣기 | ✓ 0.00% 다름 |
| 미래 행 빼고 넣기 (실험마다 앞 60%) | ✓ 0.00% 다름 (2,074행) |
| 이미지 사용 (모든 이미지를 검은 이미지로) | ✓ 2.43% 예측 바뀜 |

## 사용한 사전학습 모델
- 없음

## 주의
- XGBoost를 GPU(`device='cuda'`)로 학습했습니다. `train.py --device cpu`로도 실행되지만 결과가 조금 달라질 수 있습니다. 저장된 가중치로 `test.py`만 실행하면 장치와 무관하게 같은 예측이 나옵니다.

## Experiment Branches

The `main` branch contains the final submission.
Individual experiments, model development, and
analysis results are maintained in separate branches.

| Member | Experiment Branch |
| --- | --- |
| Yeonwoo | [yeonwoo](../../tree/yeonwoo) |
| Hwanhee | [hwanhee](../../tree/hwanhee) |
| Minjun | [jangminjun](../../tree/jangminjun) |
