# Uni-D 배터리 열폭주 6단계 분류

> 참가팀은 `<>` 부분을 자기 팀 내용으로 바꿔서 제출하십시오.

## 팀 정보
- 팀명: <팀명>
- 최종 선택 제출 파일: <리더보드에서 선택한 파일명>, Private 복원 대상

## 개발 환경
- OS: <예: Ubuntu 22.04>
- Python: <예: 3.10>
- GPU: 대회 제공 GPU 서버. 딥러닝 후보를 학습할 때만 사용합니다.
- 라이브러리: `requirements.txt` 참고. `torch`는 `torch>=2.7`처럼 하한만 적습니다. 서버에 GPU용 torch가 이미 있으면 torch와 torchvision은 다시 설치하지 않습니다.

## 파일 구성
| 파일 | 역할 |
|---|---|
| `uni_d_model_experiments.ipynb` | 실험 설계와 확인. 학습은 실행하지 않습니다. |
| `model.py` | 인과 피처, 표 모델, 딥러닝 결합 모델 |
| `train.py` | Train 실험 단위 교차검증으로 모델을 고르고 가중치를 저장 |
| `test.py` | 입력 CSV의 행만으로 추론하고 제출 파일 생성 |
| `weights/` | 학습된 가중치. 100MB를 넘으면 드라이브에 올리고 아래에 링크를 적습니다. |
| `requirements.txt` | 사용 패키지 |

## 실행 방법
학습과 추론은 노트북 셀이 아니라 tmux에서 실행합니다. 연결이 끊겨도 작업이 유지됩니다.

```bash
tmux new -s train-<내이름>
source ~/venv/bin/activate
cd ~/<내이름>/Uni-Dthon-Team-16
# GPU용 torch가 이미 있으면 아래 줄에서 torch torchvision은 빼십시오.
pip install lightgbm xgboost catboost
D=~/data/battery_data_light

# 표 모델 비교. 선택 기준은 Train GroupKFold Macro F1입니다.
python -u train.py --data_dir $D --compare tabular --model_out weights/model.joblib 2>&1 | tee train.log
```

시작한 뒤 `Ctrl+B`를 누르고 `D`를 누르면 세션에서 빠지면서 학습은 계속됩니다. 다시 들어갈 때는 `tmux attach -t train-<내이름>`.

딥러닝 후보와 조합은 표 모델 비교 뒤에, 각각 다른 파일로 저장합니다.

```bash
python -u train.py --data_dir $D --model efficientnet --model_out weights/efficientnet.joblib 2>&1 | tee train_efficientnet.log
python -u train.py --data_dir $D --model gru_efficientnet --head coral --model_out weights/gru_coral.joblib 2>&1 | tee train_gru.log
python -u train.py --data_dir $D --model ensemble --model_out weights/ensemble.joblib 2>&1 | tee train_ensemble.log
```

GPU를 쓰는 동안 다른 터미널에서 `watch -n 2 nvidia-smi`로 확인합니다. 끝낼 때는 `Ctrl+C`입니다.

추론도 tmux에서, 저장된 가중치만으로 실행합니다.

```bash
python test.py --input $D/test/test.csv --image_root $D/test --output submission.csv
```

`--model`의 기본값은 `weights/model.joblib`입니다. 다른 후보를 쓸 때는 `--model weights/<후보>.joblib`를 줍니다.

## 학습·검증 구성
- 학습 데이터: Train. `--refit_train_val`을 준 표 모델만 선택 후 Train+Validation으로 다시 학습합니다. 그 재학습 모델의 Validation 점수는 성능 숫자로 쓰지 않습니다.
- 검증 방법: Train 실험을 `GroupKFold`로 나눕니다. 표 모델은 5분할, 딥러닝은 기본 1개 홀드아웃으로 epoch을 고른 뒤 Train 전체로 다시 학습합니다.
- 선택 기준: 교차검증 Macro F1. 4단계 F1과 5단계 F1을 함께 기록합니다. Validation 19개 실험은 후보를 고른 뒤 한 번만 확인합니다.
- Test는 추론에만 사용합니다.

## 시계열 피처 구성
- 같은 `experiment_id` 안에서 `relative_time_s`가 현재 이하인 관측만 사용합니다.
- 윈도우는 30·60·120·300초입니다. 각 구간에서 평균, 최댓값, 구간 시작 대비 변화, 온도 계열의 초당 기울기를 만듭니다.
- 실험에서 가장 이른 시각 대비 변화(`from_start`)도 현재와 그 이전만 사용합니다.
- 미래 시점, 실험 전체 평균·최댓값·길이·순위, 다른 실험의 관측은 사용하지 않습니다.
- 행 순서가 바뀌거나 실험의 앞부분만 남아도 남은 행의 피처는 같게 만듭니다.
- `indu_t.1`은 Test 결측이 많아 입력에서 빼 두었습니다.

## 모델
| 이름 | 내용 |
|---|---|
| lightgbm, xgboost, catboost, extratrees, histgb | 확장 피처를 쓰는 트리 모델 |
| cumulative | 1~6단계 순서에 대한 누적 이항 5개 |
| ensemble | 위 표 모델의 클래스 확률 평균 |
| efficientnet, resnet18, mobilenet, smallcnn | 현재 열화상과 센서 피처를 함께 분류 |
| gru_efficientnet | 위 결합에 과거 300초 센서 GRU를 추가 |
| `--head coral` | 딥러닝 출력을 “k보다 큰가” 5개로 바꿈 |

최종 제출 모델은 열화상과 센서를 둘 다 사용합니다. 이미지 통계와 열화상 입력이 피처에 들어 있습니다.

## 사용한 사전학습 모델
딥러닝 후보를 선택한 경우에만 해당합니다. 대회 시작 전에 공개된 ImageNet-1K 가중치이며, 이 대회 데이터로 학습된 가중치는 쓰지 않습니다.

- efficientnet_b0: torchvision, ImageNet-1K V1, BSD-3-Clause
- resnet18: torchvision, ImageNet-1K V1, BSD-3-Clause
- mobilenet_v3_small: torchvision, ImageNet-1K V1, BSD-3-Clause
- smallcnn: 사전학습 가중치 없음

100MB를 넘는 가중치 링크: <드라이브 링크>
