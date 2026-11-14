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
cd ~/<내이름>/Uni-Dthon-Team-16/preprocess
# GPU용 torch가 이미 있으면 아래 줄에서 torch torchvision은 빼십시오.
pip install lightgbm xgboost catboost
D=~/data/battery_data_light

# 표 모델 비교. 선택 기준은 Train GroupKFold Macro F1입니다.
python -u train.py --data_dir $D --compare tabular --model_out weights/model.joblib 2>&1 | tee train.log
```

시작한 뒤 `Ctrl+B`를 누르고 `D`를 누르면 세션에서 빠지면서 학습은 계속됩니다. 다시 들어갈 때는 `tmux attach -t train-<내이름>`.

딥러닝 후보는 한 번에 하나만 실행합니다. 이미지 피처와 torch 스레드는 기본 2개입니다. `gru_efficientnet`은 실행 목록에 없습니다. 이미 끝난 EfficientNet, ResNet18, MobileNet, smallcnn 다음에 아래 순서로 진행합니다. 각 명령이 끝난 뒤 로그의 교차검증 Macro F1, 4단계 F1, 5단계 F1을 LightGBM(0.8913, 0.8835, 0.8811)과 비교합니다.

```bash
python -u train.py --data_dir $D --model swin_t --head coral --model_out weights/swin_t_coral.joblib 2>&1 | tee train_swin_coral.log
python -u train.py --data_dir $D --model swin_t --model_out weights/swin_t.joblib 2>&1 | tee train_swin.log
python -u train.py --data_dir $D --model convnext_tiny --model_out weights/convnext_tiny.joblib 2>&1 | tee train_convnext.log
python -u train.py --data_dir $D --model resnet18_gapgmp --model_out weights/resnet18_gapgmp.joblib 2>&1 | tee train_resnet_gapgmp.log
python -u train.py --data_dir $D --model cnn1d --model_out weights/cnn1d.joblib 2>&1 | tee train_cnn1d.log
```

단일 프레임 Swin의 4단계와 5단계가 모두 LightGBM보다 낮으면 세 시점 모델과 나머지 이미지 모델을 이어서 하나씩 실행합니다.

```bash
python -u train.py --data_dir $D --model swin_t_time --model_out weights/swin_t_time.joblib 2>&1 | tee train_swin_time.log
python -u train.py --data_dir $D --model deit_small --model_out weights/deit_small.joblib 2>&1 | tee train_deit.log
python -u train.py --data_dir $D --model dinov2_vits14 --model_out weights/dinov2_vits14.joblib 2>&1 | tee train_dinov2.log
python -u train.py --data_dir $D --model maxvit_tiny --model_out weights/maxvit_tiny.joblib 2>&1 | tee train_maxvit.log
python -u train.py --data_dir $D --model mobilevit_s --model_out weights/mobilevit_s.joblib 2>&1 | tee train_mobilevit.log
```

한 모델의 교차검증이 LightGBM을 넘으면 시드 43, 44를 같은 방식으로 학습한 뒤 `blend.py`로 세 확률을 평균합니다. 조합은 저장된 가중치의 확률만 사용합니다.

```bash
python blend.py --models weights/efficientnet.joblib weights/resnet18.joblib --input $D/test/test.csv --image_root $D/test --output weights/submission_eff_resnet.csv --model_out weights/eff_resnet.joblib
python blend.py --models weights/lightgbm.joblib weights/xgboost.joblib weights/catboost.joblib weights/extratrees.joblib weights/histgb.joblib weights/cumulative.joblib --input $D/test/test.csv --image_root $D/test --output weights/submission_ensemble.csv --model_out weights/ensemble.joblib
python train.py --data_dir $D --model lightgbm --oof_only --model_out weights/lightgbm.joblib
python blend.py --fit-weights --models weights/swin_t.joblib weights/lightgbm.joblib --input $D/test/test.csv --image_root $D/test --output weights/submission_swin_lgb.csv --model_out weights/swin_lgb.joblib
python blend.py --models weights/swin_t.joblib weights/swin_t_coral.joblib --input $D/test/test.csv --image_root $D/test --output weights/submission_swin_coral.csv --model_out weights/swin_coral.joblib
python blend.py --stack --models weights/swin_t.joblib weights/lightgbm.joblib --input $D/test/test.csv --image_root $D/test --output weights/submission_stack.csv --model_out weights/stack.joblib
```

`--fit-weights`와 `--stack`은 `*_oof.npz`가 있을 때만 비중을 정합니다. 없는 파일은 Test 점수로 대체하지 않고 멈춥니다. `--oof_only`는 기존 `lightgbm.joblib`을 덮어쓰지 않습니다. `--stack`은 홀드아웃 실험 단위 교차검증이 확률 평균보다 높을 때만 제출 파일을 만듭니다.

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
- 윈도우는 30·60·120·300초입니다. 행 수가 아니라 초입니다. 창 안의 행 개수는 피처로 넣지 않습니다.
- 각 구간에서 평균, 최댓값, 구간 시작 대비 변화를 만듭니다. 가스·온도·이미지 상한 면적은 초당 기울기도 만듭니다. 기울기는 명목 창 길이가 아니라 두 관측의 실제 시간 차로 나눕니다.
- 창보다 과거 관측이 없으면 변화와 기울기는 0입니다. 비교할 관측이 아직 없다는 뜻이며, 학습 중앙값으로 채우지 않습니다.
- 센서는 기본 5개에 더해 가스(`meas_ch4`, `meas_co`, `meas_co2`, `meas_hcn`, `meas_hcl`, `meas_hf`, `meas_n2o`, `meas_no`, `meas_no2`, `meas_so2`)와 `thermal_t`, `atmp_t`, `indu_t`를 씁니다. 실험 조건 `bat_class`, `bat_cc`, `soc_status`, `tr_method`도 행에 들어 있으므로 씁니다.
- `meas_t1_plus_t2`는 Train에서 `meas_t1 + meas_t2`와 같아 빼었습니다. `indu_t.1`은 중복된 열 이름이고 Train 결측이 79%라 빼었습니다. `indu_t`는 Train 결측이 39.8%라 남겼습니다. 이 판단은 Train 결측만 봤습니다.
- 이미지는 회색 최댓값을 쓰지 않습니다. 대부분의 프레임에서 밝기 최댓값이 243/255에 고정되기 때문입니다. 대신 회색의 평균·표준편차·95·99백분위와, 밝기 0.9 이상 픽셀 비율을 씁니다. R·G·B 각각도 95·99백분위와 0.9 이상 비율을 씁니다.
- 색은 고정 색상표로 바꿉니다. 자홍(색상 300도)을 찬 쪽으로 두고, 색상이 내려가 빨강·흰색에 가까울수록 뜨거운 쪽으로 둡니다. 임계값과 색상표는 상수이며 데이터로 맞추지 않습니다. 인덱스의 95·99백분위와 0.9 이상 비율을 쓰고, 가장 큰 고온 덩어리와 중심 좌표는 픽셀 수나 변 길이로 나눈 비율입니다.
- 합성곱 입력은 변의 길이를 224로 맞춥니다.
- 실험에서 가장 이른 시각 대비 변화(`from_start`)도 현재와 그 이전만 사용합니다.
- 미래 시점, 실험 전체 평균·최댓값·길이·순위, 다른 실험의 관측은 사용하지 않습니다. Validation 라벨 비율은 맞추지 않습니다.
- 행 순서가 바뀌거나 실험의 앞부분만 남아도 남은 행의 피처는 같게 만듭니다.

## 모델
| 이름 | 내용 |
|---|---|
| lightgbm, xgboost, catboost, extratrees, histgb | 확장 피처를 쓰는 트리 모델 |
| cumulative | 1~6단계 순서에 대한 누적 이항 5개 |
| ensemble | 위 표 모델의 클래스 확률 평균 |
| efficientnet, resnet18, mobilenet, smallcnn | 현재 열화상과 센서 피처를 함께 분류 |
| resnet18_gapgmp | ResNet18의 평균 풀링과 최대 풀링을 이어 붙임 |
| convnext_tiny, swin_t, deit_small, dinov2_vits14, maxvit_tiny, mobilevit_s | 현재 열화상과 센서 피처 |
| swin_t_time | 현재, 30초 전, 120초 전 열화상을 각각 임베딩 |
| cnn1d | 과거 300초 센서만 사용. 이미지는 읽지 않음 |
| `--head coral` | 딥러닝 출력을 “k보다 큰가” 5개로 바꿈 |
| blend.py | 저장된 모델의 확률 평균, 홀드아웃 비중, 스택 |

최종 제출 모델은 열화상과 센서를 둘 다 사용합니다. 이미지 통계와 열화상 입력이 피처에 들어 있습니다.

## 사용한 사전학습 모델
딥러닝 후보를 선택한 경우에만 해당합니다. 대회 시작 전에 공개된 ImageNet-1K 가중치이며, 이 대회 데이터로 학습된 가중치는 쓰지 않습니다.

- efficientnet_b0: torchvision, ImageNet-1K V1, BSD-3-Clause
- resnet18: torchvision, ImageNet-1K V1, BSD-3-Clause
- mobilenet_v3_small: torchvision, ImageNet-1K V1, BSD-3-Clause
- convnext_tiny: torchvision, ImageNet-1K V1, BSD-3-Clause
- swin_t: torchvision, ImageNet-1K V1, BSD-3-Clause
- deit_small_patch16_224.fb_in1k: timm, ImageNet-1K, Apache-2.0
- vit_small_patch14_dinov2.lvd142m: timm, LVD-142M, Apache-2.0. 입력 크기는 224
- maxvit_tiny_tf_224.in1k: timm, ImageNet-1K, Apache-2.0
- mobilevit_s.cvnets_in1k: timm, ImageNet-1K, Apache-2.0
- smallcnn, cnn1d: 사전학습 가중치 없음

100MB를 넘는 가중치 링크: <드라이브 링크>
