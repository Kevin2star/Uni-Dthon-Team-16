# v12_experiment_look_diagnosis — "실험 고유의 모습" 진단

## 목적
이미지 표현이 '단계'보다 '어떤 실험(배터리·구도)인가'에 반응하는지 분산 분해로 확인.

## 실행
`python diag_variance.py` (몇 초)

## 결과
| 표현 | 실험 차이 | 단계 차이 | 같은 실험 안 단계 |
|---|---|---|---|
| ResNet18 | **59%** | 12% | 8% |
| ConvNeXt-T | **61%** | 13% | 11% |
| 분포 64 | 39% | 23% | 14% |
| 퍼짐 13 | 46% | **36%** | **25%** |

## 결론
사전학습 CNN 임베딩은 배터리 생김새를 주로 담음. `what_is_experiment_look.png`: 행=실험, 열=단계 → 같은 단계라도 실험끼리 완전히 다르게 생김.

## 이 폴더의 파일
- `diag_variance.py`
- `diag_variance.log`
- `what_is_experiment_look.png`
