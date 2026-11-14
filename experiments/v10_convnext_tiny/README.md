# v10_convnext_tiny — ConvNeXt-Tiny 백본

## 목적
ResNet18 → ConvNeXt-Tiny (torchvision ImageNet-1K, 768차원, 가중치 고정).

## 실행
`python image_repr.py --arch resnet18 convnext_tiny` (약 10분). 임베딩 추출은 GPU 38초.
가중치: `~/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth` (BSD-3)

## 결과
| | CV | Val |
|---|---|---|
| 이미지만 ResNet18 | 0.722 | 0.690 |
| 이미지만 ConvNeXt-T | 0.718 | **0.719** |
| 센서 + ResNet18 PCA32 | 0.878 | 0.752 |
| 센서 + ConvNeXt PCA32 | 0.877 | 0.769 |

## 결론
백본 교체 효과 없음 (CV 동일).

## 이 폴더의 파일
- `image_repr.py`
