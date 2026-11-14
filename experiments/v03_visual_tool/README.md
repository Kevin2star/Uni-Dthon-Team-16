# v03_visual_tool — 데이터 눈으로 보기 도구

## 목적
단계별 열화상 샘플, 실험 하나의 센서 시계열 + 단계 + 이미지, 단계별 센서 분포를 그림으로 확인.

## 실행
```
python viz.py grid                    # 단계별 이미지 격자
python viz.py exps                    # 실험 목록
python viz.py exp 0 [--split validation]
python viz.py sensors
```

## 관찰
- 열화상 색은 **이미지마다 따로 정규화된 상대 온도** (1단계에도 빨강이 나옴)
- 실험마다 이미지 크기·구도가 다름 (작은 셀 vs 큰 모듈)

## 이 폴더의 파일
- `viz.py`
- `grid_train_seed0.png`
- `grid_train_seed3.png`
- `exp_train_E_019ea6ba.png`
- `exp_test_E_09d5d72c.png`
- `sensors_train.png`
