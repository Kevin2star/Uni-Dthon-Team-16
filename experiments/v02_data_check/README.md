# v02_data_check — 데이터 연결·분리 점검

## 목적
ID 중복, 실험 간 분리, 이미지 누락·중복(md5), 결측률, 시간 중복 점검.

## 실행
`python data_check.py` (3초) → `data_check.log`

## 결과
- ID·실험·이미지 중복 0, 누락 0, train/validation/test 실험 겹침 0
- 결측: `indu_t` 38~49%, `indu_t.1` 76~79% (**test에서는 100%** → 사용 금지)
- train·validation은 모두 5초 간격, **test는 5/10/15초 혼재** → 시계열은 행 수가 아니라 초 기준으로
- **라벨은 시간에 따라 절대 내려가지 않음** (모든 실험에서 1→6 방향)

## 이 폴더의 파일
- `data_check.py`
- `data_check.log`
