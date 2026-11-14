# v08_ordinal_alarms — "k단계 이상인가?" 경보 5개 (순서형)

## 목적
6클래스 대신 경계마다 이진 판정기 5개 → 단계 = 1 + 예라고 답한 경보 수.

## 실행
`python ordinal_eval.py` (30초), `python viz_ordinal.py`

## 결과
| 방식 | CV | Val |
|---|---|---|
| 6클래스 LGBM | **0.876** ± 0.033 | 0.775 |
| 경보 5개 · 기준값 0.5 | 0.871 ± 0.032 | 0.773 |
| 경보 5개 · 기준값 조정 | 0.865 ± 0.028 | 0.771 |

## 결론
이득 없음. 기준값 조정은 fold마다 크게 흔들려 과적합. 경보 순서 어긋남 0.06% (데이터가 실제로 순서 구조). **≥3 경보가 가장 느리고 흔들림** = 2→3이 핵심 난제.

## 이 폴더의 파일
- `ordinal_eval.py`
- `viz_ordinal.py`
- `ordinal_probs_model_image_lgbm.csv`
- `ordinal_compare.png`
- `ordinal_alarms_validation.png`
