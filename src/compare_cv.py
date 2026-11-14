"""베이스라인(model.py) vs 새 모듈의 Train GroupKFold CV / Validation Macro F1 비교.

사용법: python compare_cv.py model_image
이미지 피처는 work_<모듈명>/ 에 캐시합니다.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from paths import DATA_ROOT  # 데이터 위치: 환경변수 DATA_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                 # ~/yeonwoo
P_CACHE = ROOT / 'cache'           # 이미지 피처·임베딩 캐시
P_OUT = ROOT / 'outputs'           # 그림·로그·예측 결과
sys.path[:0] = [str(HERE), str(ROOT / 'uni_d_baseline_v7')]
DATA = DATA_ROOT


def run(mod_name):
    M = importlib.import_module(mod_name)
    cache_dir = P_CACHE / f"work_{getattr(M, 'FEATURE_CACHE', mod_name)}"
    cache_dir.mkdir(exist_ok=True)

    def load(split):
        f = pd.read_csv(DATA / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        cache = cache_dir / f'img_{split}.csv'
        if cache.is_file():
            img = pd.read_csv(cache, dtype={'sample_id': 'string'}, float_precision='round_trip').set_index('sample_id')
        else:
            img = M.extract_image_features(f, M.resolve_image_paths(f, [DATA / split, DATA]))
            img.to_csv(cache)
        return f, M.build_features(f, img), f.label.astype(int)

    tr, Xtr, ytr = load('train')
    va, Xva, yva = load('validation')
    f1 = lambda a, b: f1_score(a, b, labels=M.LABELS, average='macro', zero_division=0)
    folds = []
    for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups=tr.experiment_id):
        m = M.build_model(None).fit(Xtr.iloc[a], ytr.iloc[a])
        folds.append(f1(ytr.iloc[b], m.predict(Xtr.iloc[b])))
    m = M.build_model(None).fit(Xtr, ytr)
    pv = m.predict(Xva)
    per = f1_score(yva, pv, labels=M.LABELS, average=None, zero_division=0)
    print(f'{mod_name:>12} | 피처 {Xtr.shape[1]:>3} | CV {np.mean(folds):.4f} ± {np.std(folds):.4f} '
          f'| Val {f1(yva, pv):.4f} | Val 단계별 F1 {np.round(per, 3).tolist()}')


if __name__ == '__main__':
    for name in ['model'] + sys.argv[1:]:
        run(name)
