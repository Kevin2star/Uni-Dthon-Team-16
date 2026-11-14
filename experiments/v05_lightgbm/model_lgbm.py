"""베이스라인 피처(model.py) + LightGBM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'uni_d_baseline_v7'))
from model import *          # noqa: F401,F403  피처 코드는 베이스라인 그대로
from lgbm_common import build_lgbm as build_model  # noqa: F401

FEATURE_CACHE = 'model'      # 이미지 피처 캐시를 베이스라인과 공유
