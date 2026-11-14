"""LightGBM 분류기 공통 설정. 시드·스레드 고정으로 같은 입력에 같은 결과."""
from lightgbm import LGBMClassifier

SEED = 42


def build_lgbm(class_weight=None):
    # 결측값은 LightGBM이 직접 처리하므로 imputer·scaler 불필요
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=50,          # 실험 78개뿐이라 잎을 너무 잘게 나누지 않게
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, class_weight=class_weight,
        random_state=SEED, n_jobs=8, deterministic=True, force_row_wise=True, verbose=-1,
    )
