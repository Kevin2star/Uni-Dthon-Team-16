"""임베딩·피처의 분산이 '실험 차이'와 '단계 차이' 중 무엇으로 설명되는지 (Train)."""
from pathlib import Path
import numpy as np, pandas as pd
Y = Path('/home/ubuntu/yeonwoo')
f = pd.read_csv('/home/ubuntu/data/battery_data_light/train/train.csv')
exp, y = f.experiment_id.values, f.label.values
reps = {'ResNet18 (512)': np.load(Y / 'work_imgrep/cnn_train.npy'),
        'ConvNeXt-T (768)': np.load(Y / 'work_imgrep/cnn_convnext_tiny_train.npy'),
        'rich 64 (hand-made)': np.load(Y / 'work_imgrep/rich_train.npy'),
        'spread 13 (hand-made)': pd.read_csv(Y / 'work_model_image/img_train.csv').drop(columns='sample_id').values}
def ratio(X, g):
    X = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-8))
    return (pd.DataFrame(X).groupby(g).transform('mean').values ** 2).sum() / (X ** 2).sum()
print(f"{'표현':<24}{'실험 차이로 설명':>14}{'단계 차이로 설명':>14}{'실험 안에서 단계로 설명':>22}")
for n, X in reps.items():
    Xs = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-8))
    within = Xs - pd.DataFrame(Xs).groupby(exp).transform('mean').values
    gm = pd.DataFrame(within).groupby(y).transform('mean').values
    print(f"{n:<24}{ratio(X, exp) * 100:>13.0f}%{ratio(X, y) * 100:>13.0f}%{(gm ** 2).sum() / (within ** 2).sum() * 100:>21.0f}%")
