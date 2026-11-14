"""Uni-D 배터리 열폭주 6단계 분류 — 후보 모델 공통 모듈.

train.py, test.py, 실험 노트북이 같은 함수를 사용합니다.
피처는 입력 프레임에 있는 행만으로 만들고, 같은 실험에서 현재 시각 이전과
현재만 사용합니다. 실험 전체 평균·최댓값·길이·순위와 다른 실험의 관측은
쓰지 않습니다. 전처리 기준은 학습 호출 쪽에서 학습 데이터로만 맞춥니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from scipy import ndimage

SEED = 42
LABELS = list(range(1, 7))
SENSORS = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p']
WINDOWS = (30.0, 60.0, 120.0, 300.0)
HOT_THRESHOLD = 0.8
BLOB_SIZE = 48
IMAGE_SIZE = 224
SEQ_STEPS = 16
SEQ_WINDOW_S = 300.0

IMAGE_STATS = [
    'img_mean', 'img_std', 'img_max', 'img_p95', 'img_p99', 'img_hot_ratio',
    'img_hot60', 'img_hot90', 'img_blob_count', 'img_blob_max', 'img_cent_x', 'img_cent_y',
]
NAMED_EXTRA = ['thermal_t', 'atmp_t', 'indu_t', 'bat_class', 'bat_cc', 'soc_status', 'tr_method']
# Test에서 값이 비어 모델에 상수로만 들어가는 열. 입력으로 쓰지 않습니다.
EXCLUDED_COLUMNS = {'indu_t.1'}
ID_COLUMNS = {'sample_id', 'experiment_id', 'image_path', 'label', 'relative_time_s'}

TABULAR_MODELS = ['lightgbm', 'xgboost', 'catboost', 'extratrees', 'histgb', 'cumulative']
DEEP_MODELS = ['efficientnet', 'resnet18', 'mobilenet', 'smallcnn', 'gru_efficientnet']
COMBINATION_MODELS = ['ensemble']
ALL_MODELS = TABULAR_MODELS + DEEP_MODELS + COMBINATION_MODELS

TREE_ESTIMATORS = 300
TREE_LEARNING_RATE = 0.05


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        return


def extra_columns(frame: pd.DataFrame) -> list[str]:
    """제공 CSV에 실제로 있는 추가 센서·실험 조건만 고릅니다."""
    found = []
    for column in frame.columns:
        if column in ID_COLUMNS or column in SENSORS or column in EXCLUDED_COLUMNS:
            continue
        if column.startswith('meas_') or column in NAMED_EXTRA:
            found.append(column)
    return found


def _temperature_like(column: str) -> bool:
    return column.startswith('meas_t') or column in {'thermal_t', 'atmp_t'}


# ---------------------------------------------------------------------------
# 이미지. 샘플마다 독립이며 임계값은 고정값입니다.
# ---------------------------------------------------------------------------
def resolve_image_paths(frame: pd.DataFrame, roots: list[Path]) -> pd.Series:
    rel = frame['image_path'].astype(str)
    for root in roots:
        candidate = rel.map(lambda path: Path(root) / path)
        if candidate.map(Path.is_file).all():
            return candidate
    index = {}
    for root in roots:
        for path in Path(root).rglob('*'):
            if path.is_file():
                index.setdefault(path.name, path)
    candidate = rel.map(lambda path: index.get(Path(path).name))
    missing = int(candidate.isna().sum())
    assert missing == 0, f'이미지 {missing}개를 찾지 못했습니다. image_root를 확인하십시오.'
    return candidate


def _blob_stats(mask: np.ndarray) -> tuple[float, float]:
    count = int(ndimage.label(mask)[1])
    if count == 0:
        return 0.0, 0.0
    labeled, _ = ndimage.label(mask)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return float(count), float(sizes.max()) / float(mask.size)


def _image_stats(path) -> list[float]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    hot80 = arr >= HOT_THRESHOLD
    small = np.asarray(
        Image.fromarray((arr * 255).astype(np.uint8)).resize((BLOB_SIZE, BLOB_SIZE), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    count, max_area = _blob_stats(small >= HOT_THRESHOLD)
    ys, xs = np.where(hot80)
    if len(ys):
        cent_x = float(xs.mean() / max(arr.shape[1] - 1, 1))
        cent_y = float(ys.mean() / max(arr.shape[0] - 1, 1))
    else:
        cent_x = cent_y = float('nan')
    return [
        float(arr.mean()),
        float(arr.std()),
        float(arr.max()),
        float(np.percentile(arr, 95)),
        float(np.percentile(arr, 99)),
        float(hot80.mean()),
        float((arr >= 0.6).mean()),
        float((arr >= 0.9).mean()),
        count,
        max_area,
        cent_x,
        cent_y,
    ]


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_image_stats)(path) for path in paths)
    return pd.DataFrame(rows, columns=IMAGE_STATS, index=pd.Index(frame['sample_id'].to_numpy(), name='sample_id'))


def load_image_tensor(path, size: int = IMAGE_SIZE, normalize: str = 'imagenet', flip: bool = False):
    import torch
    with Image.open(path) as img:
        img = img.convert('RGB').resize((size, size), Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if flip:
        tensor = torch.flip(tensor, dims=[2])
    if normalize == 'imagenet':
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype)[:, None, None]
        tensor = (tensor - mean) / std
    return tensor


# ---------------------------------------------------------------------------
# 과거 방향 시계열. 시각을 정렬한 뒤 (t-window, t] 구간만 사용합니다.
# ---------------------------------------------------------------------------
def _sorted_positions(frame: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    times = frame['relative_time_s'].to_numpy(dtype=float)[positions]
    sample_ids = frame['sample_id'].astype(str).to_numpy()[positions]
    return positions[np.lexsort((sample_ids, times))]


def _lag_means(times: np.ndarray, values: np.ndarray, lag_pos: np.ndarray) -> np.ndarray:
    """같은 시각이 여러 행이면 그 시각의 평균을 씁니다. 입력 순서와 무관합니다."""
    out = np.full(len(times), np.nan)
    valid = lag_pos >= 0
    if not valid.any():
        return out
    means = pd.DataFrame({'t': times, 'x': values}).groupby('t', sort=False)['x'].mean()
    lag_time = times[lag_pos[valid]]
    out[valid] = pd.Series(lag_time).map(means).to_numpy(dtype=float)
    return out


def _causal_multi(times: np.ndarray, values: np.ndarray, windows: tuple[float, ...]):
    n = len(times)
    starts, lags = [], []
    ends = np.searchsorted(times, times, side='right')
    for window in windows:
        starts.append(np.searchsorted(times, times - window, side='right'))
        lags.append(np.searchsorted(times, times - window, side='right') - 1)
    means = [np.full(n, np.nan) for _ in windows]
    maxima = [np.full(n, np.nan) for _ in windows]
    deltas = [np.full(n, np.nan) for _ in windows]
    lag_values = [_lag_means(times, values, lag) for lag in lags]
    for i in range(n):
        for k in range(len(windows)):
            window_values = values[starts[k][i]:ends[i]]
            window_values = window_values[~np.isnan(window_values)]
            if len(window_values):
                means[k][i] = window_values.mean()
                maxima[k][i] = window_values.max()
            if lags[k][i] >= 0:
                deltas[k][i] = values[i] - lag_values[k][i]
    return means, maxima, deltas


def _from_start(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    earliest = times == np.nanmin(times)
    anchor_rows = values[earliest]
    finite = anchor_rows[np.isfinite(anchor_rows)]
    anchor = float(finite.mean()) if len(finite) else np.nan
    return values - anchor


def add_causal_features(base: pd.DataFrame, signals: list[str], slope_signals: list[str]) -> pd.DataFrame:
    columns = []
    for signal in signals:
        for window in WINDOWS:
            width = int(window)
            columns.extend([f'{signal}_mean{width}s', f'{signal}_max{width}s', f'{signal}_delta{width}s'])
            if signal in slope_signals:
                columns.append(f'{signal}_slope{width}s')
        columns.append(f'{signal}_from_start')
    out = pd.DataFrame(np.nan, index=base.index, columns=columns)
    times_all = base['relative_time_s'].to_numpy(dtype=float)
    for _, positions in base.groupby('experiment_id', sort=False).indices.items():
        positions = _sorted_positions(base, np.asarray(positions))
        times = times_all[positions]
        for signal in signals:
            values = base[signal].to_numpy(dtype=float)[positions]
            means, maxima, deltas = _causal_multi(times, values, WINDOWS)
            start = _from_start(times, values)
            out.iloc[positions, out.columns.get_loc(f'{signal}_from_start')] = start
            for k, window in enumerate(WINDOWS):
                width = int(window)
                out.iloc[positions, out.columns.get_loc(f'{signal}_mean{width}s')] = means[k]
                out.iloc[positions, out.columns.get_loc(f'{signal}_max{width}s')] = maxima[k]
                out.iloc[positions, out.columns.get_loc(f'{signal}_delta{width}s')] = deltas[k]
                if signal in slope_signals:
                    out.iloc[positions, out.columns.get_loc(f'{signal}_slope{width}s')] = deltas[k] / window
    return out


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)


def build_features(frame: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    """frame의 행 순서대로 피처를 만듭니다. 다른 실험과 미래 관측은 쓰지 않습니다."""
    frame = frame.reset_index(drop=True)
    extra = extra_columns(frame)
    base = pd.DataFrame({
        'sample_id': frame['sample_id'].astype(str).to_numpy(),
        'experiment_id': frame['experiment_id'].astype(str).to_numpy(),
        'relative_time_s': pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(),
    })
    sensor_columns = SENSORS + extra
    base[sensor_columns] = numeric(frame, sensor_columns).to_numpy()
    image = image_features.reindex(frame['sample_id'].astype(str).to_numpy())
    assert image['img_mean'].notna().all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[IMAGE_STATS] = image.to_numpy()
    signals = sensor_columns + ['img_max', 'img_p99', 'img_hot_ratio', 'img_blob_max']
    slopes = [column for column in sensor_columns if _temperature_like(column)]
    causal = add_causal_features(base, signals, slopes)
    current = ['relative_time_s'] + sensor_columns + IMAGE_STATS
    return pd.concat([base[current], causal], axis=1)


def sequence_columns(frame: pd.DataFrame) -> list[str]:
    return SENSORS + extra_columns(frame)


def build_sequences(frame: pd.DataFrame, sensor_matrix: np.ndarray, max_steps: int = SEQ_STEPS, window_s: float = SEQ_WINDOW_S):
    """각 행의 과거 max_steps개. 현재 시각보다 이후 행은 들어가지 않습니다."""
    frame = frame.reset_index(drop=True)
    n, width = sensor_matrix.shape
    sequences = np.zeros((n, max_steps, width), dtype=np.float32)
    mask = np.zeros((n, max_steps), dtype=np.float32)
    times_all = frame['relative_time_s'].to_numpy(dtype=float)
    for _, positions in frame.groupby('experiment_id', sort=False).indices.items():
        positions = _sorted_positions(frame, np.asarray(positions))
        times = times_all[positions]
        start = 0
        for j, row in enumerate(positions):
            while start < j and times[start] < times[j] - window_s:
                start += 1
            chosen = positions[start:j + 1][-max_steps:]
            length = len(chosen)
            sequences[row, :length] = sensor_matrix[chosen]
            mask[row, :length] = 1.0
    return sequences, mask


def future_cut_matches(frame: pd.DataFrame, image_features: pd.DataFrame) -> bool:
    """실험마다 앞쪽 절반만 남겨도 남은 행의 피처가 같은지 확인합니다."""
    full = build_features(frame, image_features)
    full.index = frame.reset_index(drop=True)['sample_id'].astype(str).to_numpy()
    parts = []
    for _, sub in frame.groupby('experiment_id', sort=False):
        median = sub['relative_time_s'].median()
        parts.append(sub[sub['relative_time_s'] <= median])
    cut = pd.concat(parts, ignore_index=True)
    cut_x = build_features(cut, image_features)
    cut_x.index = cut['sample_id'].astype(str).to_numpy()
    both = full.loc[cut_x.index]
    return bool(np.allclose(both.to_numpy(dtype=float), cut_x.to_numpy(dtype=float), equal_nan=True))


# ---------------------------------------------------------------------------
# 표 모델. 결측은 트리 계열의 자체 처리를 쓰고, Extra Trees만 중앙값으로 채웁니다.
# ---------------------------------------------------------------------------
def _as_float(features) -> np.ndarray:
    if isinstance(features, pd.DataFrame):
        return features.to_numpy(dtype=np.float64, copy=True)
    return np.array(features, dtype=np.float64, copy=True)


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight = {label: len(y) / (len(classes) * count) for label, count in zip(classes, counts)}
    return np.array([weight[label] for label in y], dtype=np.float64)


def _align_proba(proba: np.ndarray, classes) -> np.ndarray:
    out = np.zeros((len(proba), len(LABELS)), dtype=np.float64)
    for column, label in enumerate(classes):
        out[:, int(label) - 1] = proba[:, column]
    return out


class TabularPrep:
    """학습 데이터에서만 중앙값과 표준화 기준을 구합니다."""

    def fit(self, features):
        values = _as_float(features)
        median = np.nanmedian(values, axis=0)
        self.median_ = np.where(np.isnan(median), 0.0, median)
        filled = self._fill(values)
        self.mean_ = filled.mean(axis=0)
        scale = filled.std(axis=0)
        self.scale_ = np.where(scale < 1e-6, 1.0, scale)
        return self

    def _fill(self, values: np.ndarray) -> np.ndarray:
        values = values.copy()
        missing = np.isnan(values)
        if missing.any():
            values[missing] = np.take(self.median_, np.where(missing)[1])
        return values

    def transform(self, features) -> np.ndarray:
        return (self._fill(_as_float(features)) - self.mean_) / self.scale_


class TreeModel:
    def __init__(self, name: str, class_weight=None, seed: int = SEED):
        self.name = name
        self.class_weight = class_weight
        self.seed = seed
        self.model = None
        self.median_ = None
        self._y_offset = 0

    def fit(self, features, y):
        y = np.asarray(y).astype(int)
        # XGBoost는 클래스 번호가 0부터여야 합니다. 저장·제출 라벨은 1~6입니다.
        self._y_offset = 1 if self.name == 'xgboost' else 0
        fit_y = y - self._y_offset
        values = _as_float(features)
        if self.name == 'extratrees':
            median = np.nanmedian(values, axis=0)
            self.median_ = np.where(np.isnan(median), 0.0, median)
            values = self._fill(values)
        self.model = self._build()
        uses_constructor_weight = self.name in {'lightgbm', 'extratrees', 'histgb'}
        if self.class_weight == 'balanced' and not uses_constructor_weight:
            self.model.fit(values, fit_y, sample_weight=_balanced_weights(y))
        else:
            self.model.fit(values, fit_y)
        return self

    def _fill(self, values: np.ndarray) -> np.ndarray:
        values = values.copy()
        missing = np.isnan(values)
        if missing.any():
            values[missing] = np.take(self.median_, np.where(missing)[1])
        return values

    def _build(self):
        weight = self.class_weight if self.class_weight == 'balanced' else None
        if self.name == 'lightgbm':
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=TREE_ESTIMATORS, learning_rate=TREE_LEARNING_RATE, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, random_state=self.seed, n_jobs=-1,
                class_weight=weight, deterministic=True, force_row_wise=True, verbosity=-1,
            )
        if self.name == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=TREE_ESTIMATORS, learning_rate=TREE_LEARNING_RATE, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
                random_state=self.seed, n_jobs=-1, tree_method='hist', verbosity=0,
            )
        if self.name == 'catboost':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=TREE_ESTIMATORS, learning_rate=TREE_LEARNING_RATE, depth=6,
                loss_function='MultiClass', random_seed=self.seed, verbose=False,
                allow_writing_files=False,
            )
        if self.name == 'extratrees':
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=TREE_ESTIMATORS, random_state=self.seed, n_jobs=-1, class_weight=weight,
            )
        if self.name == 'histgb':
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=TREE_ESTIMATORS, learning_rate=TREE_LEARNING_RATE, random_state=self.seed,
                class_weight=weight,
            )
        raise ValueError(f'알 수 없는 표 모델입니다: {self.name}')

    def _values(self, features) -> np.ndarray:
        values = _as_float(features)
        if self.median_ is not None:
            values = self._fill(values)
        return values

    def predict_proba(self, features) -> np.ndarray:
        proba = self.model.predict_proba(self._values(features))
        classes = [int(label) + getattr(self, '_y_offset', 0) for label in self.model.classes_]
        return _align_proba(proba, classes)

    def predict(self, features) -> np.ndarray:
        return (np.argmax(self.predict_proba(features), axis=1) + 1).astype(int)


class CumulativeOrdinal:
    """단계 k보다 큰지에 대한 이항 모델 5개. 확률은 단계가 커질수록 내려가게 맞춥니다."""

    def __init__(self, class_weight=None, seed: int = SEED):
        self.class_weight = class_weight
        self.seed = seed
        self.models = []
        self.constants_ = []

    def fit(self, features, y):
        y = np.asarray(y).astype(int)
        self.models = []
        self.constants_ = []
        for level in range(1, 6):
            target = (y > level).astype(int)
            if len(np.unique(target)) < 2:
                self.models.append(None)
                self.constants_.append(float(target[0]) if len(target) else 0.0)
                continue
            model = TreeModel('lightgbm', self.class_weight, self.seed)
            model.fit(features, target)
            self.models.append(model)
            self.constants_.append(None)
        return self

    def _positive_probability(self, model, features) -> np.ndarray:
        values = model._values(features)
        proba = model.model.predict_proba(values)
        classes = list(model.model.classes_)
        if 1 not in classes:
            return np.zeros(values.shape[0], dtype=np.float64)
        return proba[:, classes.index(1)]

    def predict_proba(self, features) -> np.ndarray:
        n = features.shape[0]
        greater = []
        for model, constant in zip(self.models, self.constants_):
            if model is None:
                greater.append(np.full(n, constant, dtype=np.float64))
            else:
                greater.append(self._positive_probability(model, features))
        greater = np.minimum.accumulate(np.column_stack(greater), axis=1)
        extended = np.column_stack([
            np.ones(n), greater, np.zeros(n),
        ])
        return np.clip(extended[:, :-1] - extended[:, 1:], 0, None)

    def predict(self, features) -> np.ndarray:
        return (np.argmax(self.predict_proba(features), axis=1) + 1).astype(int)


class EnsembleModel:
    """표 모델들의 클래스 확률을 같은 비중으로 평균합니다."""

    def __init__(self, names: list[str] | None = None, class_weight=None, seed: int = SEED):
        self.names = list(names or TABULAR_MODELS)
        self.class_weight = class_weight
        self.seed = seed
        self.models = []

    def fit(self, features, y):
        self.models = []
        for name in self.names:
            if name == 'cumulative':
                model = CumulativeOrdinal(self.class_weight, self.seed)
            else:
                model = TreeModel(name, self.class_weight, self.seed)
            model.fit(features, y)
            self.models.append(model)
        return self

    def predict_proba(self, features) -> np.ndarray:
        return np.mean([model.predict_proba(features) for model in self.models], axis=0)

    def predict(self, features) -> np.ndarray:
        return (np.argmax(self.predict_proba(features), axis=1) + 1).astype(int)


def make_tabular(name: str, class_weight=None, seed: int = SEED):
    if name == 'cumulative':
        return CumulativeOrdinal(class_weight, seed)
    if name == 'ensemble':
        return EnsembleModel(TABULAR_MODELS, class_weight, seed)
    if name not in TABULAR_MODELS:
        raise ValueError(f'표 모델이 아닙니다: {name}')
    return TreeModel(name, class_weight, seed)


# ---------------------------------------------------------------------------
# 딥러닝. 현재 열화상과 센서 피처를 함께 쓰고, GRU는 과거 센서만 봅니다.
# ---------------------------------------------------------------------------
def deep_spec(name: str) -> tuple[str, bool]:
    if name == 'gru_efficientnet':
        return 'efficientnet', True
    if name in {'efficientnet', 'resnet18', 'mobilenet', 'smallcnn'}:
        return name, False
    raise ValueError(f'딥러닝 모델이 아닙니다: {name}')


def _import_torch():
    import torch
    from torch import nn
    return torch, nn


class SmallThermalCNN(object):
    """흑백에 가까운 열화상을 위한 작은 합성곱망. 사전학습 가중치는 없습니다."""

    def __new__(cls):
        torch, nn = _import_torch()

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )

            def forward(self, x):
                return self.features(x).flatten(1)

        return _Net()


def build_backbone(name: str, pretrained: bool):
    torch, nn = _import_torch()
    if name == 'smallcnn':
        return SmallThermalCNN(), 128, 'none'
    if name == 'efficientnet':
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)
        dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, dim, 'imagenet'
    if name == 'resnet18':
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim, 'imagenet'
    if name == 'mobilenet':
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, dim, 'imagenet'
    raise ValueError(name)


class FusionNet(object):
    def __new__(cls, backbone_name: str, image_dim: int, tabular_dim: int, sensor_dim: int, use_gru: bool, head: str, backbone):
        torch, nn = _import_torch()

        class _Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.use_gru = use_gru
                self.head_kind = head
                self.freeze_backbone = False
                self.img_proj = nn.Linear(image_dim, 128)
                self.tabular = nn.Sequential(nn.Linear(tabular_dim, 128), nn.ReLU(), nn.Dropout(0.2))
                fused = 256
                if use_gru:
                    self.gru = nn.GRU(sensor_dim, 128, batch_first=True)
                    fused += 128
                self.classifier = nn.Linear(fused, 5 if head == 'coral' else 6)

            def train(self, mode: bool = True):
                super().train(mode)
                if self.freeze_backbone:
                    self.backbone.eval()
                return self

            def forward(self, image, tabular, seq=None, mask=None):
                image_vector = self.img_proj(self.backbone(image))
                parts = [image_vector, self.tabular(tabular)]
                if self.use_gru:
                    lengths = mask.sum(dim=1).clamp(min=1).cpu()
                    packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
                    _, hidden = self.gru(packed)
                    parts.append(hidden[-1])
                return self.classifier(torch.cat(parts, dim=1))

        return _Fusion()


def logits_to_proba(logits, head: str):
    torch, _ = _import_torch()
    if head == 'ce':
        return torch.softmax(logits, dim=1)
    greater = torch.cummin(torch.sigmoid(logits), dim=1).values
    ones = torch.ones(logits.size(0), 1, device=logits.device, dtype=logits.dtype)
    zeros = torch.zeros(logits.size(0), 1, device=logits.device, dtype=logits.dtype)
    extended = torch.cat([ones, greater, zeros], dim=1)
    return (extended[:, :-1] - extended[:, 1:]).clamp_min(0)


def build_fusion(config: dict, pretrained: bool):
    backbone_name, _ = deep_spec(config['model_name'])
    backbone, image_dim, normalize = build_backbone(backbone_name, pretrained=pretrained)
    network = FusionNet(
        backbone_name, image_dim, config['tabular_dim'], config['sensor_dim'],
        config['use_gru'], config['head'], backbone,
    )
    network.freeze_backbone = config.get('freeze_backbone', True) and backbone_name != 'smallcnn'
    if network.freeze_backbone:
        for parameter in network.backbone.parameters():
            parameter.requires_grad = False
    return network, normalize


PRETRAINED_MODELS = {
    'efficientnet': 'torchvision efficientnet_b0, ImageNet-1K V1, BSD-3-Clause',
    'resnet18': 'torchvision resnet18, ImageNet-1K V1, BSD-3-Clause',
    'mobilenet': 'torchvision mobilenet_v3_small, ImageNet-1K V1, BSD-3-Clause',
    'gru_efficientnet': 'torchvision efficientnet_b0, ImageNet-1K V1, BSD-3-Clause + 과거 센서 GRU',
    'smallcnn': '사전학습 가중치 없음',
}


def predict_labels(bundle: dict, frame: pd.DataFrame, roots: list[Path], batch_size: int = 32) -> pd.Series:
    """입력 프레임의 행만으로 1~6 라벨을 예측합니다. 학습은 하지 않습니다."""
    frame = frame.reset_index(drop=True)
    paths = resolve_image_paths(frame, roots)
    image_features = extract_image_features(frame, paths)
    features = build_features(frame, image_features).reindex(columns=bundle['features'])
    if bundle['kind'] == 'tabular':
        prediction = bundle['estimator'].predict(features)
    else:
        prediction = _predict_torch(bundle, frame, paths, features, batch_size)
    return pd.Series(np.asarray(prediction).astype(int), index=frame['sample_id'].astype(str).to_numpy())


def _sensor_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    present = frame.reindex(columns=columns)
    return numeric(present, columns).to_numpy(dtype=np.float64)


def _predict_torch(bundle: dict, frame: pd.DataFrame, paths: pd.Series, features: pd.DataFrame, batch_size: int) -> np.ndarray:
    torch, _ = _import_torch()
    network, normalize = build_fusion(bundle['config'], pretrained=False)
    network.load_state_dict(bundle['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()
    tabular = bundle['tabular_prep'].transform(features).astype(np.float32)
    sequences = mask = None
    if bundle['config']['use_gru']:
        sensor = bundle['sequence_prep'].transform(_sensor_matrix(frame, bundle['sequence_cols'])).astype(np.float32)
        sequences, mask = build_sequences(frame, sensor)
    prediction = np.zeros(len(frame), dtype=int)
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            images = torch.stack([
                load_image_tensor(paths.iloc[i], normalize=normalize, flip=False) for i in range(start, stop)
            ]).to(device)
            table = torch.from_numpy(tabular[start:stop]).to(device)
            seq = mask_batch = None
            if sequences is not None:
                seq = torch.from_numpy(sequences[start:stop]).to(device)
                mask_batch = torch.from_numpy(mask[start:stop]).to(device)
            logits = network(images, table, seq, mask_batch)
            proba = logits_to_proba(logits, bundle['config']['head'])
            prediction[start:stop] = proba.argmax(dim=1).cpu().numpy() + 1
    return prediction
