"""Uni-D 배터리 열폭주 6단계 분류 — 후보 모델 공통 모듈.

train.py, test.py, 실험 노트북이 같은 함수를 사용합니다.
피처는 입력 프레임에 있는 행만으로 만들고, 같은 실험에서 현재 시각 이전과
현재만 사용합니다. 실험 전체 평균·최댓값·길이·순위와 다른 실험의 관측은
쓰지 않습니다. 전처리 기준은 학습 호출 쪽에서 학습 데이터로만 맞춥니다.
"""
from __future__ import annotations

import os
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
BLOB_SIZE = 48
IMAGE_SIZE = 224
SEQ_STEPS = 16
SEQ_WINDOW_S = 300.0

# 회색 최댓값은 대부분의 프레임이 243에 고정되므로 넣지 않습니다.
# 상한에 닿은 면적은 밝기 0.9 이상 비율로 둡니다.
IMAGE_STATS = [
    'img_mean', 'img_std', 'img_p95', 'img_p99', 'img_hot90',
    'r_p95', 'r_p99', 'r_hot90', 'g_p95', 'g_p99', 'g_hot90', 'b_p95', 'b_p99', 'b_hot90',
    'pal_p95', 'pal_p99', 'pal_hot90', 'img_blob_max', 'img_cent_x', 'img_cent_y',
]
# 창 피처에 쓰는 이미지 신호. img_max는 넣지 않습니다.
IMAGE_WINDOW_SIGNALS = ['img_p99', 'img_hot90', 'pal_p99', 'pal_hot90']
NAMED_EXTRA = ['thermal_t', 'atmp_t', 'indu_t', 'bat_class', 'bat_cc', 'soc_status', 'tr_method']
# indu_t.1은 같은 이름이 두 번인 열입니다. Train 결측이 79%입니다.
# meas_t1_plus_t2는 Train 전 행에서 meas_t1+meas_t2와 같습니다.
# indu_t는 Train 결측 39.8%라 대부분 비어 있지 않아 남깁니다. 판단은 Train만 봤습니다.
EXCLUDED_COLUMNS = {'indu_t.1', 'meas_t1_plus_t2'}
GAS_COLUMNS = {
    'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
    'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2',
}
ID_COLUMNS = {'sample_id', 'experiment_id', 'image_path', 'label', 'relative_time_s'}
# 고정 색상표. 데이터로 맞추지 않습니다.
# 찬 쪽은 자홍(색상 300도)이고, 색상이 내려가며 파랑·청록·초록·노랑·빨강으로 뜨거워집니다.
# 채도가 낮은 밝은 픽셀은 그 색상표의 흰색 끝입니다.
PALETTE_COLD_HUE = 300.0 / 360.0
PALETTE_WHITE_SAT = 0.15
PALETTE_WHITE_VALUE = 0.9
PALETTE_BLACK_VALUE = 0.05
CEILING = 0.9

TABULAR_MODELS = ['lightgbm', 'xgboost', 'catboost', 'extratrees', 'histgb', 'cumulative']
# gru_efficientnet은 저장된 가중치를 읽을 수 있게만 남겨 둡니다. 새 실험 목록에는 없습니다.
DEEP_MODELS = [
    'efficientnet', 'resnet18', 'resnet18_gapgmp', 'mobilenet', 'smallcnn',
    'cnn1d', 'convnext_tiny', 'swin_t', 'swin_t_time',
    'deit_small', 'dinov2_vits14', 'maxvit_tiny', 'mobilevit_s', 'gru_efficientnet',
]
COMPARE_DEEP_MODELS = ['efficientnet', 'resnet18', 'mobilenet', 'smallcnn']
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
    return column.startswith('meas_t') or column in {'thermal_t', 'atmp_t', 'indu_t'}


def _wants_slope(signal: str) -> bool:
    """초당 기울기. 가스·온도와 이미지 상한 면적에만 둡니다. 창 안 행 개수는 쓰지 않습니다."""
    return _temperature_like(signal) or signal in GAS_COLUMNS or signal in IMAGE_WINDOW_SIGNALS


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


def _blob_max_ratio(mask: np.ndarray) -> float:
    """가장 큰 덩어리가 차지하는 픽셀 비율. 덩어리 개수는 해상도에 따라 달라지므로 쓰지 않습니다."""
    labeled, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return float(sizes.max()) / float(mask.size)


def _channel_stats(channel: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.percentile(channel, 95)),
        float(np.percentile(channel, 99)),
        float((channel >= CEILING).mean()),
    )


def palette_index(rgb: np.ndarray) -> np.ndarray:
    """RGB를 고정 색상표 인덱스 0(찬 쪽)~1(뜨거운 쪽)로 바꿉니다.

    기준은 색상 각도입니다. 자홍에서 시작해 각도가 줄어드는 방향이
    파랑, 청록, 초록, 노랑, 빨강입니다. 채도가 낮은 밝은 색은 흰색 끝으로 둡니다.
    """
    x = rgb.astype(np.float32)
    if float(np.max(x)) > 1.5:
        x = x / 255.0
    red, green, blue = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    peak = np.maximum(np.maximum(red, green), blue)
    delta = peak - np.minimum(np.minimum(red, green), blue)
    hue = np.zeros(peak.shape, dtype=np.float32)
    colored = delta > 1e-6
    red_max = colored & (peak == red)
    green_max = colored & (peak == green) & ~red_max
    blue_max = colored & ~red_max & ~green_max
    hue[red_max] = np.mod((green[red_max] - blue[red_max]) / delta[red_max], 6.0) / 6.0
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max] + 2.0) / 6.0
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0) / 6.0
    saturation = np.zeros(peak.shape, dtype=np.float32)
    nonzero = peak > 1e-6
    saturation[nonzero] = delta[nonzero] / peak[nonzero]
    index = np.clip((PALETTE_COLD_HUE - hue) / PALETTE_COLD_HUE, 0.0, 1.0)
    white = (saturation < PALETTE_WHITE_SAT) & (peak >= PALETTE_WHITE_VALUE)
    black = peak < PALETTE_BLACK_VALUE
    index = np.where(white, 1.0, index)
    return np.where(black, 0.0, index).astype(np.float32)


def _image_stats(path) -> list[float]:
    with Image.open(path) as img:
        rgb = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
        gray = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    index = palette_index(rgb)
    hot = index >= CEILING
    small = np.asarray(
        Image.fromarray((index * 255).astype(np.uint8)).resize((BLOB_SIZE, BLOB_SIZE), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    max_area = _blob_max_ratio(small >= CEILING)
    ys, xs = np.where(hot)
    if len(ys):
        cent_x = float(xs.mean() / max(index.shape[1] - 1, 1))
        cent_y = float(ys.mean() / max(index.shape[0] - 1, 1))
    else:
        cent_x = cent_y = float('nan')
    r_p95, r_p99, r_hot90 = _channel_stats(rgb[:, :, 0])
    g_p95, g_p99, g_hot90 = _channel_stats(rgb[:, :, 1])
    b_p95, b_p99, b_hot90 = _channel_stats(rgb[:, :, 2])
    return [
        float(gray.mean()),
        float(gray.std()),
        float(np.percentile(gray, 95)),
        float(np.percentile(gray, 99)),
        float((gray >= CEILING).mean()),
        r_p95, r_p99, r_hot90,
        g_p95, g_p99, g_hot90,
        b_p95, b_p99, b_hot90,
        float(np.percentile(index, 95)),
        float(np.percentile(index, 99)),
        float(hot.mean()),
        max_area,
        cent_x,
        cent_y,
    ]


def cpu_jobs(n_jobs: int | None = None) -> int:
    """공용 서버에서 이미지 피처가 코어를 전부 쓰지 않도록 기본 2개만 씁니다."""
    if n_jobs is None:
        return max(1, int(os.environ.get('UNID_N_JOBS', '2')))
    return n_jobs


def limit_torch_threads() -> None:
    torch, _ = _import_torch()
    threads = cpu_jobs()
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int | None = None) -> pd.DataFrame:
    rows = Parallel(n_jobs=cpu_jobs(n_jobs), batch_size=64)(delayed(_image_stats)(path) for path in paths)
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
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif isinstance(normalize, dict):
        mean, std = normalize['mean'], normalize['std']
    else:
        mean = std = None
    if mean is not None:
        mean_t = torch.tensor(mean, dtype=tensor.dtype)[:, None, None]
        std_t = torch.tensor(std, dtype=tensor.dtype)[:, None, None]
        tensor = (tensor - mean_t) / std_t
    return tensor


# ---------------------------------------------------------------------------
# 과거 방향 시계열. 시각을 정렬한 뒤 (t-window, t] 구간만 사용합니다.
# ---------------------------------------------------------------------------
def causal_frame_rows(frame: pd.DataFrame, offsets: tuple[float, ...] = (0.0, 30.0, 120.0)) -> np.ndarray:
    """현재, 30초 전, 120초 전 행 번호. 각 행의 시각보다 이후는 고르지 않습니다."""
    frame = frame.reset_index(drop=True)
    out = np.zeros((len(frame), len(offsets)), dtype=np.int64)
    times_all = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float)
    for _, positions in frame.groupby('experiment_id', sort=False).indices.items():
        positions = _sorted_positions(frame, np.asarray(positions))
        times = times_all[positions]
        for row in positions:
            now = times_all[row]
            for k, offset in enumerate(offsets):
                pos = int(np.searchsorted(times, now - offset, side='right') - 1)
                out[row, k] = int(positions[pos]) if pos >= 0 else int(row)
    return out


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
    """평균·최댓값과, 창 이전 마지막 관측 대비 변화·초당 기울기.

    창보다 과거 관측이 없으면 변화와 기울기는 0입니다. 아직 비교할 관측이 없다는 뜻이고,
    중앙값으로 채우면 실험 시작이 평범한 변화처럼 보이기 때문입니다.
    기울기는 명목 창 길이가 아니라 두 관측의 실제 시간 차로 나눕니다.
    """
    n = len(times)
    starts, lags = [], []
    ends = np.searchsorted(times, times, side='right')
    for window in windows:
        starts.append(np.searchsorted(times, times - window, side='right'))
        lags.append(np.searchsorted(times, times - window, side='right') - 1)
    means = [np.full(n, np.nan) for _ in windows]
    maxima = [np.full(n, np.nan) for _ in windows]
    deltas = [np.full(n, np.nan) for _ in windows]
    slopes = [np.full(n, np.nan) for _ in windows]
    lag_values = [_lag_means(times, values, lag) for lag in lags]
    for i in range(n):
        current = values[i]
        for k in range(len(windows)):
            window_values = values[starts[k][i]:ends[i]]
            window_values = window_values[~np.isnan(window_values)]
            if len(window_values):
                means[k][i] = window_values.mean()
                maxima[k][i] = window_values.max()
            lag = int(lags[k][i])
            if not np.isfinite(current):
                continue
            if lag < 0:
                deltas[k][i] = 0.0
                slopes[k][i] = 0.0
                continue
            previous = lag_values[k][i]
            if not np.isfinite(previous):
                continue
            deltas[k][i] = current - previous
            elapsed = float(times[i] - times[lag])
            slopes[k][i] = deltas[k][i] / elapsed if elapsed > 0 else 0.0
    return means, maxima, deltas, slopes


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
            means, maxima, deltas, slopes = _causal_multi(times, values, WINDOWS)
            start = _from_start(times, values)
            out.iloc[positions, out.columns.get_loc(f'{signal}_from_start')] = start
            for k, window in enumerate(WINDOWS):
                width = int(window)
                out.iloc[positions, out.columns.get_loc(f'{signal}_mean{width}s')] = means[k]
                out.iloc[positions, out.columns.get_loc(f'{signal}_max{width}s')] = maxima[k]
                out.iloc[positions, out.columns.get_loc(f'{signal}_delta{width}s')] = deltas[k]
                if signal in slope_signals:
                    out.iloc[positions, out.columns.get_loc(f'{signal}_slope{width}s')] = slopes[k]
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
    image = image_features.reindex(frame['sample_id'].astype(str).to_numpy()).reindex(columns=IMAGE_STATS)
    assert image['img_mean'].notna().all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[IMAGE_STATS] = image.to_numpy()
    signals = sensor_columns + IMAGE_WINDOW_SIGNALS
    slopes = [column for column in signals if _wants_slope(column)]
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
    def __init__(self, name: str, class_weight=None, seed: int = SEED, n_jobs: int = -1):
        self.name = name
        self.class_weight = class_weight
        self.seed = seed
        self.n_jobs = n_jobs
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
                subsample=0.8, colsample_bytree=0.8, random_state=self.seed, n_jobs=self.n_jobs,
                class_weight=weight, deterministic=True, force_row_wise=True, verbosity=-1,
            )
        if self.name == 'xgboost':
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=TREE_ESTIMATORS, learning_rate=TREE_LEARNING_RATE, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, objective='multi:softprob',
                random_state=self.seed, n_jobs=self.n_jobs, tree_method='hist', verbosity=0,
            )
        if self.name == 'catboost':
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=TREE_ESTIMATORS, learning_rate=TREE_LEARNING_RATE, depth=6,
                loss_function='MultiClass', random_seed=self.seed, verbose=False,
                allow_writing_files=False, thread_count=self.n_jobs,
            )
        if self.name == 'extratrees':
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=TREE_ESTIMATORS, random_state=self.seed, n_jobs=self.n_jobs, class_weight=weight,
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


class ExtraTreesModel:
    """탐색된 ExtraTrees. 결측은 학습 행의 중앙값으로 채우고, 확률 열은 1~6 순서입니다.

    test.py가 가중치를 읽을 수 있도록 이 모듈에 둡니다.
    """

    def __init__(self, params: dict, seed: int, n_jobs: int):
        self.params = dict(params)
        self.seed = seed
        self.n_jobs = n_jobs
        self.model = None
        self.median_ = None
        self._y_offset = 0

    def fit(self, features, y):
        from sklearn.ensemble import ExtraTreesClassifier
        y = np.asarray(y).astype(int)
        values = _as_float(features)
        median = np.nanmedian(values, axis=0)
        self.median_ = np.where(np.isnan(median), 0.0, median)
        self.model = ExtraTreesClassifier(
            n_estimators=int(self.params['n_estimators']),
            max_depth=self.params['max_depth'],
            min_samples_leaf=int(self.params['min_samples_leaf']),
            max_features=self.params['max_features'],
            class_weight=self.params['class_weight'],
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.model.fit(self._fill(values), y)
        return self

    def _fill(self, values: np.ndarray) -> np.ndarray:
        values = values.copy()
        missing = np.isnan(values)
        if missing.any():
            values[missing] = np.take(self.median_, np.where(missing)[1])
        return values

    def predict_proba(self, features) -> np.ndarray:
        proba = self.model.predict_proba(self._fill(_as_float(features)))
        return _align_proba(proba, self.model.classes_)

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
# 딥러닝. 현재 열화상과 센서 피처를 함께 쓰고, 시퀀스 모델은 과거 센서만 봅니다.
# ---------------------------------------------------------------------------
DEEP_TIMM = {
    'deit_small': 'deit_small_patch16_224.fb_in1k',
    'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
    'maxvit_tiny': 'maxvit_tiny_tf_224.in1k',
    'mobilevit_s': 'mobilevit_s.cvnets_in1k',
}


def deep_config(name: str) -> dict:
    """백본, 센서 시퀀스, 이미지 사용, 시점 수를 반환합니다. GRU는 새 실험에서 쓰지 않습니다."""
    if name == 'gru_efficientnet':
        return {'backbone': 'efficientnet', 'sequence': 'gru', 'image': True, 'frames': 1}
    if name == 'cnn1d':
        return {'backbone': None, 'sequence': 'cnn1d', 'image': False, 'frames': 1}
    if name == 'swin_t_time':
        return {'backbone': 'swin_t', 'sequence': 'none', 'image': True, 'frames': 3}
    if name in {'efficientnet', 'resnet18', 'resnet18_gapgmp', 'mobilenet', 'smallcnn', 'convnext_tiny', 'swin_t'} or name in DEEP_TIMM:
        return {'backbone': name, 'sequence': 'none', 'image': True, 'frames': 1}
    raise ValueError(f'딥러닝 모델이 아닙니다: {name}')


def deep_spec(name: str) -> tuple[str, bool]:
    spec = deep_config(name)
    return spec['backbone'] or name, spec['sequence'] == 'gru'


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


def _timm_normalize(net) -> dict:
    cfg = getattr(net, 'pretrained_cfg', {}) or {}
    mean = tuple(cfg.get('mean', (0.485, 0.456, 0.406)))
    std = tuple(cfg.get('std', (0.229, 0.224, 0.225)))
    return {'mean': mean, 'std': std}


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
    if name in {'resnet18', 'resnet18_gapgmp'}:
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        if name == 'resnet18':
            dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, dim, 'imagenet'

        class _GapGmp(nn.Module):
            def __init__(self, resnet):
                super().__init__()
                self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
                self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                self.avg, self.mx = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)

            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
                return torch.cat([self.avg(x).flatten(1), self.mx(x).flatten(1)], dim=1)

        return _GapGmp(net), 1024, 'imagenet'
    if name == 'mobilenet':
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, dim, 'imagenet'
    if name == 'convnext_tiny':
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = convnext_tiny(weights=weights)
        dim = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        return net, dim, 'imagenet'
    if name == 'swin_t':
        from torchvision.models import Swin_T_Weights, swin_t
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        net = swin_t(weights=weights)
        dim = net.head.in_features
        net.head = nn.Identity()
        return net, dim, 'imagenet'
    if name in DEEP_TIMM:
        import timm
        net = timm.create_model(DEEP_TIMM[name], pretrained=pretrained, num_classes=0, img_size=IMAGE_SIZE)
        return net, int(net.num_features), _timm_normalize(net)
    raise ValueError(name)


class FusionNet(object):
    def __new__(cls, image_dim: int, tabular_dim: int, sensor_dim: int, sequence: str, frames: int, head: str, backbone):
        torch, nn = _import_torch()

        class _Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.sequence = sequence
                self.frames = frames
                self.use_image = backbone is not None
                self.tabular_dim = tabular_dim
                self.use_gru = sequence == 'gru'
                self.head_kind = head
                self.freeze_backbone = False
                fused = 0
                if self.use_image:
                    self.img_proj = nn.Linear(image_dim, 128)
                    fused += 128
                    if frames > 1:
                        layer = nn.TransformerEncoderLayer(
                            d_model=128, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.1,
                        )
                        self.time_encoder = nn.TransformerEncoder(layer, num_layers=1)
                if tabular_dim:
                    self.tabular = nn.Sequential(nn.Linear(tabular_dim, 128), nn.ReLU(), nn.Dropout(0.2))
                    fused += 128
                if sequence == 'gru':
                    self.gru = nn.GRU(sensor_dim, 128, batch_first=True)
                    fused += 128
                if sequence == 'cnn1d':
                    self.conv1d = nn.Sequential(
                        nn.Conv1d(sensor_dim, 64, 3, padding=1), nn.ReLU(),
                        nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
                    )
                    fused += 128
                self.classifier = nn.Linear(fused, 5 if head == 'coral' else 6)

            def train(self, mode: bool = True):
                super().train(mode)
                if self.freeze_backbone and self.backbone is not None:
                    self.backbone.eval()
                return self

            def _encode(self, image):
                return self.img_proj(self.backbone(image))

            def forward(self, image, tabular, seq=None, mask=None):
                parts = []
                if self.use_image:
                    if self.frames == 1:
                        parts.append(self._encode(image))
                    else:
                        tokens = torch.stack([self._encode(image[:, t]) for t in range(image.size(1))], dim=1)
                        parts.append(self.time_encoder(tokens).mean(dim=1))
                if self.tabular_dim:
                    parts.append(self.tabular(tabular))
                if self.sequence == 'gru':
                    lengths = mask.sum(dim=1).clamp(min=1).cpu()
                    packed = nn.utils.rnn.pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
                    _, hidden = self.gru(packed)
                    parts.append(hidden[-1])
                if self.sequence == 'cnn1d':
                    hidden = self.conv1d((seq * mask.unsqueeze(-1)).transpose(1, 2))
                    weight = mask.unsqueeze(1)
                    parts.append((hidden * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1))
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
    spec = deep_config(config['model_name'])
    backbone_name = config.get('backbone', spec['backbone'])
    sequence = config.get('sequence', spec['sequence'])
    frames = int(config.get('frames', spec['frames']))
    if backbone_name is None:
        backbone, image_dim, normalize = None, 0, 'none'
    else:
        backbone, image_dim, normalize = build_backbone(backbone_name, pretrained=pretrained)
    network = FusionNet(
        image_dim, int(config['tabular_dim']), int(config['sensor_dim']),
        sequence, frames, config['head'], backbone,
    )
    default_freeze = backbone_name not in (None, 'smallcnn')
    network.freeze_backbone = config.get('freeze_backbone', default_freeze) and backbone is not None
    if network.freeze_backbone:
        for parameter in network.backbone.parameters():
            parameter.requires_grad = False
    return network, normalize


PRETRAINED_MODELS = {
    'efficientnet': 'torchvision efficientnet_b0, ImageNet-1K V1, BSD-3-Clause',
    'resnet18': 'torchvision resnet18, ImageNet-1K V1, BSD-3-Clause',
    'resnet18_gapgmp': 'torchvision resnet18, ImageNet-1K V1, BSD-3-Clause. 평균 풀링과 최대 풀링을 이어 붙임',
    'mobilenet': 'torchvision mobilenet_v3_small, ImageNet-1K V1, BSD-3-Clause',
    'convnext_tiny': 'torchvision convnext_tiny, ImageNet-1K V1, BSD-3-Clause',
    'swin_t': 'torchvision swin_t, ImageNet-1K V1, BSD-3-Clause',
    'swin_t_time': 'torchvision swin_t, ImageNet-1K V1, BSD-3-Clause. 현재·30초 전·120초 전 세 시점',
    'deit_small': 'timm deit_small_patch16_224.fb_in1k, ImageNet-1K, Apache-2.0',
    'dinov2_vits14': 'timm vit_small_patch14_dinov2.lvd142m, LVD-142M, Apache-2.0. 입력 224',
    'maxvit_tiny': 'timm maxvit_tiny_tf_224.in1k, ImageNet-1K, Apache-2.0',
    'mobilevit_s': 'timm mobilevit_s.cvnets_in1k, ImageNet-1K, Apache-2.0',
    'gru_efficientnet': 'torchvision efficientnet_b0, ImageNet-1K V1, BSD-3-Clause + 과거 센서 GRU',
    'cnn1d': '사전학습 가중치 없음. 과거 300초 센서만 사용',
    'smallcnn': '사전학습 가중치 없음',
}


def _needs_image(bundle: dict) -> bool:
    if bundle.get('kind') == 'blend':
        return any(_needs_image(child) for child in bundle['bundles'])
    if bundle.get('kind') == 'tabular':
        return True
    return bool(bundle.get('config', {}).get('use_image', True))


def prepare_frame(frame: pd.DataFrame, roots: list[Path]):
    """입력 행만으로 경로와 피처를 만듭니다. 조합에서 이미지를 한 번만 읽습니다."""
    frame = frame.reset_index(drop=True)
    paths = resolve_image_paths(frame, roots)
    features = build_features(frame, extract_image_features(frame, paths))
    return frame, paths, features


def predict_proba(bundle: dict, frame: pd.DataFrame, roots: list[Path], prepared=None, batch_size: int = 32) -> np.ndarray:
    """행 순서를 유지한 채 클래스 1~6 확률을 반환합니다. 학습은 하지 않습니다."""
    if bundle.get('kind') == 'blend':
        if prepared is None and _needs_image(bundle):
            prepared = prepare_frame(frame, roots)
        elif prepared is None:
            prepared = (frame.reset_index(drop=True), None, None)
        parts = [predict_proba(child, prepared[0], roots, prepared=prepared, batch_size=batch_size) for child in bundle['bundles']]
        stacked = np.stack(parts, axis=0)
        if bundle.get('stack') is not None:
            aligned = _align_proba(bundle['stack'].predict_proba(np.concatenate(parts, axis=1)), bundle['stack'].classes_)
            return aligned
        weights = np.asarray(bundle['weights'], dtype=np.float64)
        weights = weights / weights.sum()
        return np.tensordot(weights, stacked, axes=(0, 0))
    if prepared is None:
        if _needs_image(bundle):
            frame, paths, features = prepare_frame(frame, roots)
        else:
            frame, paths, features = frame.reset_index(drop=True), None, None
    else:
        frame, paths, features = prepared
    if bundle['kind'] == 'tabular':
        return bundle['estimator'].predict_proba(features.reindex(columns=bundle['features']))
    return _predict_torch(bundle, frame, paths, features, batch_size)


def predict_labels(bundle: dict, frame: pd.DataFrame, roots: list[Path], batch_size: int = 32) -> pd.Series:
    """입력 프레임의 행만으로 1~6 라벨을 예측합니다. 학습은 하지 않습니다."""
    frame = frame.reset_index(drop=True)
    proba = predict_proba(bundle, frame, roots, batch_size=batch_size)
    labels = (np.argmax(proba, axis=1) + 1).astype(int)
    return pd.Series(labels, index=frame['sample_id'].astype(str).to_numpy())


def _sensor_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    present = frame.reindex(columns=columns)
    return numeric(present, columns).to_numpy(dtype=np.float64)


def _predict_torch(bundle: dict, frame: pd.DataFrame, paths, features, batch_size: int) -> np.ndarray:
    torch, _ = _import_torch()
    limit_torch_threads()
    config = bundle['config']
    network, normalize = build_fusion(config, pretrained=False)
    network.load_state_dict(bundle['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    network.eval()
    use_image = bool(config.get('use_image', True))
    frames = int(config.get('frames', 1))
    sequence = config.get('sequence', 'gru' if config.get('use_gru') else 'none')
    tabular = None
    if use_image:
        tabular = bundle['tabular_prep'].transform(features.reindex(columns=bundle['features'])).astype(np.float32)
    sequences = mask = None
    if sequence != 'none':
        sensor = bundle['sequence_prep'].transform(_sensor_matrix(frame, bundle['sequence_cols'])).astype(np.float32)
        sequences, mask = build_sequences(frame, sensor)
    frame_index = causal_frame_rows(frame) if frames > 1 else None
    proba_out = np.zeros((len(frame), len(LABELS)), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            images = table = seq = mask_batch = None
            if use_image:
                if frames == 1:
                    images = torch.stack([
                        load_image_tensor(paths.iloc[i], normalize=normalize, flip=False) for i in range(start, stop)
                    ]).to(device)
                else:
                    shots = []
                    for row in range(start, stop):
                        shots.append(torch.stack([
                            load_image_tensor(paths.iloc[int(frame_index[row, slot])], normalize=normalize, flip=False)
                            for slot in range(frames)
                        ]))
                    images = torch.stack(shots).to(device)
                table = torch.from_numpy(tabular[start:stop]).to(device)
            if sequences is not None:
                seq = torch.from_numpy(sequences[start:stop]).to(device)
                mask_batch = torch.from_numpy(mask[start:stop]).to(device)
            logits = network(images, table, seq, mask_batch)
            proba_out[start:stop] = logits_to_proba(logits, config['head']).cpu().numpy()
    return proba_out
