"""Uni-D 배터리 열폭주 6단계 분류 — 공통 모듈.

구조
  1) 순간의 단서 : 센서·가스·온도 값, 실험 조건, 열화상 이미지 통계
  2) 과거 요약   : 같은 실험의 현재·이전 관측만으로 만든 시간 피처 (초 단위)
  3) 판정        : LightGBM 6단계 분류
  4) 순서 보정   : "단계는 앞으로만 진행한다"를 넣은 순방향 필터 (직전까지의 예측만 사용)

규정 준수
  - 피처와 보정은 실험마다 따로, 시간순으로, 현재 시점까지의 행만으로 계산한다.
  - 실험 전체 통계(전체 평균·최댓값·길이)와 다른 실험의 행은 쓰지 않는다.
  - 입력 행 순서와 무관하게 같은 결과가 나오도록 내부에서 정렬한다.

train.py, test.py, selfcheck.py가 모두 이 파일의 함수를 쓴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

SEED = 42
LABELS = [1, 2, 3, 4, 5, 6]
N_STAGES = len(LABELS)

GASES = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
         'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
# 제외한 컬럼: meas_a(모든 행이 0), indu_t.1(test에서 전부 결측), meas_t1_plus_t2(t1+t2와 같은 정보)
SENSORS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'atmp_t', 'thermal_t', 'indu_t'] + GASES
CONDITIONS = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
DERIVED = ['gas_sum', 'temp_hi', 'temp_gap']
IMAGE_STATS = ['img_r_mean', 'img_g_mean', 'img_b_mean', 'img_r_std', 'img_g_std', 'img_b_std',
               'img_l_mean', 'img_l_std', 'img_l_p10', 'img_l_p50', 'img_l_p90',
               'img_hot50', 'img_hot70', 'img_hot80', 'img_cold20', 'img_g_frac',
               'img_cx', 'img_cy', 'img_spread', 'img_block_max', 'img_block_std']
# 시간 피처를 만들 신호: 센서 + 이미지에서 뽑은 값
SIGNALS = ['meas_v', 'meas_t1', 'meas_t2', 'temp_hi', 'meas_p', 'thermal_t', 'indu_t', 'gas_sum',
           'img_l_mean', 'img_g_mean', 'img_b_mean', 'img_hot80', 'img_l_std']
# 과거 윈도우 길이(초). 행 수가 아니라 relative_time_s 기준. test 간격이 15초까지 넓어 30초보다 짧게 잡지 않는다.
WINDOWS = [30, 60, 120, 300, 600]


# ---------------------------------------------------------------------------
# 1. 이미지
# ---------------------------------------------------------------------------
def resolve_image_paths(frame: pd.DataFrame, roots: list[Path]) -> pd.Series:
    """CSV의 image_path를 실제 파일 경로로 바꾼다. roots를 순서대로 시도하고, 실패하면 파일명으로 찾는다."""
    rel = frame['image_path'].astype(str)
    for root in roots:
        candidate = rel.map(lambda p: Path(root) / p)
        if candidate.map(Path.is_file).all():
            return candidate
    index = {}
    for root in roots:
        for path in Path(root).rglob('*'):
            if path.is_file():
                index.setdefault(path.name, path)
    candidate = rel.map(lambda p: index.get(Path(p).name))
    missing = int(candidate.isna().sum())
    assert missing == 0, f'이미지 {missing}개를 찾지 못했습니다. image_root를 확인하십시오.'
    return candidate


def _image_stats(path) -> list[float]:
    """이미지 한 장의 통계. 이미지 크기가 실험마다 달라서 크기와 무관한 값(비율·백분위·정규화 좌표)만 쓴다.

    열화상은 프레임마다 색 범위가 자동으로 맞춰진 색 팔레트 이미지라서, 절대 온도가 아니라
    "프레임 안에서 열이 얼마나 넓게, 어디에 몰려 있는가"를 담는다.
    """
    with Image.open(path) as img:
        rgb = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    h, w = lum.shape
    p10, p50, p90 = np.percentile(lum, [10, 50, 90])
    ys, xs = np.nonzero(lum >= p90)                     # 프레임 안에서 가장 뜨거운 쪽 픽셀
    sx, sy = max(w - 1, 1), max(h - 1, 1)
    blocks = np.array([cell.mean() for row in np.array_split(lum, 4, axis=0)
                       for cell in np.array_split(row, 4, axis=1) if cell.size])
    values = [
        r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std(),
        lum.mean(), lum.std(), p10, p50, p90,
        (lum >= 0.5).mean(), (lum >= 0.7).mean(), (lum >= 0.8).mean(), (lum <= 0.2).mean(), (g >= 0.5).mean(),
        xs.mean() / sx, ys.mean() / sy, np.sqrt(xs.var() / sx ** 2 + ys.var() / sy ** 2),
        blocks.max() - lum.mean(), blocks.std(),
    ]
    return [float(v) for v in values]


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    """sample_id를 인덱스로 하는 이미지 통계 표. 이미지마다 따로 계산하므로 다른 행의 정보가 섞이지 않는다."""
    rows = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_image_stats)(p) for p in paths)
    index = pd.Index(frame['sample_id'].astype(str).to_numpy(), name='sample_id')
    return pd.DataFrame(rows, columns=IMAGE_STATS, index=index)


# ---------------------------------------------------------------------------
# 2. 과거 방향 시간 피처 (한 실험, 시간순)
# ---------------------------------------------------------------------------
def _causal_one_experiment(t: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """t: 경과 시간(초, 오름차순), x: (행 수, 신호 수). 각 행은 자기 자신과 그 이전 행만 본다."""
    n = len(t)
    blocks = {}
    cmax = np.fmax.accumulate(x, axis=0)                 # 지금까지의 최댓값
    cmin = np.fmin.accumulate(x, axis=0)                 # 지금까지의 최솟값
    blocks['d0'] = x - x[0]                              # 첫 관측과의 차이
    blocks['cmax'] = cmax
    blocks['cmin'] = cmin
    blocks['rmax'] = x - cmax                            # 최고점에서 얼마나 내려왔나
    blocks['rmin'] = x - cmin                            # 최저점에서 얼마나 올라왔나
    last_peak = np.maximum.accumulate(np.where(x >= cmax, np.arange(n)[:, None], 0), axis=0)
    since_peak = t[:, None] - t[last_peak]               # 최고점을 찍은 뒤 흐른 시간(초)
    since_peak[np.isnan(cmax)] = np.nan
    blocks['tmax'] = since_peak

    rate0 = np.full_like(x, np.nan)                      # 직전 관측 대비 초당 변화
    if n > 1:
        dt = np.diff(t)
        ok = dt > 0
        rate0[1:][ok] = (x[1:][ok] - x[:-1][ok]) / dt[ok][:, None]
    blocks['rate0'] = rate0

    rolling_frame = pd.DataFrame(x, index=pd.to_timedelta(t, unit='s'))
    for w in WINDOWS:
        roll = rolling_frame.rolling(f'{w}s')            # (t-w, t] 구간: 현재와 과거만
        blocks[f'mean{w}'] = roll.mean().to_numpy()
        blocks[f'max{w}'] = roll.max().to_numpy()
        blocks[f'min{w}'] = roll.min().to_numpy()
        lag = np.searchsorted(t, t - w, side='right') - 1   # t-w 이하의 마지막 관측
        ok = lag >= 0
        rate = np.full_like(x, np.nan)
        rate[ok] = (x[ok] - x[lag[ok]]) / (t[ok] - t[lag[ok]])[:, None]
        blocks[f'rate{w}'] = rate                        # w초 전 대비 초당 변화
    return blocks


# ---------------------------------------------------------------------------
# 3. 전체 피처 조립
# ---------------------------------------------------------------------------
def make_dataset(frame: pd.DataFrame, image_features: pd.DataFrame, step: int = 1):
    """frame에 들어 있는 행만으로 피처를 만든다.

    반환: (X, meta). 둘 다 실험·시간순으로 정렬된 같은 행 순서다.
    step > 1이면 실험마다 step행에 하나만 남긴 뒤 피처를 만든다(넓은 관측 간격을 흉내 내는 학습용 복사본).
    """
    base = pd.DataFrame({
        'sample_id': frame['sample_id'].astype(str).to_numpy(),
        'experiment_id': frame['experiment_id'].astype(str).to_numpy(),
        'relative_time_s': pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float),
    })
    assert not base['relative_time_s'].isna().any(), 'relative_time_s에 빈 값이 있습니다.'
    for c in SENSORS + CONDITIONS:
        base[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    img = image_features.reindex(base['sample_id'].to_numpy())
    assert img.notna().all(axis=1).all(), '이미지 피처가 없는 샘플이 있습니다.'
    for c in IMAGE_STATS:
        base[c] = img[c].to_numpy(dtype=float)
    if 'label' in frame.columns:
        base['label'] = pd.to_numeric(frame['label'], errors='raise').to_numpy(dtype=int)

    base['gas_sum'] = base[GASES].sum(axis=1)
    base['temp_hi'] = np.fmax(base['meas_t1'], base['meas_t2'])
    base['temp_gap'] = base['meas_t1'] - base['meas_t2']

    # 입력 행 순서와 무관하게 항상 같은 순서가 되도록 sample_id까지 넣어 정렬한다.
    base = base.sort_values(['experiment_id', 'relative_time_s', 'sample_id'], kind='mergesort').reset_index(drop=True)
    if step > 1:
        base = base[base.groupby('experiment_id').cumcount() % step == 0].reset_index(drop=True)

    names = None
    causal = np.empty((len(base), 0))
    parts = []
    t_all = base['relative_time_s'].to_numpy()
    x_all = base[SIGNALS].to_numpy(dtype=float)
    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        blocks = _causal_one_experiment(t_all[idx], x_all[idx])
        if names is None:
            names = [f'{s}_{k}' for k in blocks for s in SIGNALS]
        parts.append((idx, np.hstack(list(blocks.values()))))
    if parts:
        causal = np.empty((len(base), len(names)))
        for idx, values in parts:
            causal[idx] = values

    static = ['relative_time_s'] + SENSORS + CONDITIONS + DERIVED + IMAGE_STATS
    X = pd.concat([base[static], pd.DataFrame(causal, columns=names, index=base.index)], axis=1)
    meta = base[['sample_id', 'experiment_id', 'relative_time_s'] + (['label'] if 'label' in base.columns else [])]
    return X, meta


# ---------------------------------------------------------------------------
# 4. 모델
# ---------------------------------------------------------------------------
def build_model(n_estimators: int = 300):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        objective='multiclass', n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
        class_weight='balanced',         # Macro F1: 수가 적은 4·5·6단계도 같은 비중으로 학습
        random_state=SEED, n_jobs=-1, verbose=-1, deterministic=True, force_row_wise=True,
    )


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """항상 (행 수, 6) 모양으로 1~6단계 확률을 돌려준다."""
    prob = np.zeros((len(X), N_STAGES))
    if len(X):
        raw = model.predict_proba(X)
        for j, c in enumerate(model.classes_):
            prob[:, LABELS.index(int(c))] = raw[:, j]
    return prob


# ---------------------------------------------------------------------------
# 5. 순서 보정 (순방향 필터)
# ---------------------------------------------------------------------------
def transition_matrix(meta: pd.DataFrame, pseudo: float = 0.5) -> np.ndarray:
    """학습 데이터의 라벨에서 '1초 동안 다음 단계로 넘어갈 확률'을 구한다.

    단계는 뒤로 가지 않으므로 앞으로 한 칸 또는 두 칸(건너뛰기)만 허용한다.
    """
    count = np.zeros((N_STAGES, N_STAGES))
    seconds = np.zeros(N_STAGES)
    for _, g in meta.groupby('experiment_id', sort=False):
        g = g.sort_values('relative_time_s', kind='mergesort')
        lab = g['label'].to_numpy(dtype=int) - 1
        t = g['relative_time_s'].to_numpy(dtype=float)
        np.add.at(seconds, lab[:-1], np.diff(t))
        move = lab[1:] != lab[:-1]
        np.add.at(count, (lab[:-1][move], lab[1:][move]), 1)
    T = np.zeros((N_STAGES, N_STAGES))
    for i in range(N_STAGES):
        for j in (i + 1, i + 2):
            if j < N_STAGES:
                T[i, j] = (count[i, j] + pseudo) / max(seconds[i], 1.0)
        T[i, i] = 1.0 - T[i].sum()
    return T


def forward_filter(prob: np.ndarray, t: np.ndarray, T: np.ndarray, gamma: float) -> np.ndarray:
    """한 실험(시간순)의 확률을 '직전까지의 판단 + 현재 확률'로 고친다. 미래 행은 쓰지 않는다."""
    n = len(t)
    evidence = np.clip(prob, 1e-4, 1.0) ** gamma
    alpha = np.full(N_STAGES, 1e-3)
    alpha[0] = 1.0                                       # 모든 실험은 1단계에서 시작한다
    alpha /= alpha.sum()
    powers = {}
    out = np.empty((n, N_STAGES))
    for i in range(n):
        if i > 0:
            dt = max(int(round(t[i] - t[i - 1])), 0)
            if dt not in powers:
                powers[dt] = np.linalg.matrix_power(T, dt)
            alpha = alpha @ powers[dt]
        alpha = np.maximum(alpha * evidence[i], 1e-12)   # 바닥값: 오경보 뒤에도 되돌아올 수 있게
        alpha = alpha / alpha.sum()
        out[i] = alpha
    return out


def decode(prob: np.ndarray, meta: pd.DataFrame, T, mode: str, gamma: float) -> np.ndarray:
    """prob: meta와 같은 행 순서(실험·시간순). 반환: 1~6 라벨.

    mode = 'raw'(보정 없음) / 'filter'(순방향 필터) / 'filter_latch'(필터 뒤 한 번 올라간 단계는 유지)
    T는 행렬 하나이거나 {experiment_id: 행렬} (교차검증용).
    """
    if mode == 'raw':
        return prob.argmax(axis=1) + 1
    out = np.empty(len(meta), dtype=int)
    t = meta['relative_time_s'].to_numpy(dtype=float)
    for exp, idx in meta.groupby('experiment_id', sort=False).indices.items():
        matrix = T[exp] if isinstance(T, dict) else T
        lab = forward_filter(prob[idx], t[idx], matrix, gamma).argmax(axis=1) + 1
        out[idx] = np.maximum.accumulate(lab) if mode == 'filter_latch' else lab
    return out
