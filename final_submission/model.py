"""Uni-D 배터리 열폭주 6단계 분류 — 최종 모델 공통 모듈.

구조
  1) 순간의 단서 : 센서·가스·온도 값, 실험 조건, 열화상 온도 지수 평균(th_mean) 하나
  2) 과거 요약   : 같은 실험의 현재·이전 관측만으로 만든 시간 피처 (relative_time_s 기준, 초 단위 창)
  3) 판정        : XGBoost 6단계 분류 (행 가중치로 단계 균형)
  4) 순서 보정   : 단계는 앞으로만 진행한다는 순방향 필터 (직전까지의 예측만 사용)

규정 준수
  - 피처와 보정은 실험마다 따로, 시간순으로, 현재 시점까지의 행만으로 계산한다.
  - 실험 전체 통계(전체 평균·최댓값·길이·순위)와 다른 실험의 행은 쓰지 않는다.
  - 입력 행 순서와 무관하게 같은 결과가 나오도록 내부에서 정렬한다.

출처: 시간 피처·순서 보정은 팀원(jangminjun) 코드를 그대로 옮겼고(v1 피처, 최신 decode), 열화상 온도 지수는
     yeonwoo 실험(v04 역변환, v44·v47 이미지 축소)에서 가져왔다.
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
N_THREADS = 8

GASES = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
         'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
# 제외한 컬럼: meas_a(모든 행이 0), indu_t.1(test에서 전부 결측), meas_t1_plus_t2(t1+t2와 같은 정보)
SENSORS = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'atmp_t', 'thermal_t', 'indu_t'] + GASES
CONDITIONS = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
DERIVED = ['gas_sum', 'temp_hi', 'temp_gap']
SIGNALS = ['meas_v', 'meas_t1', 'meas_t2', 'temp_hi', 'meas_p', 'thermal_t', 'indu_t', 'gas_sum']
IMAGE_SIGNAL = 'th_mean'
WINDOWS = [30, 60, 120, 300, 600]
STEPS = (1, 2, 3)        # 학습용 복사본: 원본(5초) + 2행·3행마다 하나만 남긴 것(10초·15초). Test 간격이 15초까지 섞여 있다.


# ---------------------------------------------------------------------------
# 1. 열화상 → 온도 지수
# ---------------------------------------------------------------------------
# PNG는 256색 컬러맵을 입힌 8비트 값이다(전체 데이터에서 색이 정확히 256개). thermal_palette.npy는 그 컬러맵을
# 차가움(0) → 뜨거움(255) 순서로 복원한 표다. 프레임마다 범위가 자동으로 맞춰져 있어 값은 상대 온도다.
_PAL = np.load(Path(__file__).with_name('thermal_palette.npy')).astype(np.int64)
_LUT = np.zeros(256 ** 3, np.uint8)
_LUT[(_PAL[:, 0] << 16) | (_PAL[:, 1] << 8) | _PAL[:, 2]] = np.arange(256, dtype=np.uint8)


def thermal_index(path) -> np.ndarray:
    x = np.asarray(Image.open(path).convert('RGB')).astype(np.int64)
    return _LUT[(x[..., 0] << 16) | (x[..., 1] << 8) | x[..., 2]]


def _th_mean(path) -> float:
    return float(thermal_index(path).astype(np.float32).mean())


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


def extract_image_features(frame: pd.DataFrame, paths: pd.Series, n_jobs: int = -1) -> pd.Series:
    """sample_id → th_mean. 이미지마다 따로 계산하므로 다른 행의 정보가 섞이지 않는다."""
    vals = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_th_mean)(p) for p in paths)
    return pd.Series(vals, index=pd.Index(frame['sample_id'].astype(str).to_numpy(), name='sample_id'), name=IMAGE_SIGNAL)


# ---------------------------------------------------------------------------
# 2. 과거 방향 시간 피처 (한 실험, 시간순) — jangminjun v1 그대로
# ---------------------------------------------------------------------------
def _causal_one_experiment(t: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    n = len(t)
    blocks = {}
    cmax = np.fmax.accumulate(x, axis=0)
    cmin = np.fmin.accumulate(x, axis=0)
    blocks['d0'] = x - x[0]                              # 첫 관측과의 차이
    blocks['cmax'] = cmax                                # 지금까지의 최댓값
    blocks['cmin'] = cmin                                # 지금까지의 최솟값
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
        lag = np.searchsorted(t, t - w, side='right') - 1
        ok = lag >= 0
        rate = np.full_like(x, np.nan)
        rate[ok] = (x[ok] - x[lag[ok]]) / (t[ok] - t[lag[ok]])[:, None]
        blocks[f'rate{w}'] = rate                        # w초 전 대비 초당 변화
    return blocks


def _causal(base: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    t_all = base['relative_time_s'].to_numpy()
    x_all = base[signals].to_numpy(dtype=float)
    names, out = None, None
    for _, idx in base.groupby('experiment_id', sort=False).indices.items():
        blocks = _causal_one_experiment(t_all[idx], x_all[idx])
        if names is None:
            names = [f'{s}_{k}' for k in blocks for s in signals]
            out = np.empty((len(base), len(names)))
        out[idx] = np.hstack(list(blocks.values()))
    return pd.DataFrame(out, columns=names, index=base.index)


def make_dataset(frame: pd.DataFrame, th: pd.Series, step: int = 1):
    """frame에 들어 있는 행만으로 피처를 만든다. 반환 (X, meta): 실험·시간순으로 정렬된 같은 행 순서.

    step > 1이면 실험마다 step행에 하나만 남긴 뒤 피처를 만든다(넓은 관측 간격을 흉내 내는 학습용 복사본).
    meta에는 순서 보정 규칙 옵션('+r')이 쓰는 원본 값(meas_t1, meas_v, tr_method)도 담는다.
    """
    base = pd.DataFrame({
        'sample_id': frame['sample_id'].astype(str).to_numpy(),
        'experiment_id': frame['experiment_id'].astype(str).to_numpy(),
        'relative_time_s': pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(dtype=float),
    })
    assert not base['relative_time_s'].isna().any(), 'relative_time_s에 빈 값이 있습니다.'
    for c in SENSORS + CONDITIONS:
        base[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    img = th.reindex(base['sample_id'].to_numpy())
    assert img.notna().all(), '이미지 피처가 없는 샘플이 있습니다.'
    base[IMAGE_SIGNAL] = img.to_numpy(dtype=float)
    if 'label' in frame.columns:
        base['label'] = pd.to_numeric(frame['label'], errors='raise').to_numpy(dtype=int)
    base['gas_sum'] = base[GASES].sum(axis=1)
    base['temp_hi'] = np.fmax(base['meas_t1'], base['meas_t2'])
    base['temp_gap'] = base['meas_t1'] - base['meas_t2']

    base = base.sort_values(['experiment_id', 'relative_time_s', 'sample_id'], kind='mergesort').reset_index(drop=True)
    if step > 1:
        base = base[base.groupby('experiment_id').cumcount() % step == 0].reset_index(drop=True)

    static = ['relative_time_s'] + SENSORS + CONDITIONS + DERIVED
    X = pd.concat([base[static], _causal(base, SIGNALS), base[[IMAGE_SIGNAL]], _causal(base, [IMAGE_SIGNAL])], axis=1)
    meta = base[['sample_id', 'experiment_id', 'relative_time_s', 'meas_t1', 'meas_v', 'tr_method']
                + (['label'] if 'label' in base.columns else [])]
    return X, meta


# ---------------------------------------------------------------------------
# 3. 모델
# ---------------------------------------------------------------------------
class StageXGB:
    """XGBoost 6단계 분류. class_weight가 없어 행 가중치로 단계 균형('balanced')을 맞춘다.
    GPU hist로 학습했다(재현은 같은 설정의 GPU에서 확인). GPU가 없으면 device='cpu'(결과가 조금 달라질 수 있음)."""

    def __init__(self, n_estimators: int = 300, seed: int = SEED, device: str = 'cuda'):
        self.n_estimators, self.seed, self.device = n_estimators, seed, device

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier
        self.classes_ = np.unique(y)
        self.model = XGBClassifier(
            objective='multi:softprob', n_estimators=self.n_estimators, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, reg_lambda=1.0, tree_method='hist', device=self.device,
            max_depth=6, min_child_weight=1, random_state=self.seed, n_jobs=N_THREADS)
        self.model.fit(X.replace([np.inf, -np.inf], np.nan), np.searchsorted(self.classes_, y),
                       sample_weight=compute_sample_weight('balanced', y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prob = np.zeros((len(X), N_STAGES))
        if len(X):
            prob[:, np.asarray(self.classes_) - 1] = self.model.predict_proba(X.replace([np.inf, -np.inf], np.nan))
        return prob


# ---------------------------------------------------------------------------
# 4. 순서 보정 — jangminjun 최신 model.py 그대로
# ---------------------------------------------------------------------------
def transition_matrix(meta: pd.DataFrame, pseudo: float = 0.5) -> np.ndarray:
    """학습 라벨에서 '1초 동안 다음 단계로 넘어갈 확률'. 단계는 뒤로 가지 않으므로 한 칸 또는 두 칸(건너뛰기)만 허용."""
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


RULE_COLS = ['meas_t1', 'meas_v', 'tr_method']
RULE_T1_STAGE2 = 45.0    # meas_t1이 이 값에 닿기 전에는 1단계, 닿은 뒤에는 2단계 이상
RULE_T1_STAGE3 = 50.0    # meas_t1이 이 값에 닿기 전에는 2단계 이하
RULE_V_DROP = 0.80       # 가열 실험에서 전압이 지금까지 최고 전압의 이 비율 아래로 떨어지면 4단계 이상
RULE_V_MIN = 1.0         # 최고 전압이 이보다 낮으면(센서 값이 0인 경우 등) 전압 규칙을 쓰지 않는다
FAST_FROM = 4            # '+fG' 옵션: 이 단계 이상으로 넘어가는 전환만 빠른 필터(gamma=G)로 앞당긴다


def stage_bounds(t1: np.ndarray, v: np.ndarray, method: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """한 실험(시간순)의 각 행에서 가능한 단계 범위 (lo, hi), 0부터 센 번호. 현재와 이전 행만 본다."""
    n = len(t1)
    lo = np.zeros(n, dtype=int)
    hi = np.full(n, N_STAGES - 1, dtype=int)
    with np.errstate(invalid='ignore'):
        t1_max = np.fmax.accumulate(t1)
        hi[t1_max < RULE_T1_STAGE3] = 1
        hi[t1_max < RULE_T1_STAGE2] = 0
        lo[t1_max >= RULE_T1_STAGE2] = 1
        v_max = np.fmax.accumulate(v)
        dropped = np.maximum.accumulate((v <= RULE_V_DROP * v_max) & (v_max >= RULE_V_MIN))
        lo[dropped & (method == 1) & (t1_max >= RULE_T1_STAGE3)] = 3
    return lo, hi


def forward_filter(prob: np.ndarray, t: np.ndarray, T: np.ndarray, gamma: float,
                   lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> np.ndarray:
    """한 실험(시간순)의 확률을 '직전까지의 판단 + 현재 확률'로 고친다. 미래 행은 쓰지 않는다."""
    n = len(t)
    evidence = np.clip(prob, 1e-4, 1.0) ** gamma
    alpha = np.full(N_STAGES, 1e-3)
    alpha[0] = 1.0                                       # 모든 실험은 1단계에서 시작한다
    alpha /= alpha.sum()
    powers = {}
    stage = np.arange(N_STAGES)
    out = np.empty((n, N_STAGES))
    for i in range(n):
        if i > 0:
            dt = max(int(round(t[i] - t[i - 1])), 0)
            if dt not in powers:
                powers[dt] = np.linalg.matrix_power(T, dt)
            alpha = alpha @ powers[dt]
        alpha = np.maximum(alpha * evidence[i], 1e-12)   # 바닥값: 오경보 뒤에도 되돌아올 수 있게
        if lo is not None:
            alpha = np.where((stage >= lo[i]) & (stage <= hi[i]), alpha, 0.0)
        alpha = alpha / alpha.sum()
        out[i] = alpha
    return out


def _confirmed(lab: np.ndarray, t: np.ndarray, hold: float) -> np.ndarray:
    """'k단계 이상' 예측이 hold초 이상 끊기지 않고 이어진 뒤에만 그 단계를 확정한다. 과거만 본다."""
    n = len(lab)
    held = np.zeros(n, dtype=int)
    pos = np.arange(n)
    for k in range(2, N_STAGES + 1):
        on = lab >= k
        starts = on & ~np.r_[False, on[:-1]]
        first = np.maximum.accumulate(np.where(starts, pos, 0))
        held[np.maximum.accumulate(on & (t - t[first] >= hold))] = k
    return held


def _parse_mode(mode: str) -> tuple[str, float | None, bool]:
    """'filter_hold30+f3+r' → ('filter_hold30', 3.0, True)."""
    base, *options = mode.split('+')
    fast, use_rules = None, False
    for option in options:
        if option == 'r':
            use_rules = True
        elif option.startswith('f'):
            fast = float(option[1:])
        else:
            raise ValueError(f'알 수 없는 보정 옵션: {option}')
    return base, fast, use_rules


def decode(prob: np.ndarray, meta: pd.DataFrame, T, mode: str, gamma: float = 1.0) -> np.ndarray:
    """prob: meta와 같은 행 순서(실험·시간순). 반환: 1~6 라벨.

    mode: 'filter_latch'(필터 뒤 한 번 올라간 단계는 유지) / 'filter_holdS'(S초 이상 이어진 단계만 유지) 등.
    옵션 '+fG': 4단계 이상으로의 전환을 gamma=G 필터로 앞당김. '+r': 온도·전압 규칙 범위 밖 단계를 제외.
    T는 행렬 하나이거나 {experiment_id: 행렬} (교차검증용).
    """
    if mode == 'raw':
        return prob.argmax(axis=1) + 1
    base, fast, use_rules = _parse_mode(mode)
    out = np.empty(len(meta), dtype=int)
    t = meta['relative_time_s'].to_numpy(dtype=float)
    rule_inputs = [meta[c].to_numpy(dtype=float) for c in RULE_COLS] if use_rules else None
    for exp, idx in meta.groupby('experiment_id', sort=False).indices.items():
        matrix = T[exp] if isinstance(T, dict) else T
        lo, hi = stage_bounds(*(x[idx] for x in rule_inputs)) if use_rules else (None, None)
        lab = forward_filter(prob[idx], t[idx], matrix, gamma, lo, hi).argmax(axis=1) + 1
        if fast is not None:
            quick = forward_filter(prob[idx], t[idx], matrix, fast, lo, hi).argmax(axis=1) + 1
            lab = np.maximum(lab, np.where(quick >= FAST_FROM, quick, 0))
        if base.startswith('filter_latch'):
            k = int(base[len('filter_latch'):] or 2)
            lab = np.maximum(lab, np.maximum.accumulate(np.where(lab >= k, lab, 0)))
        elif base.startswith('filter_hold'):
            lab = np.maximum(lab, _confirmed(lab, t[idx], float(base[len('filter_hold'):])))
        out[idx] = lab
    return out
