"""열화상 PNG → 0~255 '상대 온도 지수' 복원 + 열 퍼짐 피처.

이미지는 8비트 값에 256색 컬러맵을 입힌 것입니다(전체 데이터에서 색이 정확히 256개).
thermal_palette.npy 는 그 컬러맵을 차가움(0, 검정) → 뜨거움(255, 흰색) 순서로 복원한 것입니다.
이미지마다 최솟값~최댓값이 거의 0~255로 늘어나 있으니(자동 범위), 값은 절대 온도가 아니라 이미지 안에서의 상대 온도입니다.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

_PAL = np.load(Path(__file__).with_name('thermal_palette.npy')).astype(np.int64)
_LUT = np.zeros(256 ** 3, np.uint8)
_LUT[(_PAL[:, 0] << 16) | (_PAL[:, 1] << 8) | _PAL[:, 2]] = np.arange(256, dtype=np.uint8)

SPREAD_FEATURES = ['th_mean', 'th_std', 'th_p50', 'th_p90', 'th_p99',
                   'hot200_ratio', 'hot230_ratio', 'hot200_n_blobs', 'hot200_largest',
                   'hot200_cy', 'hot200_cx', 'hot200_spread', 'th_entropy']


def decode(path) -> np.ndarray:
    """PNG 경로 → (H, W) uint8 온도 지수."""
    x = np.asarray(Image.open(path).convert('RGB')).astype(np.int64)
    return _LUT[(x[..., 0] << 16) | (x[..., 1] << 8) | x[..., 2]]


def spread_features(path) -> list[float]:
    t = decode(path).astype(np.float32)
    h, w = t.shape
    hot = t > 200
    lab, n = ndimage.label(hot)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    if hot.any():
        ys, xs = np.nonzero(hot)
        cy, cx = ys.mean() / h, xs.mean() / w          # 고온 영역 중심 (0~1, 이미지 크기와 무관)
        spread = np.sqrt(ys.var() / h ** 2 + xs.var() / w ** 2)  # 고온 영역이 얼마나 흩어져 있나
    else:
        cy = cx = spread = np.nan
    hist = np.bincount(t.astype(np.uint8).ravel(), minlength=256) / t.size
    entropy = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
    return [float(t.mean()), float(t.std()), *np.percentile(t, [50, 90, 99]).tolist(),
            float(hot.mean()), float((t > 230).mean()), float(n), float(sizes.max() / t.size),
            float(cy), float(cx), float(spread), entropy]


def to_model_input(path, size=224) -> np.ndarray:
    """CNN 입력용: 비율 유지 letterbox(빈 곳은 0=가장 차가움)로 size×size, (1, size, size) float 0~1."""
    t = decode(path)
    h, w = t.shape
    s = size / max(h, w)
    im = Image.fromarray(t).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - im.height) // 2, (size - im.width) // 2
    canvas[y0:y0 + im.height, x0:x0 + im.width] = np.asarray(im)
    return canvas[None].astype(np.float32) / 255.0


def to_model_input_diff(path, ref_path, size=224) -> np.ndarray:
    """CNN 입력용 '처음 프레임 대비 변화' 이미지: (지금 − 기준) 온도 지수를 0~255로 (128 = 변화 없음), letterbox.
    기준 이미지 크기가 다르면 지금 이미지 크기로 맞춥니다."""
    cur = decode(path).astype(np.int16)
    ref = decode(ref_path)
    if ref.shape != cur.shape:
        ref = np.asarray(Image.fromarray(ref).resize((cur.shape[1], cur.shape[0]), Image.BILINEAR))
    d = np.clip((cur - ref.astype(np.int16) + 256) // 2, 0, 255).astype(np.uint8)   # 변화 0 → 128
    h, w = d.shape
    s = size / max(h, w)
    im = Image.fromarray(d).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    canvas = np.full((size, size), 128, np.uint8)
    y0, x0 = (size - im.height) // 2, (size - im.width) // 2
    canvas[y0:y0 + im.height, x0:x0 + im.width] = np.asarray(im)
    return canvas[None].astype(np.float32) / 255.0


GRID = 4
THRESHOLDS = [150, 180, 200, 230]
RICH_FEATURES = ([f'hist{i:02d}' for i in range(16)]
                 + [f'g{r}{c}_mean' for r in range(GRID) for c in range(GRID)]
                 + [f'g{r}{c}_hot' for r in range(GRID) for c in range(GRID)]
                 + [f'area{t}' for t in THRESHOLDS] + [f'blobs{t}' for t in THRESHOLDS]
                 + ['hot_bbox_h', 'hot_bbox_w', 'hot_fill', 'hot_ecc', 'hot_perim_ratio',
                    'center_minus_edge', 'top_minus_bottom', 'left_minus_right'])


def rich_features(path) -> list[float]:
    """분포를 더 자세히: 히스토그램 16칸 + 4×4 격자 평균·고온비율 + 기준값별 면적·덩어리 수 + 고온 영역 모양."""
    t = decode(path).astype(np.float32)
    h, w = t.shape
    hist = np.bincount((t // 16).astype(int).ravel(), minlength=16)[:16] / t.size
    ys = np.linspace(0, h, GRID + 1).astype(int); xs = np.linspace(0, w, GRID + 1).astype(int)
    cells = [t[ys[r]:max(ys[r + 1], ys[r] + 1), xs[c]:max(xs[c + 1], xs[c] + 1)] for r in range(GRID) for c in range(GRID)]
    g_mean = [float(c.mean()) for c in cells]
    g_hot = [float((c > 200).mean()) for c in cells]
    area = [float((t > th).mean()) for th in THRESHOLDS]
    blobs = [float(ndimage.label(t > th)[1]) for th in THRESHOLDS]
    hot = t > 200
    if hot.sum() >= 3:
        yy, xx = np.nonzero(hot)
        bh, bw = (yy.max() - yy.min() + 1) / h, (xx.max() - xx.min() + 1) / w
        fill = hot.sum() / ((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1))
        ev = np.linalg.eigvalsh(np.cov(np.vstack([yy / h, xx / w])) + 1e-9 * np.eye(2))
        ecc = float(np.sqrt(1 - ev[0] / ev[1]))
        perim = (hot ^ ndimage.binary_erosion(hot)).sum() / np.sqrt(hot.sum())
    else:
        bh = bw = fill = ecc = perim = np.nan
    ch, cw = h // 4, w // 4
    center = t[ch:h - ch, cw:w - cw].mean() if h > 4 and w > 4 else t.mean()
    edge = (t.sum() - t[ch:h - ch, cw:w - cw].sum()) / max(1, t.size - t[ch:h - ch, cw:w - cw].size)
    return [*hist.tolist(), *g_mean, *g_hot, *area, *blobs, float(bh), float(bw), float(fill), float(ecc), float(perim),
            float(center - edge), float(t[:h // 2].mean() - t[h // 2:].mean()), float(t[:, :w // 2].mean() - t[:, w // 2:].mean())]


# ---------------------------------------------------------------------------
# 단계별 이미지 관찰에서 나온 피처 (stage_eda.py 참고). 모두 한 프레임 안에서만 계산.
# ---------------------------------------------------------------------------
STAGE_FEATURES = ['noise_all', 'noise_cool', 'noise_ratio',                      # 1 노이즈(실제 온도 범위가 좁을수록 큼)
                  'hole_frac', 'hole_n', 'hot_inner_std',                        # 2 고온 영역 내부 얼룩
                  'dark20', 'dark40',                                            # 3 배경 어두움
                  'warm_cy', 'hot_bottom_frac',                                  # 4 위아래 분포
                  'warm_largest_frac', 'warm_n']                                 # 5 형태 유지·사라짐


def stage_features(path) -> list[float]:
    t = decode(path).astype(np.float32)
    h, w = t.shape
    nan = float('nan')
    # 1) 노이즈: 3×3 중앙값과의 차이
    resid = np.abs(t - ndimage.median_filter(t, size=3))
    cool = t <= np.percentile(t, 30)
    noise_all = float(resid.mean())
    noise_cool = float(resid[cool].mean()) if cool.any() else nan
    noise_ratio = noise_all / (float(t.std()) + 1.0)
    # 2) 고온 영역 내부 얼룩
    hot = t > 200
    if hot.sum() >= 5:
        filled = ndimage.binary_fill_holes(hot)
        holes = filled & ~hot
        hole_frac = float(holes.sum() / filled.sum())
        hole_n = float(ndimage.label(holes)[1])
        hot_inner_std = float(t[filled].std())
    else:
        hole_frac = hole_n = hot_inner_std = nan
    # 3) 배경 어두움
    dark20, dark40 = float((t < 20).mean()), float((t < 40).mean())
    # 4) 위아래 분포
    warm = t > 150
    warm_cy = float(np.nonzero(warm)[0].mean() / h) if warm.any() else nan
    hot_bottom_frac = float(hot[h // 2:].sum() / hot.sum()) if hot.any() else nan
    # 5) 형태 유지·사라짐
    lab, n = ndimage.label(warm)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        warm_largest_frac, warm_n = float(sizes.max() / sizes.sum()), float(n)
    else:
        warm_largest_frac, warm_n = nan, 0.0
    return [noise_all, noise_cool, noise_ratio, hole_frac, hole_n, hot_inner_std, dark20, dark40,
            warm_cy, hot_bottom_frac, warm_largest_frac, warm_n]
