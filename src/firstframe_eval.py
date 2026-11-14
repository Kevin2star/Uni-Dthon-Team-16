"""'처음 프레임 기준' 이미지 표현 실험.

기준 프레임 = 각 experiment_id에서 relative_time_s가 가장 작은 프레임 1장 (모든 행보다 과거 → 미래 정보 없음)
R0 ConvNeXt 임베딩 그대로              (기준)
R1 ConvNeXt 임베딩 차이                 E(지금) − E(처음)
R2 직접 만든 피처 차이                  [퍼짐 13 + 분포 64](지금) − (처음)
R3 픽셀 차이 이미지의 ConvNeXt 임베딩   (온도지수 지금 − 처음) 이미지 → ConvNeXt

평가: ① 분산 진단(실험/단계)  ② 이미지만 GroupKFold(5)+Validation  ③ 센서 + 이미지 확률 6개 (중첩 CV 스태킹)
사용법: python -u firstframe_eval.py | tee ../outputs/firstframe.log     (약 20~30분)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from paths import DATA_ROOT  # 데이터 위치: 환경변수 DATA_DIR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                 # ~/yeonwoo
P_CACHE = ROOT / 'cache'           # 이미지 피처·임베딩 캐시
P_OUT = ROOT / 'outputs'           # 그림·로그·예측 결과
sys.path[:0] = [str(HERE), str(ROOT / 'uni_d_baseline_v7')]
import thermal as T  # noqa: E402
from image_repr import build_backbone  # noqa: E402
from lgbm_common import build_lgbm  # noqa: E402
from stack_temporal import load as load_sensor  # noqa: E402

DATA = DATA_ROOT
CACHE = P_CACHE / 'work_imgrep'
SPLITS = ['train', 'validation', 'test']
LABELS = [1, 2, 3, 4, 5, 6]
SEED = 42
PCA_DIM = 64


def macro(y, p):
    return f1_score(y, p, labels=LABELS, average='macro', zero_division=0)


def groupk(groups):
    return list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))


def first_frame_index(frame):
    """행마다 같은 실험의 가장 이른 프레임 위치."""
    t = frame.relative_time_s.to_numpy(float)
    first = frame.assign(_t=t).groupby('experiment_id')['_t'].transform('idxmin').to_numpy()
    return first


# ---------------------------------------------------------------- 픽셀 차이 임베딩 (GPU)
def diff_embeddings(split, frame, first):
    f = CACHE / f'cnn_convnext_tiny_diff_{split}.npy'
    if f.is_file():
        return np.load(f)
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_backbone('convnext_tiny').eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    paths = [str(DATA / split / p) for p in frame.image_path]
    refs = [paths[i] for i in first]

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i): return torch.from_numpy(T.to_model_input_diff(paths[i], refs[i]))

    out = []
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(DS(), batch_size=256, num_workers=12):
            x = x.to(dev).repeat(1, 3, 1, 1)
            out.append(net((x - mean) / std).float().cpu().numpy())
    np.save(f, np.concatenate(out).astype(np.float32))
    return np.load(f)


# ---------------------------------------------------------------- 진단
def variance_shares(X, exp, y):
    Xs = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-8))
    tot = (Xs ** 2).sum()
    gm = lambda Z, g: pd.DataFrame(Z).groupby(g).transform('mean').values
    within = Xs - gm(Xs, exp)
    return ((gm(Xs, exp) ** 2).sum() / tot, (gm(Xs, y) ** 2).sum() / tot,
            (gm(within, y) ** 2).sum() / max((within ** 2).sum(), 1e-12))


# ---------------------------------------------------------------- 이미지 모델
def img_fit_predict(A, ya, B):
    if A.shape[1] > 100:
        p = PCA(n_components=PCA_DIM, random_state=SEED).fit(A)
        A, B = p.transform(A), p.transform(B)
    return build_lgbm().fit(A, ya).predict_proba(B)


def main():
    t0 = time.time()
    fr = {s: pd.read_csv(DATA / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'}) for s in SPLITS}
    first = {s: first_frame_index(fr[s]) for s in SPLITS}
    reps = {k: {} for k in ['R0 임베딩 그대로', 'R1 임베딩 차이', 'R2 직접 피처 차이', 'R3 픽셀 차이 → ConvNeXt']}
    for s in SPLITS:
        E = np.load(CACHE / f'cnn_convnext_tiny_{s}.npy')
        spread = pd.read_csv(P_CACHE / 'work_model_image' / f'img_{s}.csv', dtype={'sample_id': 'string'}) \
            .set_index('sample_id').loc[fr[s].sample_id].values
        H = np.hstack([spread, np.load(CACHE / f'rich_{s}.npy')])
        reps['R0 임베딩 그대로'][s] = E
        reps['R1 임베딩 차이'][s] = E - E[first[s]]
        reps['R2 직접 피처 차이'][s] = H - H[first[s]]
        reps['R3 픽셀 차이 → ConvNeXt'][s] = diff_embeddings(s, fr[s], first[s])
        print(f'{s} 준비 완료 ({time.time() - t0:.0f}s)', flush=True)

    ytr, yva = fr['train'].label.values, fr['validation'].label.values
    g = fr['train'].experiment_id.values
    folds = groupk(g)

    print('\n=== ① 분산 진단 (Train): 실험 차이↓, 단계↑ 일수록 좋음')
    print(f"{'표현':<24}{'실험 차이':>10}{'단계 차이':>10}{'실험 안 단계':>14}")
    for n, R in reps.items():
        a, b, c = variance_shares(R['train'], g, ytr)
        print(f'{n:<24}{a * 100:>9.0f}%{b * 100:>9.0f}%{c * 100:>13.0f}%', flush=True)

    print('\n=== ② 이미지만 (한 프레임 + 처음 프레임 기준)')
    oof_all = {}
    for n, R in reps.items():
        oof = np.zeros((len(ytr), 6))
        for a, b in folds:
            oof[b] = img_fit_predict(R['train'][a], ytr[a], R['train'][b])
        pva = img_fit_predict(R['train'], ytr, R['validation'])
        oof_all[n] = (oof, pva)
        sc = [macro(ytr[b], oof[b].argmax(1) + 1) for _, b in folds]
        per = f1_score(yva, pva.argmax(1) + 1, labels=LABELS, average=None, zero_division=0)
        print(f'{n:<24} | CV {np.mean(sc):.4f} ± {np.std(sc):.4f} | Val {macro(yva, pva.argmax(1) + 1):.4f} '
              f'| Val S1~6 {np.round(per, 2).tolist()} ({time.time() - t0:.0f}s)', flush=True)

    print('\n=== ③ 센서 + 이미지 확률 6개 (중첩 CV: 평가 실험은 이미지·최종 모델 학습 모두에서 제외)')
    _, Str, _, _ = load_sensor('train')
    _, Sva, _, _ = load_sensor('validation')

    def line(name, sc, pv):
        per = f1_score(yva, pv, labels=LABELS, average=None, zero_division=0)
        print(f'{name:<30} | CV {np.mean(sc):.4f} ± {np.std(sc):.4f} | fold {" ".join(f"{x:.3f}" for x in sc)} '
              f'| Val {macro(yva, pv):.4f} | Val S1~6 {np.round(per, 2).tolist()}', flush=True)

    sc = [macro(ytr[b], build_lgbm().fit(Str[a], ytr[a]).predict(Str[b])) for a, b in folds]
    line('센서만', sc, build_lgbm().fit(Str, ytr).predict(Sva))
    for n, R in reps.items():
        X = R['train']
        sc = []
        for a, b in folds:
            inner = np.zeros((len(a), 6))
            for ia, ib in groupk(g[a]):
                inner[ib] = img_fit_predict(X[a[ia]], ytr[a[ia]], X[a[ib]])
            ev = img_fit_predict(X[a], ytr[a], X[b])
            m = build_lgbm().fit(np.hstack([Str[a], inner]), ytr[a])
            sc.append(macro(ytr[b], m.predict(np.hstack([Str[b], ev]))))
        oof, pva = oof_all[n]   # Train 안 OOF / Train 전체로 학습한 이미지 모델의 Validation 확률
        pv = build_lgbm().fit(np.hstack([Str, oof]), ytr).predict(np.hstack([Sva, pva]))
        line(f'센서 + {n} 확률', sc, pv)
    print(f'총 {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
