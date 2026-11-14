"""이미지 표현 비교: 어떤 표현이 열 분포(→단계)를 가장 잘 잡나.

A. spread : thermal.spread_features 13개 (지금 쓰는 것)
B. rich   : thermal.rich_features 64개 (히스토그램·4×4 격자·기준값별 면적·고온 영역 모양)
C. cnn    : ImageNet 사전학습 ResNet18 임베딩 512차원 (가중치 고정, 온도 지수 → 3채널, letterbox 224)

1단계: 이미지만으로 LightGBM 6클래스 (Train GroupKFold 5 + Validation)
2단계: 센서 모델(model_image_lgbm 피처)에 B / C(PCA 32) 를 더했을 때

캐시: work_imgrep/{rich,cnn}_{split}.npy  (CSV 행 순서 그대로)
사용법: python image_repr.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'uni_d_baseline_v7')]
import thermal as T  # noqa: E402
from lgbm_common import build_lgbm  # noqa: E402

DATA = Path('/home/ubuntu/data/battery_data_light')
CACHE = HERE / 'work_imgrep'
SPLITS = ['train', 'validation', 'test']
LABELS = [1, 2, 3, 4, 5, 6]
SEED = 42


def frame(split):
    return pd.read_csv(DATA / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})


def paths(split):
    return [str(DATA / split / p) for p in frame(split).image_path]


def rich(split):
    f = CACHE / f'rich_{split}.npy'
    if not f.is_file():
        rows = Parallel(n_jobs=-1, batch_size=64)(delayed(T.rich_features)(p) for p in paths(split))
        np.save(f, np.asarray(rows, np.float32))
    return np.load(f)


ARCHS = ['resnet18', 'convnext_tiny']


def build_backbone(arch):
    """ImageNet 사전학습 백본, 분류층을 떼고 임베딩만 출력 (resnet18 512차원, convnext_tiny 768차원)."""
    import torch
    import torchvision.models as tvm
    if arch == 'resnet18':
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
    elif arch == 'convnext_tiny':
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        net.classifier[2] = torch.nn.Identity()   # LayerNorm2d → Flatten → (Linear 제거)
    else:
        raise ValueError(arch)
    return net


def cnn(split, arch='resnet18'):
    f = CACHE / (f'cnn_{split}.npy' if arch == 'resnet18' else f'cnn_{arch}_{split}.npy')
    if f.is_file():
        return np.load(f)
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_backbone(arch)
    net.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    class DS(torch.utils.data.Dataset):
        def __init__(self, ps): self.ps = ps
        def __len__(self): return len(self.ps)
        def __getitem__(self, i): return torch.from_numpy(T.to_model_input(self.ps[i]))

    dl = torch.utils.data.DataLoader(DS(paths(split)), batch_size=256, num_workers=12, shuffle=False)
    out = []
    with torch.no_grad():
        for x in dl:
            x = x.to(dev).repeat(1, 3, 1, 1)
            out.append(net((x - mean) / std).float().cpu().numpy())
    np.save(f, np.concatenate(out).astype(np.float32))
    return np.load(f)


def macro(y, p):
    return f1_score(y, p, labels=LABELS, average='macro', zero_division=0)


def evaluate(name, Xtr, ytr, groups, Xva, yva, pca=None):
    folds = GroupKFold(n_splits=5).split(Xtr, ytr, groups=groups)
    scores = []
    for a, b in folds:
        A, B = Xtr[a], Xtr[b]
        if pca:  # PCA는 학습 fold로만 fit (임베딩 열에만)
            A, B = pca_apply(A, B, pca)
        scores.append(macro(ytr[b], build_lgbm().fit(A, ytr[a]).predict(B)))
    A, B = (pca_apply(Xtr, Xva, pca) if pca else (Xtr, Xva))
    pv = build_lgbm().fit(A, ytr).predict(B)
    per = f1_score(yva, pv, labels=LABELS, average=None, zero_division=0)
    print(f'{name:<34} | 피처 {Xtr.shape[1] if not pca else pca[0] + pca[1]:>4} | CV {np.mean(scores):.4f} ± {np.std(scores):.4f} '
          f'| Val {macro(yva, pv):.4f} | Val S1~6 {np.round(per, 2).tolist()}', flush=True)


def pca_apply(A, B, spec):
    """spec = (앞쪽 그대로 둘 열 수, PCA 차원). 뒤쪽 열(임베딩)만 PCA."""
    keep, k = spec
    p = PCA(n_components=k, random_state=SEED).fit(A[:, keep:])
    return (np.hstack([A[:, :keep], p.transform(A[:, keep:])]), np.hstack([B[:, :keep], p.transform(B[:, keep:])]))


def main():
    CACHE.mkdir(exist_ok=True)
    fr = {s: frame(s) for s in SPLITS}
    t0 = time.time()
    R = {s: rich(s) for s in SPLITS}
    print(f'rich 피처 준비 {time.time() - t0:.0f}s', flush=True)
    t0 = time.time()
    C = {s: cnn(s) for s in SPLITS}
    print(f'CNN 임베딩 준비 {time.time() - t0:.0f}s  shape {C["train"].shape}', flush=True)
    S = {s: pd.read_csv(HERE / 'work_model_image' / f'img_{s}.csv', dtype={'sample_id': 'string'})
         .set_index('sample_id').loc[fr[s].sample_id].values for s in ['train', 'validation']}

    ytr, yva = fr['train'].label.values, fr['validation'].label.values
    g = fr['train'].experiment_id.values
    print('\n=== 1단계: 이미지만으로 (시간 정보 없음, 한 프레임씩)')
    evaluate('A. spread 13', S['train'], ytr, g, S['validation'], yva)
    evaluate('B. rich 64', R['train'], ytr, g, R['validation'], yva)
    evaluate('A+B', np.hstack([S['train'], R['train']]), ytr, g, np.hstack([S['validation'], R['validation']]), yva)
    evaluate('C. CNN ResNet18 512', C['train'], ytr, g, C['validation'], yva)
    evaluate('C. CNN → PCA 32', C['train'], ytr, g, C['validation'], yva, pca=(0, 32))

    print('\n=== 2단계: 센서 모델(model_image_lgbm 피처 43개)에 더하기')
    import model_image_lgbm as M
    X = {}
    for s in ['train', 'validation']:
        img = pd.read_csv(HERE / 'work_model_image' / f'img_{s}.csv', dtype={'sample_id': 'string'},
                          float_precision='round_trip').set_index('sample_id')
        X[s] = M.build_features(fr[s], img).values
    evaluate('센서+열퍼짐 (현재 최고)', X['train'], ytr, g, X['validation'], yva)
    evaluate('+ B. rich 64', np.hstack([X['train'], R['train']]), ytr, g, np.hstack([X['validation'], R['validation']]), yva)
    k = X['train'].shape[1]
    evaluate('+ C. CNN PCA 32', np.hstack([X['train'], C['train']]), ytr, g, np.hstack([X['validation'], C['validation']]),
             yva, pca=(k, 32))


def compare_cnn(archs):
    """CNN 백본끼리만 비교: 이미지만 / 센서 모델 + PCA 32."""
    import model_image_lgbm as M
    CACHE.mkdir(exist_ok=True)
    fr = {s: frame(s) for s in SPLITS}
    ytr, yva = fr['train'].label.values, fr['validation'].label.values
    g = fr['train'].experiment_id.values
    C = {}
    for arch in archs:
        t0 = time.time()
        C[arch] = {s: cnn(s, arch) for s in SPLITS}
        print(f'{arch} 임베딩 준비 {time.time() - t0:.0f}s  shape {C[arch]["train"].shape}', flush=True)
    X = {}
    for s in ['train', 'validation']:
        img = pd.read_csv(HERE / 'work_model_image' / f'img_{s}.csv', dtype={'sample_id': 'string'},
                          float_precision='round_trip').set_index('sample_id')
        X[s] = M.build_features(fr[s], img).values
    k = X['train'].shape[1]
    print('\n=== 이미지만으로 (한 프레임씩)')
    for arch in archs:
        evaluate(f'{arch} {C[arch]["train"].shape[1]}', C[arch]['train'], ytr, g, C[arch]['validation'], yva)
        evaluate(f'{arch} → PCA 32', C[arch]['train'], ytr, g, C[arch]['validation'], yva, pca=(0, 32))
    print('\n=== 센서 모델(model_image_lgbm 피처 43개)에 더하기')
    evaluate('센서+열퍼짐 (현재 최고)', X['train'], ytr, g, X['validation'], yva)
    for arch in archs:
        evaluate(f'+ {arch} PCA 32', np.hstack([X['train'], C[arch]['train']]), ytr, g,
                 np.hstack([X['validation'], C[arch]['validation']]), yva, pca=(k, 32))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', nargs='+', choices=ARCHS,
                    help='CNN 백본끼리만 비교 (예: --arch resnet18 convnext_tiny). 생략하면 A/B/C 전체 비교')
    a = ap.parse_args()
    compare_cnn(a.arch) if a.arch else main()
