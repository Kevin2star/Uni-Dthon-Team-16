import importlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
HERE = Path('/home/ubuntu/yeonwoo'); sys.path[:0] = [str(HERE), str(HERE / 'uni_d_baseline_v7')]
DATA = Path('/home/ubuntu/data/battery_data_light')
out = []
for name in ['model', 'model_lgbm', 'model_image', 'model_image_lgbm']:
    M = importlib.import_module(name); cd = HERE / f"work_{getattr(M, 'FEATURE_CACHE', name)}"
    def load(s):
        f = pd.read_csv(DATA / s / f'{s}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        img = pd.read_csv(cd / f'img_{s}.csv', dtype={'sample_id': 'string'}, float_precision='round_trip').set_index('sample_id')
        return f, M.build_features(f, img), f.label.astype(int)
    tr, Xtr, ytr = load('train'); va, Xva, yva = load('validation')
    f1 = lambda a, b: f1_score(a, b, labels=M.LABELS, average='macro', zero_division=0)
    folds = [f1(ytr.iloc[b], M.build_model(None).fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b]))
             for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups=tr.experiment_id)]
    pv = M.build_model(None).fit(Xtr, ytr).predict(Xva)
    va = va.assign(pred=pv)
    exps = {eid: f1(g.label, g.pred) for eid, g in va.groupby('experiment_id')}
    out.append({'name': name, 'feat': int(Xtr.shape[1]), 'folds': folds, 'val': f1(yva, pv),
                'per': f1_score(yva, pv, labels=M.LABELS, average=None, zero_division=0).tolist(),
                'cm': confusion_matrix(yva, pv, labels=M.LABELS).tolist(), 'exps': exps,
                'back': int(sum((np.diff(g.sort_values('relative_time_s').pred.values) < 0).sum() for _, g in va.groupby('experiment_id')))})
    print(name, np.mean(folds).round(4), round(out[-1]['val'], 4), out[-1]['back'])
json.dump(out, open(sys.argv[1], 'w'))
