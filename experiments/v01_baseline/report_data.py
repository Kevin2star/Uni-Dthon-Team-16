"""베이스라인 결과를 report.json으로 추출 (노트북과 같은 설정, work/ 이미지 피처 캐시 사용)."""
import base64, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GroupKFold

BASE = Path('/home/ubuntu/yeonwoo/uni_d_baseline_v7')
sys.path.insert(0, str(BASE))
import model as M

DATA = Path('/home/ubuntu/data/battery_data_light')
OUT = Path(sys.argv[1])
L = M.LABELS


def load(split):
    f = pd.read_csv(DATA / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    img = pd.read_csv(BASE / 'work' / f'img_features_{split}.csv', dtype={'sample_id': 'string'},
                      float_precision='round_trip').set_index('sample_id')
    return f, M.build_features(f, img)


tr, Xtr = load('train'); va, Xva = load('validation'); te, Xte = load('test')
ytr, yva = tr.label.astype(int), va.label.astype(int)
mf1 = lambda a, b: f1_score(a, b, labels=L, average='macro', zero_division=0)

cv = {}
for name, cw in [('none', None), ('balanced', 'balanced')]:
    folds = []
    for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups=tr.experiment_id):
        m = M.build_model(cw).fit(Xtr.iloc[a], ytr.iloc[a])
        folds.append(mf1(ytr.iloc[b], m.predict(Xtr.iloc[b])))
    m = M.build_model(cw).fit(Xtr, ytr)
    cv[name] = {'folds': folds, 'val': mf1(yva, m.predict(Xva))}

model = M.build_model(None).fit(Xtr, ytr)
pv = model.predict(Xva)
proba = model.predict_proba(Xva)
p, r, f, s = precision_recall_fscore_support(yva, pv, labels=L, zero_division=0)
va = va.assign(pred=pv, conf=proba.max(1))

# 실험별 결과 (시간 순 true/pred 시퀀스)
exps = []
for eid, e in va.sort_values(['experiment_id', 'relative_time_s']).groupby('experiment_id'):
    exps.append({'id': eid, 'rows': len(e), 'bat_class': int(e.bat_class.iloc[0]), 'bat_cc': float(e.bat_cc.iloc[0]),
                 'soc': int(e.soc_status.iloc[0]), 'tr': int(e.tr_method.iloc[0]),
                 'f1': mf1(e.label, e.pred), 'acc': accuracy_score(e.label, e.pred),
                 'y': e.label.tolist(), 'p': e.pred.astype(int).tolist(),
                 'backsteps': int((np.diff(e.pred.values) < 0).sum())})


def thumb(split, rel, w=150):
    im = Image.open(DATA / split / rel).convert('RGB')
    im.thumbnail((w, w))
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=72)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


rng = np.random.default_rng(0)
def pick(df, n):
    # 서로 다른 실험에서 최대한 고르게
    out = []
    for eid in rng.permutation(df.experiment_id.unique()):
        out.append(df[df.experiment_id == eid].sample(1, random_state=int(rng.integers(1e6))).iloc[0])
        if len(out) == n: break
    return out

def card(row):
    return {'src': thumb('validation', row.image_path), 'y': int(row.label), 'p': int(row.pred),
            't': int(row.relative_time_s), 'conf': round(float(row.conf), 2), 'exp': row.experiment_id[:8]}

correct = {l: [card(x) for x in pick(va[(va.label == l) & (va.pred == l)], 4)] for l in L}
cm = confusion_matrix(yva, pv, labels=L)
pairs = sorted([(cm[i, j], L[i], L[j]) for i in range(6) for j in range(6) if i != j], reverse=True)[:4]
confusions = [{'y': y, 'p': q, 'n': int(n), 'cards': [card(x) for x in pick(va[(va.label == y) & (va.pred == q)], 6)]}
              for n, y, q in pairs]

# 대표 실험 하나: 단계별 이미지 + 예측
worst = min(exps, key=lambda e: e['f1'] if e['rows'] > 200 else 9)
e = va[va.experiment_id == worst['id']].sort_values('relative_time_s')
strip = []
for l, seg in e.groupby('label'):
    for i in [0, len(seg) // 2, len(seg) - 1]:
        strip.append(card(seg.iloc[i]))

test_pred = model.predict(Xte)
coef = pd.DataFrame(model.named_steps[list(model.named_steps)[-1]].coef_, index=L, columns=Xtr.columns)
top = coef.abs().mean().sort_values(ascending=False).head(12)

rep = {
    'counts': {'train': [len(tr), tr.experiment_id.nunique()], 'validation': [len(va), va.experiment_id.nunique()],
               'test': [len(te), te.experiment_id.nunique()], 'features': Xtr.shape[1]},
    'label_dist': {'train': ytr.value_counts().reindex(L).tolist(), 'validation': yva.value_counts().reindex(L).tolist(),
                   'test_pred': pd.Series(test_pred).value_counts().reindex(L, fill_value=0).tolist()},
    'cv': cv, 'val_acc': accuracy_score(yva, pv), 'val_f1': mf1(yva, pv),
    'per_class': {'precision': p.tolist(), 'recall': r.tolist(), 'f1': f.tolist(), 'support': s.tolist()},
    'cm': cm.tolist(), 'exps': exps, 'correct': correct, 'confusions': confusions,
    'strip': {'id': worst['id'], 'f1': worst['f1'], 'cards': strip},
    'importance': [[k, round(float(v), 3)] for k, v in top.items()],
}
OUT.write_text(json.dumps(rep))
print('saved', OUT, OUT.stat().st_size // 1024, 'KB')
print('val f1', rep['val_f1'], 'pairs', [(c['y'], c['p'], c['n']) for c in confusions], 'worst', worst['id'][:10], worst['f1'])
print('backsteps total', sum(x['backsteps'] for x in exps), 'importance', rep['importance'][:6])
