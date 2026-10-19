import numpy as np
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, cohen_kappa_score, confusion_matrix)

R = "/work/users/z/i/zixinl/Emotion/au_out/arrays"
ALPHAS = np.logspace(-3, 3, 10)
CH = open(f"{R}/channels.txt").read().split()

Xs, ys, clips = [], [], []
for s in ['train', 'val', 'test']:
    Xs.append(np.load(f"{R}/X_{s}.npy"))
    ys.append(np.load(f"{R}/y_{s}.npy"))
    clips += open(f"{R}/clips_{s}.txt").read().split()
X = np.nan_to_num(np.concatenate(Xs)).astype(np.float32)
y = np.concatenate(ys)
g = np.array([c[:6] for c in clips])
print(f"X {X.shape}  y {np.bincount(y)}  subjects {len(set(g))}", flush=True)

for c in ['gaze_yaw', 'gaze_pitch']:
    i = CH.index(c)
    n = int((np.abs(X[:, i, :]) > 90).sum())
    X[:, i, :] = np.clip(X[:, i, :], -90, 90)
    print(f"clip {c}: {n} frames -> +/-90", flush=True)

SETS = [('NOAU5',  ['pitch', 'yaw', 'roll', 'gaze_yaw', 'gaze_pitch']),
        ('PITCH1', ['pitch']),
        ('YAW1',   ['yaw']),
        ('ROLL1',  ['roll'])]

TASKS = [('4class', np.ones(len(y), bool), y,                       0.25),
         ('3class', y < 3,                 y,                       1/3),
         ('binary', np.ones(len(y), bool), (y > 0).astype(np.int64), 0.5)]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(sgkf.split(X, y, groups=g))
res, cms = {}, {}

for sname, cols in SETS:
    idx = [CH.index(c) for c in cols]
    print(f"\n{'='*64}\n{sname}  ({len(idx)} channels)\n{'='*64}", flush=True)

    for k, (tr, te) in enumerate(folds):
        Xtr, Xte = X[tr][:, idx, :], X[te][:, idx, :]
        mu = Xtr.mean(axis=(0, 2), keepdims=True)
        sd = Xtr.std(axis=(0, 2), keepdims=True)
        sd[sd == 0] = 1.0
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

        mv = MiniRocketMultivariate(num_kernels=10000, random_state=0)
        Ftr = np.asarray(mv.fit_transform(Xtr), dtype=np.float32)
        Fte = np.asarray(mv.transform(Xte),     dtype=np.float32)

        for tname, mask, yy, _ in TASKS:
            mtr, mte = mask[tr], mask[te]
            t, labs = yy[te][mte], np.unique(yy[mask])
            clf = RidgeClassifierCV(alphas=ALPHAS, class_weight='balanced')
            clf.fit(Ftr[mtr], yy[tr][mtr])
            p = clf.predict(Fte[mte])
            d = dict(acc=accuracy_score(t, p),
                     bal=balanced_accuracy_score(t, p),
                     f1 =f1_score(t, p, average='macro', zero_division=0),
                     qwk=cohen_kappa_score(t, p, weights='quadratic', labels=labs))
            res.setdefault((sname, tname), []).append(d)
            cms.setdefault((sname, tname), []).append(
                confusion_matrix(t, p, labels=labs))
            print(f"  fold{k} {tname:7s}  acc {d['acc']:.4f}  bal {d['bal']:.4f}  "
                  f"F1 {d['f1']:.4f}  QWK {d['qwk']:.4f}", flush=True)

    print(f"\n--- {sname} 汇总 (mean +/- SD over 5 folds) ---", flush=True)
    print(f"{'task':8s} {'accuracy':>17s} {'balanced':>17s} "
          f"{'macroF1':>17s} {'QWK':>17s}", flush=True)
    for tname, _, _, _ in TASKS:
        v = res[(sname, tname)]
        def ms(key):
            a = np.array([x[key] for x in v])
            return f"{a.mean():.4f}+/-{a.std(ddof=1):.4f}"
        print(f"{tname:8s} {ms('acc'):>17s} {ms('bal'):>17s} "
              f"{ms('f1'):>17s} {ms('qwk'):>17s}", flush=True)

print("\n\n############ 总表 balanced accuracy ############")
print(f"{'set':8s} {'nch':>4s} {'4class':>17s} {'3class':>17s} {'binary':>17s}")
for sname, cols in SETS:
    row = [f"{np.mean([x['bal'] for x in res[(sname,t)]]):.4f}"
           f"+/-{np.std([x['bal'] for x in res[(sname,t)]], ddof=1):.4f}"
           for t, _, _, _ in TASKS]
    print(f"{sname:8s} {len(cols):4d} " + " ".join(f"{v:>17s}" for v in row))

print("\n############ 五折合并混淆矩阵 ############")
for key in cms:
    print(f"\n{key[0]}  {key[1]}")
    print(sum(cms[key]))
