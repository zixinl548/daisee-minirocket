import numpy as np
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

R = "/work/users/z/i/zixinl/Emotion/au_out/arrays"
ALPHAS = np.logspace(-3, 3, 10)

Xs, ys, clips = [], [], []
for s in ['train', 'val', 'test']:
    Xs.append(np.load(f"{R}/X_{s}.npy"))
    ys.append(np.load(f"{R}/y_{s}.npy"))
    clips += open(f"{R}/clips_{s}.txt").read().split()

X = np.nan_to_num(np.concatenate(Xs)).astype(np.float32)
y = np.concatenate(ys)
g = np.array([c[:6] for c in clips])
print(f"X {X.shape}   y {np.bincount(y)}   subjects {len(set(g))}", flush=True)

TASKS = [('4class', lambda v: (np.ones(len(v), bool), v),                      0.25),
         ('3class', lambda v: (v < 3,                 v),                      1/3),
         ('binary', lambda v: (np.ones(len(v), bool), (v > 0).astype(np.int64)), 0.5)]
WEIGHTS = [('bal', 'balanced'), ('raw', None)]

res, cms = {}, {}
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

for k, (tr, te) in enumerate(sgkf.split(X, y, groups=g)):
    mv = MiniRocketMultivariate(num_kernels=10000, random_state=0)
    Ftr = np.asarray(mv.fit_transform(X[tr]), dtype=np.float32)
    Fte = np.asarray(mv.transform(X[te]),     dtype=np.float32)
    print(f"\nfold {k}: train {len(tr)} ({len(set(g[tr]))} subj)  "
          f"test {len(te)} ({len(set(g[te]))} subj)", flush=True)

    for tname, fn, _ in TASKS:
        mask, yy = fn(y)
        labs = np.unique(yy[mask])
        mtr, mte = mask[tr], mask[te]
        t = yy[te][mte]
        maj = np.bincount(t).max() / len(t)
        for wname, w in WEIGHTS:
            clf = RidgeClassifierCV(alphas=ALPHAS, class_weight=w)
            clf.fit(Ftr[mtr], yy[tr][mtr])
            p = clf.predict(Fte[mte])
            key = (tname, wname)
            d = dict(acc=accuracy_score(t, p),
                     bal=balanced_accuracy_score(t, p),
                     f1 =f1_score(t, p, average='macro', zero_division=0),
                     maj=maj)
            res.setdefault(key, []).append(d)
            cms.setdefault(key, []).append(confusion_matrix(t, p, labels=labs))
            print(f"   {tname:7s} {wname:3s}  acc {d['acc']:.4f}  "
                  f"bal {d['bal']:.4f}  macroF1 {d['f1']:.4f}  "
                  f"(majority {maj:.4f})", flush=True)

print("\n\n=== 5-fold subject-level CV, MiniRocketMultivariate + Ridge ===")
print(f"{'task':8s} {'weight':7s} {'accuracy':>17s} {'balanced':>17s} "
      f"{'macroF1':>17s} {'major':>7s} {'chance':>7s}")
for tname, _, ch in TASKS:
    for wname, _ in WEIGHTS:
        v = res[(tname, wname)]
        def ms(k):
            a = np.array([d[k] for d in v])
            return f"{a.mean():.4f}+/-{a.std(ddof=1):.4f}"
        maj = np.mean([d['maj'] for d in v])
        print(f"{tname:8s} {wname:7s} {ms('acc'):>17s} {ms('bal'):>17s} "
              f"{ms('f1'):>17s} {maj:7.4f} {ch:7.4f}")

print("\n=== 每折 balanced accuracy ===")
for key in res:
    a = np.array([d['bal'] for d in res[key]])
    print(f"{key[0]:8s} {key[1]:3s}  {np.round(a, 4)}")

print("\n=== 五折合并混淆矩阵 ===")
for key in cms:
    print(f"\n{key[0]}  class_weight={key[1]}")
    print(sum(cms[key]))
