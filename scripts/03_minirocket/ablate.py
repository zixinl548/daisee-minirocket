import numpy as np
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

R  = "/work/users/z/i/zixinl/Emotion/au_out/arrays"
CH = open(f"{R}/channels.txt").read().split()
Xtr = np.concatenate([np.load(f"{R}/X_train.npy"), np.load(f"{R}/X_val.npy")])
ytr = np.concatenate([np.load(f"{R}/y_train.npy"), np.load(f"{R}/y_val.npy")])
Xte, yte = np.load(f"{R}/X_test.npy"), np.load(f"{R}/y_test.npy")
Xtr = np.nan_to_num(Xtr).astype(np.float32); Xte = np.nan_to_num(Xte).astype(np.float32)

ftr, fte, grp = [], [], []
for c in range(Xtr.shape[1]):
    m = MiniRocket(num_kernels=500, random_state=0)
    ftr.append(np.asarray(m.fit_transform(Xtr[:,c:c+1,:]), dtype=np.float32))
    fte.append(np.asarray(m.transform(Xte[:,c:c+1,:]),    dtype=np.float32))
    grp += [CH[c]] * ftr[-1].shape[1]
Ftr, Fte, grp = np.concatenate(ftr,1), np.concatenate(fte,1), np.array(grp)
print("features", Ftr.shape, flush=True)

POSE = ['pitch','yaw','roll','gaze_yaw','gaze_pitch']
AU   = [c for c in CH if c.startswith('au_')]

def run(name, cols, cw):
    k = np.isin(grp, cols)
    clf = RidgeClassifierCV(alphas=np.logspace(-3,3,10), class_weight=cw).fit(Ftr[:,k], ytr)
    p = clf.predict(Fte[:,k])
    rec = confusion_matrix(yte,p).diagonal() / np.bincount(yte)
    print(f"\n### {name}  (class_weight={cw}, {k.sum()} feats)")
    print(f"acc {accuracy_score(yte,p):.4f}   bal {balanced_accuracy_score(yte,p):.4f}"
          f"   macroF1 {f1_score(yte,p,average='macro'):.4f}")
    print("recall per class:", np.round(rec,3))
    print(confusion_matrix(yte,p))

for cw in [None, 'balanced']:
    run("ALL 29 channels", CH,   cw)
    run("AU only (24)",    AU,   cw)
    run("POSE+GAZE only (5)", POSE, cw)
print(f"\nmajority baseline {np.bincount(yte).max()/len(yte):.4f}")
