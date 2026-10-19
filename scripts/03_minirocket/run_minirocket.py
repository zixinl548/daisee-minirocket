import numpy as np, time
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, classification_report)

R  = "/work/users/z/i/zixinl/Emotion/au_out/arrays"
CH = open(f"{R}/channels.txt").read().split()

Xtr = np.concatenate([np.load(f"{R}/X_train.npy"), np.load(f"{R}/X_val.npy")])
ytr = np.concatenate([np.load(f"{R}/y_train.npy"), np.load(f"{R}/y_val.npy")])
Xte, yte = np.load(f"{R}/X_test.npy"), np.load(f"{R}/y_test.npy")
Xtr = np.nan_to_num(Xtr).astype(np.float32)
Xte = np.nan_to_num(Xte).astype(np.float32)
print("train", Xtr.shape, " test", Xte.shape, flush=True)

t0, ftr, fte, grp = time.time(), [], [], []
for c in range(Xtr.shape[1]):
    m = MiniRocket(num_kernels=500, random_state=0)
    a = np.asarray(m.fit_transform(Xtr[:, c:c+1, :]), dtype=np.float32)
    b = np.asarray(m.transform(Xte[:, c:c+1, :]),    dtype=np.float32)
    ftr.append(a); fte.append(b); grp += [CH[c]] * a.shape[1]
    print(f"  {CH[c]:22s} {a.shape[1]:5d} feats  {time.time()-t0:6.1f}s", flush=True)

Ftr, Fte = np.concatenate(ftr,1), np.concatenate(fte,1)
print("features", Ftr.shape, f"  {time.time()-t0:.0f}s", flush=True)

clf = RidgeClassifierCV(alphas=np.logspace(-3,3,10), class_weight="balanced").fit(Ftr, ytr)
p = clf.predict(Fte)

print("\n=== MiniRocket (per-channel) + Ridge ===")
print(f"Accuracy          {accuracy_score(yte,p):.4f}")
print(f"Balanced accuracy {balanced_accuracy_score(yte,p):.4f}")
print(f"Macro F1          {f1_score(yte,p,average='macro'):.4f}")
print(f"Majority baseline {np.bincount(yte).max()/len(yte):.4f}")
print("\nConfusion matrix (rows=true, cols=pred)")
print(confusion_matrix(yte,p))
print()
print(classification_report(yte,p,digits=3,zero_division=0))

g, w = np.array(grp), np.abs(clf.coef_).mean(0)
print("=== channel importance ===")
for v,c in sorted(((w[g==c].sum(), c) for c in CH), reverse=True):
    print(f"{c:22s} {v:10.2f}")
