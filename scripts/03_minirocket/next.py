import numpy as np
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

R  = "/work/users/z/i/zixinl/Emotion/au_out/arrays"
CH = open(f"{R}/channels.txt").read().split()
Xtr = np.concatenate([np.load(f"{R}/X_train.npy"), np.load(f"{R}/X_val.npy")])
ytr = np.concatenate([np.load(f"{R}/y_train.npy"), np.load(f"{R}/y_val.npy")])
Xte, yte = np.load(f"{R}/X_test.npy"), np.load(f"{R}/y_test.npy")
Xtr = np.nan_to_num(Xtr).astype(np.float32); Xte = np.nan_to_num(Xte).astype(np.float32)

# A: per channel
a_tr, a_te = [], []
for c in range(Xtr.shape[1]):
    m = MiniRocket(num_kernels=500, random_state=0)
    a_tr.append(np.asarray(m.fit_transform(Xtr[:,c:c+1,:]), dtype=np.float32))
    a_te.append(np.asarray(m.transform(Xte[:,c:c+1,:]),    dtype=np.float32))
A_tr, A_te = np.concatenate(a_tr,1), np.concatenate(a_te,1)

# B: multivariate
mv = MiniRocketMultivariate(num_kernels=10000, random_state=0)
B_tr = np.asarray(mv.fit_transform(Xtr), dtype=np.float32)
B_te = np.asarray(mv.transform(Xte),     dtype=np.float32)

C_tr, C_te = np.concatenate([A_tr,B_tr],1), np.concatenate([A_te,B_te],1)
print("A", A_tr.shape, " B", B_tr.shape, " C", C_tr.shape, flush=True)

def ev(tag, Ftr, Fte, y_tr, y_te, k):
    clf = RidgeClassifierCV(alphas=np.logspace(-3,3,10),
                            class_weight='balanced').fit(Ftr, y_tr)
    p = clf.predict(Fte)
    cm = confusion_matrix(y_te, p)
    rec = cm.diagonal() / np.bincount(y_te)
    print(f"\n### {tag}")
    print(f"bal {balanced_accuracy_score(y_te,p):.4f}   macroF1 {f1_score(y_te,p,average='macro'):.4f}"
          f"   chance {1/k:.4f}")
    print("recall:", np.round(rec,3))
    print(cm)

for tag, Ftr, Fte in [("A per-channel", A_tr, A_te),
                      ("B multivariate", B_tr, B_te),
                      ("C = A + B", C_tr, C_te)]:
    ev(f"{tag}  4-class", Ftr, Fte, ytr, yte, 4)
    ev(f"{tag}  3-class (drop VeryHigh)",
       Ftr[:, :][ytr<3], Fte[yte<3], ytr[ytr<3], yte[yte<3], 3)
    ev(f"{tag}  BINARY 0 vs rest", Ftr, Fte,
       (ytr>0).astype(int), (yte>0).astype(int), 2)
