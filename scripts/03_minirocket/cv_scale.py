import numpy as np, pandas as pd
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

R = "/work/users/z/i/zixinl/Emotion/au_out/arrays_scale"
ALPHAS = np.logspace(-3, 3, 10)
CH = open(f"{R}/channels.txt").read().split()
K  = pd.read_csv(f"{R}/clips.csv", dtype={'clip': str})
y  = np.load(f"{R}/y.npy")
y2 = (y > 0).astype(np.int64)
g  = K['clip'].str[:6].to_numpy()
print(f"{len(y)} clips / {len(set(g))} subjects", flush=True)

INT = [c for c in CH if c.endswith('_intensity')]
BIN = [c for c in CH if c.startswith('au_') and not c.endswith('_intensity')]
SETS   = [('ALL29', CH), ('AU24', INT + BIN), ('POSE3', ['pitch', 'yaw', 'roll'])]
LEVELS = ['orig', 's1', 's2', 's4']

sgkf  = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(sgkf.split(np.zeros(len(y)), y, groups=g))
res = {}

for lv in LEVELS:
    X = np.nan_to_num(np.load(f"{R}/X_{lv}.npy")).astype(np.float32)
    for c in ['gaze_yaw', 'gaze_pitch']:
        i = CH.index(c); X[:, i, :] = np.clip(X[:, i, :], -90, 90)
    print(f"\n===== {lv}   X {X.shape} =====", flush=True)
    for sname, cols in SETS:
        idx, bals = [CH.index(c) for c in cols], []
        for k, (tr, te) in enumerate(folds):
            Xtr, Xte = X[tr][:, idx, :], X[te][:, idx, :]
            mu = Xtr.mean(axis=(0, 2), keepdims=True)
            sd = Xtr.std(axis=(0, 2), keepdims=True); sd[sd == 0] = 1.0
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            mv  = MiniRocketMultivariate(num_kernels=10000, random_state=0)
            Ftr = np.asarray(mv.fit_transform(Xtr), dtype=np.float32)
            Fte = np.asarray(mv.transform(Xte),     dtype=np.float32)
            clf = RidgeClassifierCV(alphas=ALPHAS, class_weight='balanced')
            clf.fit(Ftr, y2[tr])
            bals.append(balanced_accuracy_score(y2[te], clf.predict(Fte)))
        a = np.array(bals); res[(lv, sname)] = a
        print(f"  {sname:6s} bal {a.mean():.4f} +/- {a.std(ddof=1):.4f}   {np.round(a,4)}",
              flush=True)

print("\n\n########## 分辨率阶梯 (binary balanced accuracy) ##########")
print(f"{'level':8s} " + "  ".join(f"{s:>18s}" for s, _ in SETS))
for lv in LEVELS:
    print(f"{lv:8s} " + "  ".join(
        f"{res[(lv,s)].mean():.4f}+/-{res[(lv,s)].std(ddof=1):.4f}" for s, _ in SETS))
