import numpy as np, pandas as pd
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

R   = "/work/users/z/i/zixinl/Emotion/au_out/arrays"
OUT = "/work/users/z/i/zixinl/Emotion/align_check"
ALPHAS = np.logspace(-3, 3, 10)
CH = open(f"{R}/channels.txt").read().split()

Xs, ys, clips = [], [], []
for s in ['train', 'val', 'test']:
    Xs.append(np.load(f"{R}/X_{s}.npy"))
    ys.append(np.load(f"{R}/y_{s}.npy"))
    clips += open(f"{R}/clips_{s}.txt").read().split()
X = np.nan_to_num(np.concatenate(Xs)).astype(np.float32)
y = np.concatenate(ys)
clips = np.array(clips)
g = np.array([c[:6] for c in clips])
n = len(clips)
print(f"X {X.shape}  subjects {len(set(g))}", flush=True)

for c in ['gaze_yaw', 'gaze_pitch']:
    i = CH.index(c)
    X[:, i, :] = np.clip(X[:, i, :], -90, 90)

INT = [c for c in CH if c.endswith('_intensity')]
BIN = [c for c in CH if c.startswith('au_') and not c.endswith('_intensity')]
SETS  = [('ALL29', CH), ('AU24', INT + BIN), ('POSE3', ['pitch', 'yaw', 'roll'])]
TASKS = [('binary', (y > 0).astype(np.int64)), ('4class', y)]

sgkf  = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(sgkf.split(X, y, groups=g))

fold = np.full(n, -1, np.int64)
for k, (tr, te) in enumerate(folds):
    fold[te] = k
pred = {f'{s}_{t}': np.full(n, -1, np.int64) for s, _ in SETS for t, _ in TASKS}

for sname, cols in SETS:
    idx = [CH.index(c) for c in cols]
    print(f"\n{sname}  ({len(idx)} channels)", flush=True)
    for k, (tr, te) in enumerate(folds):
        Xtr, Xte = X[tr][:, idx, :], X[te][:, idx, :]
        mu = Xtr.mean(axis=(0, 2), keepdims=True)
        sd = Xtr.std(axis=(0, 2), keepdims=True); sd[sd == 0] = 1.0
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        mv  = MiniRocketMultivariate(num_kernels=10000, random_state=0)
        Ftr = np.asarray(mv.fit_transform(Xtr), dtype=np.float32)
        Fte = np.asarray(mv.transform(Xte),     dtype=np.float32)
        for tname, yy in TASKS:
            clf = RidgeClassifierCV(alphas=ALPHAS, class_weight='balanced')
            clf.fit(Ftr, yy[tr])
            pred[f'{sname}_{tname}'][te] = clf.predict(Fte)
        print(f"  fold{k} done", flush=True)

df = pd.DataFrame({'clip': clips, 'subj': g, 'y4': y,
                   'y2': (y > 0).astype(int), 'fold': fold, **pred})
df.to_csv(f"{OUT}/oof_preds.csv", index=False)
print("\nsaved", df.shape, "->", f"{OUT}/oof_preds.csv", flush=True)
for s, _ in SETS:
    for t, _ in TASKS:
        yy = df['y2'] if t == 'binary' else df['y4']
        print(f"  {s:6s} {t:7s} acc {(df[f'{s}_{t}'] == yy).mean():.4f}")
