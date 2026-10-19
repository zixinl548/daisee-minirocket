import os
import numpy as np, pandas as pd

ROOT = "/work/users/z/i/zixinl/Emotion"
CSV, OUT, T = f"{ROOT}/au_out/csv", f"{ROOT}/au_out/arrays", 300

INT = ['au_1_intensity','au_2_intensity','au_4_intensity','au_5_intensity',
       'au_6_intensity','au_9_intensity','au_12_intensity','au_15_intensity',
       'au_17_intensity','au_20_intensity','au_25_intensity','au_26_intensity']
BIN = ['au_1','au_2','au_4','au_6','au_7','au_10','au_12','au_14','au_15','au_17','au_23','au_24']
POSE = ['pitch','yaw','roll','gaze_yaw','gaze_pitch']
CH = INT + BIN + POSE

os.makedirs(OUT, exist_ok=True)
rows = []

for split, fn in [('train','TrainLabels.csv'),('val','ValidationLabels.csv'),('test','TestLabels.csv')]:
    lab = pd.read_csv(f"{ROOT}/Labels/{fn}")
    lab.columns = [c.strip() for c in lab.columns]
    lab['clip'] = lab['ClipID'].astype(str).str.replace('.avi','',regex=False).str.strip()

    X, y, keep = [], [], []
    for _, r in lab.iterrows():
        p = f"{CSV}/{r['clip']}.csv"
        rec = dict(split=split, clip=r['clip'], conf=int(r['Confusion']))
        if not os.path.exists(p):
            rows.append({**rec, 'status':'missing', 'nframe':0}); continue
        d = pd.read_csv(p)
        if len(d) == 0 or any(c not in d.columns for c in CH):
            rows.append({**rec, 'status':'bad', 'nframe':len(d)}); continue
        a = d[CH].to_numpy(dtype=np.float32)
        n = a.shape[0]
        a = a[:T] if n >= T else np.vstack([a, np.repeat(a[-1:], T-n, axis=0)])
        X.append(a.T); y.append(int(r['Confusion'])); keep.append(r['clip'])
        rows.append({**rec, 'status':'ok', 'nframe':n})

    X = np.stack(X).astype(np.float32) if X else np.zeros((0,len(CH),T), np.float32)
    y = np.array(y, np.int64)
    np.save(f"{OUT}/X_{split}.npy", X); np.save(f"{OUT}/y_{split}.npy", y)
    open(f"{OUT}/clips_{split}.txt","w").write("\n".join(keep))
    print(f"{split}: X={X.shape}  labels={np.bincount(y, minlength=4)}", flush=True)

open(f"{OUT}/channels.txt","w").write("\n".join(CH))
m = pd.DataFrame(rows); m.to_csv(f"{OUT}/manifest.csv", index=False)

print("\n=== 按困惑等级看抽取成功率 ===")
print(pd.crosstab(m['conf'], m['status']))
print()
print(pd.crosstab(m['conf'], m['status'], normalize='index').round(4))
