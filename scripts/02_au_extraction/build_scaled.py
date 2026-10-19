import os
import numpy as np, pandas as pd

ROOT, T = "/work/users/z/i/zixinl/Emotion", 300
SRC = {'orig': f"{ROOT}/au_out/csv",   's1': f"{ROOT}/au_out/csv_s1",
       's2':   f"{ROOT}/au_out/csv_s2", 's4': f"{ROOT}/au_out/csv_s4"}

INT = ['au_1_intensity','au_2_intensity','au_4_intensity','au_5_intensity',
       'au_6_intensity','au_9_intensity','au_12_intensity','au_15_intensity',
       'au_17_intensity','au_20_intensity','au_25_intensity','au_26_intensity']
BIN = ['au_1','au_2','au_4','au_6','au_7','au_10','au_12','au_14','au_15','au_17','au_23','au_24']
CH  = INT + BIN + ['pitch','yaw','roll','gaze_yaw','gaze_pitch']

have = {k: {f[:-4] for f in os.listdir(v) if f.endswith('.csv')} for k, v in SRC.items()}
for k, v in have.items():
    print(f"{k:5s} {len(v)} csv")
common = set.intersection(*have.values())
print(f"四档共有 {len(common)} 个 clip")

lab = []
for fn in ['TrainLabels.csv', 'ValidationLabels.csv', 'TestLabels.csv']:
    d = pd.read_csv(f"{ROOT}/Labels/{fn}")
    d.columns = [c.strip() for c in d.columns]
    d['clip'] = d['ClipID'].astype(str).str.replace('.avi', '', regex=False).str.strip()
    lab.append(d[['clip', 'Confusion']])
lab = pd.concat(lab, ignore_index=True)
lab = lab[lab['clip'].isin(common)].reset_index(drop=True)
print(f"其中有标签的 {len(lab)} 个")

def load(path, clip):
    d = pd.read_csv(f"{path}/{clip}.csv")
    if len(d) == 0 or any(c not in d.columns for c in CH):
        return None
    a = d[CH].to_numpy(dtype=np.float32); n = a.shape[0]
    return (a[:T] if n >= T else np.vstack([a, np.repeat(a[-1:], T - n, 0)])).T

keep, store = [], {k: [] for k in SRC}
for i, r in lab.iterrows():
    arr = {k: load(p, r['clip']) for k, p in SRC.items()}
    if any(v is None for v in arr.values()):
        continue
    for k in SRC:
        store[k].append(arr[k])
    keep.append((r['clip'], int(r['Confusion'])))
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(lab)}", flush=True)

K = pd.DataFrame(keep, columns=['clip', 'conf'])
OUT = f"{ROOT}/au_out/arrays_scale"; os.makedirs(OUT, exist_ok=True)
for k in SRC:
    X = np.stack(store[k]).astype(np.float32)
    np.save(f"{OUT}/X_{k}.npy", X)
    print(f"{k:5s} X {X.shape}")
K.to_csv(f"{OUT}/clips.csv", index=False)
np.save(f"{OUT}/y.npy", K['conf'].to_numpy(np.int64))
open(f"{OUT}/channels.txt", "w").write("\n".join(CH))
print(f"\n最终 {len(K)} clips / {K['clip'].str[:6].nunique()} subjects")
print("标签分布", np.bincount(K['conf'], minlength=4))
