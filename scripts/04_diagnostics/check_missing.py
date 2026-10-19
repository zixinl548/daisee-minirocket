import os
import pandas as pd

ROOT = "/work/users/z/i/zixinl/Emotion"
vids = {os.path.basename(l.strip())[:-4]
        for l in open(f"{ROOT}/au_out/videos.txt") if l.strip()}
csvs = {f[:-4] for f in os.listdir(f"{ROOT}/au_out/csv") if f.endswith('.csv')}

rows = []
for split, fn in [('train','TrainLabels.csv'),
                  ('val','ValidationLabels.csv'),
                  ('test','TestLabels.csv')]:
    lab = pd.read_csv(f"{ROOT}/Labels/{fn}")
    lab.columns = [c.strip() for c in lab.columns]
    lab['clip'] = lab['ClipID'].astype(str).str.replace('.avi','',regex=False).str.strip()
    for _, r in lab.iterrows():
        c = r['clip']
        if   c in csvs: s = '3_ok'
        elif c in vids: s = '2_extract_fail'
        else:           s = '1_no_video'
        rows.append(dict(split=split, conf=int(r['Confusion']), state=s))

m = pd.DataFrame(rows)
print("=== 计数 ===")
print(pd.crosstab(m['conf'], m['state']))
print("\n=== 按行归一 ===")
print(pd.crosstab(m['conf'], m['state'], normalize='index').round(4))
print("\n=== 分 split ===")
print(pd.crosstab([m['split'], m['conf']], m['state']))
