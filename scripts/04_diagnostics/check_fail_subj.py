import os
import pandas as pd

ROOT = "/work/users/z/i/zixinl/Emotion"
paths = {os.path.basename(l.strip())[:-4]: l.strip()
         for l in open(f"{ROOT}/au_out/videos.txt") if l.strip()}
csvs = {f[:-4] for f in os.listdir(f"{ROOT}/au_out/csv") if f.endswith('.csv')}

rows = []
for split, fn in [('Train','TrainLabels.csv'),
                  ('Validation','ValidationLabels.csv'),
                  ('Test','TestLabels.csv')]:
    lab = pd.read_csv(f"{ROOT}/Labels/{fn}")
    lab.columns = [c.strip() for c in lab.columns]
    lab['clip'] = lab['ClipID'].astype(str).str.replace('.avi','',regex=False).str.strip()
    for _, r in lab.iterrows():
        c = r['clip']
        rows.append(dict(split=split, clip=c, subj=c[:6],
                         conf=int(r['Confusion']),
                         on_disk=c in paths, has_csv=c in csvs))
m = pd.DataFrame(rows)

fs = ['510009','510045','510046','928901','500067']
print("=== NOFACE 涉及的 5 个 subject ===")
t = (m[m.subj.isin(fs)].groupby(['split','subj'])
     .agg(labeled=('clip','size'), on_disk=('on_disk','sum'), got_csv=('has_csv','sum')))
t['noface'] = t.on_disk - t.got_csv
t['wipeout'] = t.got_csv == 0
print(t)

print("\n=== 每个 split 实际可用的 subject 数 ===")
print(pd.concat([m.groupby('split').subj.nunique().rename('in_labels'),
                 m[m.has_csv].groupby('split').subj.nunique().rename('with_data')],
                axis=1))

print("\n=== test 每个 subject 的 clip 数和困惑分布 ===")
te = m[(m.split=='Test') & m.has_csv]
print(te.pivot_table(index='subj', columns='conf', values='clip',
                     aggfunc='size', fill_value=0))
