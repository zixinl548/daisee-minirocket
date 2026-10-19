import os
import pandas as pd

ROOT = "/work/users/z/i/zixinl/Emotion"

paths = {}
for l in open(f"{ROOT}/au_out/videos.txt"):
    l = l.strip()
    if l:
        paths[os.path.basename(l)[:-4]] = l

csvs = {f[:-4] for f in os.listdir(f"{ROOT}/au_out/csv") if f.endswith('.csv')}

bad = [c for c, p in paths.items() if c[:6] not in p.split(os.sep)]
print(f"clip 前 6 位不是 subject 目录名的: {len(bad)}  (期望 0)")
for c in bad[:3]:
    print("   例:", c, paths[c])

subj_on_disk = {c[:6] for c in paths}
print(f"盘上 subject 目录数: {len(subj_on_disk)}")

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
                         conf=int(r['Confusion']), has_video=c in paths))

m = pd.DataFrame(rows)
miss = m[~m.has_video]
gone = miss[~miss.subj.isin(subj_on_disk)]
part = miss[ miss.subj.isin(subj_on_disk)]

print(f"\n缺视频 {len(miss)} 个，涉及 {miss.subj.nunique()} 个 subject")
print(f"  整个 subject 目录不在盘上: {len(gone)} clips / {gone.subj.nunique()} subjects")
print(f"  subject 在、单个 clip 缺:  {len(part)} clips / {part.subj.nunique()} subjects")

print("\n=== 每个 split 有多少 subject，缺了多少 ===")
print(m.groupby('split').agg(subj_total=('subj','nunique')))
print(gone.groupby('split').agg(subj_gone=('subj','nunique'), clips=('clip','size')))

print("\n=== 整 subject 缺失 vs 零散缺失，按困惑等级 ===")
print(pd.DataFrame({'整subject缺': gone.conf.value_counts(),
                    '零散缺':      part.conf.value_counts()}).fillna(0).astype(int).sort_index())

miss[['split','subj','clip','conf']].sort_values(['split','subj','clip']) \
    .to_csv(f"{ROOT}/au_out/missing_clips.csv", index=False)
print("\n清单 -> au_out/missing_clips.csv")
