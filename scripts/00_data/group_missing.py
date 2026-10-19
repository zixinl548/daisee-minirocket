import pandas as pd, os
from collections import Counter

df = pd.read_csv('train_ready.csv')
df.columns = df.columns.str.strip()

missing = [r['video_path'] for _, r in df.iterrows() if not os.path.exists(r['video_path'])]
# 路径格式 DataSet/Train/<subject>/<clip>/<clip>.avi
subjects = Counter(p.split('/')[2] for p in missing)

print(f"Missing clips: {len(missing)}")
print(f"Affected subject folders: {len(subjects)}\n")
for s, n in sorted(subjects.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n}")
