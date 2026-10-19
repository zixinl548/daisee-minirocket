import os, subprocess
import numpy as np, pandas as pd
from PIL import Image
import mediapipe as mp

ROOT = "/work/users/z/i/zixinl/Emotion"
OUT  = f"{ROOT}/align_check/size"
os.makedirs(OUT, exist_ok=True)
task  = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
ntask = int(os.environ.get("NTASKS", "1"))

vids = [l.strip() for l in open(f"{ROOT}/au_out/videos.txt") if l.strip()]
mine = vids[task::ntask]
print(f"task {task}/{ntask}: {len(mine)} videos", flush=True)

fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                     refine_landmarks=True,
                                     min_detection_confidence=0.3)
rows = []
for i, v in enumerate(mine):
    cid = os.path.basename(v)[:-4]
    png = f"{OUT}/t{task}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "5", "-i", v,
                    "-frames:v", "1", png], check=False)
    if not os.path.exists(png):
        rows.append(dict(clip=cid, w=np.nan, h=np.nan)); continue
    im = np.array(Image.open(png).convert('RGB'))
    H, W = im.shape[:2]
    r = fm.process(im)
    if r.multi_face_landmarks:
        lm = r.multi_face_landmarks[0].landmark
        xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm])
        rows.append(dict(clip=cid, w=(xs.max()-xs.min())*W,
                                   h=(ys.max()-ys.min())*H))
    else:
        rows.append(dict(clip=cid, w=np.nan, h=np.nan))
    os.remove(png)
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(mine)}", flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/size_{task}.csv", index=False)
print("done", flush=True)
