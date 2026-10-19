import sys, os, shutil
import pandas as pd
import libreface

ROOT = "/work/users/z/i/zixinl/Emotion"
OUT  = f"{ROOT}/au_out/csv"
task  = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
ntask = int(os.environ.get("NTASKS", "1"))

videos = [l.strip() for l in open(f"{ROOT}/au_out/videos.txt") if l.strip()]
mine = videos[task::ntask]
print(f"task {task}/{ntask}: {len(mine)} videos", flush=True)

HEAD = ['frame_idx','pitch','yaw','roll','gaze_yaw','gaze_pitch','facial_expression']

for v in mine:
    cid = os.path.basename(v).replace('.avi','')
    out = f"{OUT}/{cid}.csv"
    if os.path.exists(out):
        continue
    tmp = f"{ROOT}/au_out/tmp/t{task}_{cid}"
    big = f"{tmp}/full.csv"
    os.makedirs(tmp, exist_ok=True)
    try:
        libreface.get_facial_attributes(v, output_save_path=big,
                                        device='cpu', temp_dir=tmp)
        if not os.path.exists(big):
            print(f"FAIL_NOFACE {cid}", flush=True)
        else:
            df = pd.read_csv(big)
            cols = [c for c in HEAD if c in df.columns] + \
                   [c for c in df.columns if c.startswith('au_')]
            df[cols].to_csv(out, index=False)
            print(f"OK {cid} {len(df)}", flush=True)
    except Exception as e:
        print(f"FAIL {cid} {type(e).__name__} {e}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
