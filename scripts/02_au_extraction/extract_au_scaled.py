import os, shutil, subprocess
import pandas as pd
import libreface

ROOT  = "/work/users/z/i/zixinl/Emotion"
SCALE = int(os.environ["SCALE"])
OUT   = f"{ROOT}/au_out/csv_s{SCALE}"
TMPR  = f"{ROOT}/au_out/tmp_s{SCALE}"
os.makedirs(OUT, exist_ok=True); os.makedirs(TMPR, exist_ok=True)

task  = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
ntask = int(os.environ.get("NTASKS", "1"))
videos = [l.strip() for l in open(f"{ROOT}/au_out/videos.txt") if l.strip()]
mine = videos[task::ntask]
print(f"scale 1/{SCALE}  task {task}/{ntask}: {len(mine)} videos", flush=True)

HEAD = ['frame_idx','pitch','yaw','roll','gaze_yaw','gaze_pitch','facial_expression']
W, H = 640, 480
vf = [] if SCALE == 1 else [
    "-vf", f"scale={W//SCALE}:{H//SCALE}:flags=bilinear,scale={W}:{H}:flags=bilinear"]

for v in mine:
    cid = os.path.basename(v).replace('.avi', '')
    out = f"{OUT}/{cid}.csv"
    if os.path.exists(out):
        continue
    tmp = f"{TMPR}/t{task}_{cid}"
    os.makedirs(tmp, exist_ok=True)
    deg, big = f"{tmp}/deg.avi", f"{tmp}/full.csv"
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v] + vf +
                           ["-c:v", "ffv1", "-an", deg],
                           capture_output=True, text=True)
        if not os.path.exists(deg):
            print(f"FAIL_FFMPEG {cid} {r.stderr[:150]}", flush=True)
        else:
            libreface.get_facial_attributes(deg, output_save_path=big,
                                            device='cpu', temp_dir=f"{tmp}/lf")
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
print("task done", flush=True)
