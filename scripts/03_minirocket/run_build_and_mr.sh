#!/bin/bash
#SBATCH --job-name=mr
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/work/users/z/i/zixinl/Emotion/au_out/logs/minirocket.out

PY=$HOME/.conda/envs/ts_env/bin/python
cd /work/users/z/i/zixinl/Emotion
echo "=== rebuild arrays ==="
$PY build_au_arrays.py
echo
echo "=== minirocket ==="
$PY run_minirocket.py
