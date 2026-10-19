#!/bin/bash
#SBATCH --job-name=cvscale
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=16:00:00
#SBATCH --output=/work/users/z/i/zixinl/Emotion/au_out/logs/cv_scale.out

PY=$HOME/.conda/envs/ts_env/bin/python
export OMP_NUM_THREADS=8
cd /work/users/z/i/zixinl/Emotion
$PY cv_scale.py
