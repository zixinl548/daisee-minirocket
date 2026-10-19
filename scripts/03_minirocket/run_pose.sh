#!/bin/bash
#SBATCH --job-name=cvpose
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/work/users/z/i/zixinl/Emotion/au_out/logs/cv_pose.out

PY=$HOME/.conda/envs/ts_env/bin/python
export OMP_NUM_THREADS=8
cd /work/users/z/i/zixinl/Emotion
$PY cv_pose.py
