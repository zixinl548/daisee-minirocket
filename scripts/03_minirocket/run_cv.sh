#!/bin/bash
#SBATCH --job-name=cv
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/work/users/z/i/zixinl/Emotion/au_out/logs/cv_subject.out

PY=$HOME/.conda/envs/ts_env/bin/python
export OMP_NUM_THREADS=8
cd /work/users/z/i/zixinl/Emotion
$PY cv_subject.py
