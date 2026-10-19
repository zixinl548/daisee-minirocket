#!/bin/bash
#SBATCH --job-name=ab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/work/users/z/i/zixinl/Emotion/au_out/logs/ablate.out

PY=$HOME/.conda/envs/ts_env/bin/python
cd /work/users/z/i/zixinl/Emotion
$PY ablate.py
