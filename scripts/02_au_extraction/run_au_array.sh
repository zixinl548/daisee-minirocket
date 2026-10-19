#!/bin/bash
#SBATCH --job-name=au
#SBATCH --array=0-99
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/work/users/z/i/zixinl/Emotion/au_out/logs/au_%a.out

module load ffmpeg

PY=$HOME/.conda/envs/libreface_env/bin/python

export NTASKS=100
export OMP_NUM_THREADS=1

cd /work/users/z/i/zixinl/Emotion
echo "PY   = $PY"
echo "FFMPEG = $(which ffmpeg)"
$PY extract_au.py
