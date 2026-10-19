#!/bin/bash
#SBATCH --job-name=fsize
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/work/users/z/i/zixinl/Emotion/au_out/logs/fsize_%a.out

module load ffmpeg
export NTASKS=10
export OMP_NUM_THREADS=1
cd /work/users/z/i/zixinl/Emotion
$HOME/.conda/envs/libreface_env/bin/python face_size.py
