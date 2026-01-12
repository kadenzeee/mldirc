#!/bin/bash
#SBATCH --job-name=film_s_scan
#SBATCH --output=logs/film_s_%A_%a.out
#SBATCH --error=logs/film_s_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-20

# ---- paths ----
SIF=/lustre/nyx/panda/markhoff/containers/tf-rocm/tensorflow_rocm6.4.1-py3.12-tf2.18-dev.sif
WORKDIR=/lustre/panda/markhoff/dirc/mldirc/dense/film
LOGDIR=$WORKDIR/logs

mkdir -p $LOGDIR

# ---- scan params ----
S_MIN=0.2
S_MAX=2.0
N_POINTS=21

S=$(awk -v i="$i" -v n="$N_POINTS" -v smin="$S_MIN" -v smax="$S_MAX" \
    'BEGIN {
        ratio = smax / smin
        s = smin * exp( (i / (n - 1)) * log(ratio) )
        printf "%.4f", s
    }')

echo "[INFO] Running with s = $S"

# ---- run inside container ----
apptainer exec \
  --rocm \
  --bind $WORKDIR:/workspace \
  --pwd /workspace \
  $SIF \
  python prtai_film.py -s $S
