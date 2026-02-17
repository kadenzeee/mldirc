#!/bin/bash
#SBATCH --job-name=clconv
#SBATCH --output=clconv_%A_%a.out
#SBATCH --error=clconv_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --mem=8G
#SBATCH --array=0-31

TOTAL=10783500
CHUNK=$((TOTAL/32))

START=$((SLURM_ARRAY_TASK_ID*CHUNK))
END=$(((SLURM_ARRAY_TASK_ID+1)*CHUNK))

python clconv.py \
    -i piahits.root \
    -o data \
    --start $START \
    --end $END \
    --shuffle
