#!/bin/bash
#SBATCH --job-name=clconv
#SBATCH --output=logs/clconv_%A_%a.out
#SBATCH --error=logs/clconv_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=main
#SBATCH --mem=8G
#SBATCH --array=0-31

# ----- Paths ----- #

INPUT_ROOT=/lustre/nyx/panda/markhoff/dirc/prtdirc/macro/piahitsout/piahits.root
SIF=/lustre/nyx/panda/markhoff/containers/root/root_6.30.04-ubuntu22.04.sif
WORKDIR=/lustre/panda/nyx/markhoff/dirc/mldirc/dense/dualfilm
LOGDIR=$WORKDIR/logs

mkdir -p $LOGDIR

# ----- Run ----- #

TOTAL=10783500
CHUNK=$((TOTAL/32))

START=$((SLURM_ARRAY_TASK_ID*CHUNK))
END=$(((SLURM_ARRAY_TASK_ID+1)*CHUNK))

singularity exec \
	$SIF \
    python clconv.py \
    -i $INPUT_ROOT \
    -o data \
    --start $START \
    --end $END \
    --shuffle


