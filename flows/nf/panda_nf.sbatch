#!/bin/bash

# Takes a directory containing pure pion or pure kaon shards and uses it to train a conditional flows model.

#SBATCH --job-name=condnflow
#SBATCH --output=logs/condnflow_%A_%a.out
#SBATCH --error=logs/condnflow_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=amd_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G


INPUT_DIR=/lustre/nyx/panda/markhoff/dirc/mldirc/flows/nf/randhits-kaons-p3.5
SIF=/lustre/nyx/panda/markhoff/containers/pytorch-rocm/pytorch_rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0.sif
WORKDIR=/lustre/nyx/panda/markhoff/dirc/mldirc/flows/nf
LOGDIR=$WORKDIR/logs

ID=$SLURM_JOB_ID
OUTPUT="kaons-p3.5-$ID"

singularity exec --bind $WORKDIR:$WORKDIR $SIF bash -c "
    export PYTHONPATH=$WORKDIR/pythonlib &&
    python panda_nf.py -i $INPUT_DIR -o $OUTPUT --batch
"