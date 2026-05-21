#!/bin/bash

#SBATCH --job-name=4L256dimfwlmgnn
#SBATCH --output=logs/panda_gnn_4_layers_256_dim_%A_%a.out
#SBATCH --error=logs/panda_gnn_4_layers_256_dim_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=amd_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# -----------------------------
# Paths
# -----------------------------

INPUT_DIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph/fwlmgnn/data/randhits-p1.5-3.5-2x-photons-no-N/reduced/
SIF=/lustre/nyx/panda/markhoff/containers/pytorch-rocm/pytorch_rocm6.3.3_ubuntu24.04_py3.12_pytorch_release_2.4.0.sif
WORKDIR=/lustre/nyx/panda/markhoff/dirc/mldirc/graph/fwlmgnn
LOGDIR=$WORKDIR/logs

ID=$SLURM_JOB_ID
OUTPUT="2x-photons-p1.5-3.5-$ID"

singularity exec --bind $WORKDIR:$WORKDIR $SIF bash -c "
    pip install --quiet torch-geometric &&
    python panda_fwlmgnn.py -i $INPUT_DIR -o $OUTPUT -nepochs 50 -global-dim 7 --batch
"
