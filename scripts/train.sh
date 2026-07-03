#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-sse}"
if [ -d "${HOME}/miniconda3/envs/${CONDA_ENV_NAME}" ]; then
  CONDA_ENV_PATH="${HOME}/miniconda3/envs/${CONDA_ENV_NAME}"
else
  CONDA_ENV_PATH="${HOME}/.conda/envs/${CONDA_ENV_NAME}"
fi

if [ ! -x "$CONDA_ENV_PATH/bin/python" ]; then
  echo "Conda environment not found: $CONDA_ENV_NAME" >&2
  echo "Expected python at $CONDA_ENV_PATH/bin/python" >&2
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

# Select project, model, and dataset
DEBUG=${DEBUG:-false}
PROJECT="${PROJECT:-mgse}"
MODEL="${MODEL:-marse}"
DATASET="${DATASET:-labeled_libri_mix}"
N_EPOCHS="${N_EPOCHS:-300}"

echo "Project: $PROJECT"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Conda environment: $CONDA_ENV_NAME"
echo "Number of epochs: $N_EPOCHS"

export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"

# Configuration
SCRIPT="train.py \
 --config-name $PROJECT \
 model_name=$MODEL model=$MODEL \
 dataset_name=$DATASET dataset=$DATASET \
 train.num_epochs=$N_EPOCHS"

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_port=29500 \
  $SCRIPT

echo "Training completed!" 
