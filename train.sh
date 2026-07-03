#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --tmp=40G

GPU_NODE_TYPE_RAW="${GPU_NODE_TYPE:-auto}"
GPU_NODE_TYPE="$(printf '%s' "$GPU_NODE_TYPE_RAW" | tr '[:upper:]' '[:lower:]')"

resolve_conda_env_name() {
  case "$GPU_NODE_TYPE" in
    a100)
      printf '%s\n' "sse-V100"
      ;;
    v100)
      printf '%s\n' "sse-V100"
      ;;
    auto)
      local detected_gpu
      detected_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
      if [[ "$detected_gpu" == *"A100"* ]]; then
        printf '%s\n' "sse-V100"
      elif [[ "$detected_gpu" == *"V100"* ]]; then
        printf '%s\n' "sse-V100"
      else
        printf '%s\n' "sse"
      fi
      ;;
    *)
      echo "Unsupported GPU_NODE_TYPE: $GPU_NODE_TYPE_RAW" >&2
      echo "Use one of: auto, A100, V100" >&2
      exit 1
      ;;
  esac
}

# Module load
if command -v module &> /dev/null; then
  module load gcc/15.1.0/gcc-15.1.0
  module load miniconda3/25.5.1/none-none
  case "$GPU_NODE_TYPE" in
    a100)
      module load cuda/13.0.2/none-none
      ;;
    v100)
      module load cuda/12.8.1/none-none
      ;;
    auto)
      local detected_gpu
      detected_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
      if [[ "$detected_gpu" == *"A100"* ]]; then
        module load cuda/13.0.2/none-none
      elif [[ "$detected_gpu" == *"V100"* ]]; then
        module load cuda/12.8.1/none-none
      fi
      ;;
  esac
  cd /gpfs/workdir/fujitayo/semi-supervisedSE
else
  REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
  cd "$REPO_ROOT"
fi

# Get conda environment path
if [ -z "$HOME" ]; then
  HOME=~
fi

CONDA_ENV_NAME="$(resolve_conda_env_name)"

if [ -d "$HOME/miniconda3/envs/$CONDA_ENV_NAME" ]; then
  CONDA_ENV_PATH="$HOME/miniconda3/envs/$CONDA_ENV_NAME"
else
  CONDA_ENV_PATH="$HOME/.conda/envs/$CONDA_ENV_NAME"
fi

if [ ! -x "$CONDA_ENV_PATH/bin/python" ]; then
  echo "Conda environment not found: $CONDA_ENV_NAME" >&2
  echo "Expected python at $CONDA_ENV_PATH/bin/python" >&2
  exit 1
fi

# Number of processes per node (default: 1, can be overridden)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# Select project, model, and dataset
DEBUG=${DEBUG:-false}
PROJECT="${PROJECT:-mgse}"
MODEL="${MODEL:-marse}"
DATASET="${DATASET:-labeled_libri_mix}"
N_EPOCHS="${N_EPOCHS:-300}"

echo "Project: $PROJECT"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "GPU node type: $GPU_NODE_TYPE_RAW"
echo "Conda environment: $CONDA_ENV_NAME"
echo "Number of epochs: $N_EPOCHS"

source activate $CONDA_ENV_NAME

if command -v module &> /dev/null; then
  CONDA_ENV_PATH="$HOME/.conda/envs/$CONDA_ENV_NAME"
else
  CONDA_ENV_PATH="$HOME/miniconda3/envs/$CONDA_ENV_NAME"
fi
export LD_PRELOAD="$CONDA_ENV_PATH/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"
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
