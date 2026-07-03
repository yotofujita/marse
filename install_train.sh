#!/bin/bash

set -euo pipefail

# Load cluster modules when available.
if command -v module &> /dev/null; then
  module purge
  module load miniconda3/25.5.1/none-none
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
conda_env_name="sse"
torch_index_url="https://download.pytorch.org/whl/cu130"

if [[ "$gpu_name" == *"A100"* ]]; then
  conda_env_name="sse-A100"
  if command -v module &> /dev/null; then
    module load cuda/13.0.2/none-none
  fi
elif [[ "$gpu_name" == *"V100"* ]]; then
  conda_env_name="sse-V100"
  torch_index_url="https://download.pytorch.org/whl/cu128"
  if command -v module &> /dev/null; then
    module load cuda/12.8.1/none-none
  fi
fi

echo "gpu name: ${gpu_name}"
echo "conda_env_name: ${conda_env_name}"
echo "torch_index_url: ${torch_index_url}"

if conda env list | awk '{print $1}' | grep -Fxq "$conda_env_name"; then
  conda remove -yn "$conda_env_name" --all
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -yn "$conda_env_name" -c conda-forge \
  python=3.10 ffmpeg=8 "libstdcxx-ng>=12" "libgcc-ng>=12" \
  libsndfile wandb sentencepiece cmake ninja pkg-config

# torch / vision / audio
conda run -n "$conda_env_name" python -m pip install --upgrade "pip<24.1"
conda run -n "$conda_env_name" python -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url "$torch_index_url"

# torchcodec も同じ index から
conda run -n "$conda_env_name" python -m pip install \
  torchcodec==0.10.0+cu128 \
  --index-url "$torch_index_url"

conda run -n "$conda_env_name" python -m pip install \
  omegaconf==2.0.6 hydra-core==1.0.7 \
  tqdm ipdb comet_ml soundfile pystoi matplotlib librosa scipy einops descript-audio-codec

conda run -n "$conda_env_name" python -c "import dac; dac.utils.download(model_type='16khz')"