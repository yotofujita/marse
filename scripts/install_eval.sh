#!/bin/bash

set -euo pipefail

if conda env list | grep -qE '^\s*sseeval\s'; then
  conda remove -yn sseeval --all
fi

conda create -yn sseeval -c conda-forge python=3.10 libsndfile wandb sentencepiece cmake ninja pkg-config

# Install dependencies
if [ -d $HOME/miniconda3/envs/sseeval ]; then
  CONDA_ENV_PATH=$HOME/miniconda3/envs/sseeval
else
  CONDA_ENV_PATH=$HOME/.conda/envs/sseeval
fi

$CONDA_ENV_PATH/bin/pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

$CONDA_ENV_PATH/bin/pip install \
  tqdm \
  soundfile \
  hydra-core \
  librosa \
  pystoi \
  pesq \
  transformers \
  jiwer \
  torchmetrics \
  matplotlib \
  einops \
  onnxruntime \
  onnxruntime-gpu \
  peft \
  s3prl \
  torchcodec \
  descript-audio-codec \
  # openai-whisper \
  # git+https://github.com/wenet-e2e/wespeaker.git \
  # pyannote.audio \
  # git+https://github.com/marianne-m/brouhaha-vad.git
