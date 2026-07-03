#!/bin/bash

if conda env list | grep -qE '^\s*eval_baselines\s'; then
  conda remove -yn eval_baselines --all
fi

conda create -yn eval_baselines -c conda-forge python=3.10 libsndfile wandb sentencepiece cmake ninja pkg-config

# Install dependencies
if [ -d $HOME/miniconda3/envs/eval_baselines ]; then
  CONDA_ENV_PATH=$HOME/miniconda3/envs/eval_baselines
else
  CONDA_ENV_PATH=$HOME/.conda/envs/eval_baselines
fi

$CONDA_ENV_PATH/bin/pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

$CONDA_ENV_PATH/bin/pip install \
  tqdm \
  soundfile \
  hydra-core \
  librosa \
  matplotlib \
  descript-audio-codec \
  asteroid-filterbanks \
  asteroid 
  # openai-whisper \
  # git+https://github.com/wenet-e2e/wespeaker.git \
  # pyannote.audio \
  # git+https://github.com/marianne-m/brouhaha-vad.git
