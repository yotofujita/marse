# MASKED AUTOREGRESSIVE SPEECH ENHANCEMENT

This repository contains the training, inference, and evaluation code used to reproduce the MARSE, C-NAR, and C-AR experiments described in `docs/iwaenc2026_paper.md`.

## Scope

The reproduction trains three neural audio codec based speech enhancement models:

- `c_nar`: C-NAR model, all-at-once noisy-to-clean mapping.
- `c_ar`: C-AR model, frame-wise causal autoregressive decoding.
- `marse`: MARSE/MGSE model, trained once with random masking and evaluated with different decoding policies.

For MARSE, the training objective is shared across all policies. At inference time, set:

- `decoding_policy=random` for `MARSE-NC-random`.
- `decoding_policy=causal` for `MARSE-causal`.
- `decoding_policy=oracle` for `MARSE-NC-oracle`.

## Data Layout

The configs assume `<storage_dir>=./datasets`.

Dataset generation repositories are included as Git submodules:

```bash
git submodule update --init --recursive
```

Generate Libri1Mix-compatible in-domain data with the LibriMix generator:

```bash
mkdir -p datasets
cd external/LibriMix
./generate_librimix.sh ../../datasets
cd ../..
ln -sfnT Libri2Mix datasets/LibriMix
```

The upstream LibriMix script generates several variants by default. To save disk space, edit `external/LibriMix/generate_librimix.sh` before running it and keep only `n_src=2`, `freqs=16k`, `modes=min`, and `types=mix_single`.

Generate out-of-domain LibriDEMAND data with the Libri1MixDEMAND generator. Place DEMAND first under `datasets/DEMAND/<noise_type>/ch01.wav`, then run:

```bash
cd external/Libri1MixDEMAND
pip install -r requirements.txt
./generate_libri1mix_demand.sh ../../datasets
cd ../..
ln -sfnT Libri1MixDemand datasets/Libri1MixDEMAND
```

After generation, the following directories should exist:

```text
datasets/
  LibriMix/
    wav16k/min/train-360/{mix_single,s1}/
    wav16k/min/dev/{mix_single,s1}/
    wav16k/min/test/{mix_single,s1}/
  Libri1MixDEMAND/
    wav16k/min/train-100/{mix_single,s1}/
    wav16k/min/dev/{mix_single,s1}/
    wav16k/min/test/{mix_single,s1}/
```

`LibriMix` is used for in-domain Libri1Mix training, validation, and testing. `Libri1MixDEMAND` is used for out-of-domain LibriDEMAND evaluation.

## Installation

Training and evaluation environments are split because evaluation installs ASR and metric dependencies.

```bash
./install_train.sh
./install_eval.sh
```

## Training

The default Hydra config is `configs/mgse.yaml`.

```bash
# MARSE / MGSE
python train.py model=marse model_name=marse dataset=labeled_libri_mix dataset_name=labeled_libri_mix

# C-NAR
python train.py model=c_nar model_name=c_nar dataset=labeled_libri_mix dataset_name=labeled_libri_mix

# C-AR
python train.py model=c_ar model_name=c_ar dataset=labeled_libri_mix dataset_name=labeled_libri_mix
```

The local wrapper uses the same config names:

```bash
MODEL=marse DATASET=labeled_libri_mix ./train.sh
MODEL=c_nar DATASET=labeled_libri_mix ./train.sh
MODEL=c_ar DATASET=labeled_libri_mix ./train.sh
```

Outputs are written under:

```text
../outputs/mgse/<dataset_name>/<model_name>/<timestamp>/
```

## Inference And Evaluation

For MARSE Table 1 with 10 decoding steps:

```bash
export EXPDIR=../outputs/mgse/labeled_libri_mix/marse/<timestamp>
export CKPT=$EXPDIR/checkpoints/300ep.pt

# In-domain Libri1Mix
EXPNAME=libri1mix_marse_random DECODING_POLICY=random N_ITERS="10" DATASET_CFGPATH=configs/dataset/labeled_libri_mix.yaml bash test.sh
EXPNAME=libri1mix_marse_causal DECODING_POLICY=causal N_ITERS="10" DATASET_CFGPATH=configs/dataset/labeled_libri_mix.yaml bash test.sh
EXPNAME=libri1mix_marse_oracle DECODING_POLICY=oracle N_ITERS="10" DATASET_CFGPATH=configs/dataset/labeled_libri_mix.yaml bash test.sh

# Out-of-domain LibriDEMAND
EXPNAME=demand_marse_random DECODING_POLICY=random N_ITERS="10" DATASET_CFGPATH=configs/dataset/labeled_libri_mix_demand.yaml bash test.sh
EXPNAME=demand_marse_causal DECODING_POLICY=causal N_ITERS="10" DATASET_CFGPATH=configs/dataset/labeled_libri_mix_demand.yaml bash test.sh
EXPNAME=demand_marse_oracle DECODING_POLICY=oracle N_ITERS="10" DATASET_CFGPATH=configs/dataset/labeled_libri_mix_demand.yaml bash test.sh
```

For Fig. 2 style iteration sweeps:

```bash
EXPNAME=libri1mix_marse_random DECODING_POLICY=random N_ITERS="1 5 10 20 30 40 50" DATASET_CFGPATH=configs/dataset/labeled_libri_mix.yaml bash test.sh
EXPNAME=libri1mix_marse_causal DECODING_POLICY=causal N_ITERS="1 5 10 20 30 40 50" DATASET_CFGPATH=configs/dataset/labeled_libri_mix.yaml bash test.sh
EXPNAME=libri1mix_marse_oracle DECODING_POLICY=oracle N_ITERS="1 5 10 20 30 40 50" DATASET_CFGPATH=configs/dataset/labeled_libri_mix.yaml bash test.sh
```

For C-NAR and C-AR, set `EXPDIR` and `CKPT` to the corresponding run directory and use `N_ITERS="1"`. Extra MARSE policy settings are ignored by `inference.py` for these models.

Each evaluation directory contains per-sample audio, per-sample `metrics.csv`, and aggregate `results.csv`.
