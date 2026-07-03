#!/usr/bin/env python3
"""Inference-only test script compatible with scripts/test.sh metric pipeline."""

import argparse
import os
import random
from pathlib import Path

import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import tqdm
from omegaconf import OmegaConf
from torch.utils.flop_counter import FlopCounterMode

from utils.metric_csv_common import update_metrics_csv
from utils.visualization import save_spectrograms


def run_inference_with_gflops(model, batch):
    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        output = model.separate_batch_long(batch)
    gflops = float(flop_counter.get_total_flops()) / 1e9
    return output, gflops


@torch.no_grad()
def run_inference(
    model,
    labeled_data_list,
    *,
    sample_rate=16000,
    output_dir="eval_output",
    tiny=False,
    max_eval_samples=None,
):
    """
    Run model inference only and save per-sample audio files.

    Output layout:
      {output_dir}/audiosamples/{name}/reconstructed.wav
      {output_dir}/audiosamples/{name}/noisy.wav
      {output_dir}/audiosamples/{name}/clean.wav
      {output_dir}/audiosamples/{name}/spectrogram.png
      {output_dir}/audiosamples/{name}/metrics.csv
    """
    if tiny:
        tiny_data_ids = [
            "1089-134691-0018_4970-29095-0024",
            "2961-961-0000_237-134500-0013",
            "8555-284447-0020_908-157963-0014",
            "8555-284447-0017_1284-1180-0005",
            "672-122797-0073_7127-75946-0006",
            "8455-210777-0029_7176-92135-0017",
            "2961-960-0009_6829-68769-0027",
            "260-123440-0007_1320-122612-0005",
            "5639-40744-0015_5683-32866-0017",
            "3570-5694-0010_7176-92135-0002",
            "121-127105-0000_2961-961-0016",
        ]
        labeled_data_list = [d for d in labeled_data_list if d[1].split("/")[-1].split(".")[0] in tiny_data_ids]

    if max_eval_samples is not None:
        max_eval_samples = int(max_eval_samples)
        if max_eval_samples <= 0:
            raise ValueError(f"max_eval_samples must be >= 1, got {max_eval_samples}")
        labeled_data_list = labeled_data_list[:max_eval_samples]

    os.makedirs(output_dir, exist_ok=True)
    audio_root = os.path.join(output_dir, "audiosamples")
    os.makedirs(audio_root, exist_ok=True)

    model.eval()
    num_done = 0
    for noisy_wav_path, clean_wav_path, name in tqdm.tqdm(labeled_data_list, desc="Inference"):
        noisy_wav = torchaudio.load(noisy_wav_path)[0]
        clean_wav = torchaudio.load(clean_wav_path)[0]

        batch = {
            "noisy_wav": noisy_wav,
            "clean_wav": clean_wav,
            "metadata": {"name": name},
        }

        output, gflops = run_inference_with_gflops(model, batch)
        reconstructed = output["reconstructed"]
        noisy = output["noisy"]
        clean = output["clean"]
        sample_name = output["metadata"]["name"]

        if reconstructed.dim() == 3 and reconstructed.shape[1] == 1:
            reconstructed = reconstructed.squeeze(1)
        if reconstructed.dim() == 2 and reconstructed.shape[0] == 1:
            reconstructed = reconstructed.squeeze(0)

        reconstructed_np = reconstructed.detach().cpu().numpy().squeeze()
        noisy_np = noisy.detach().cpu().numpy().squeeze()
        clean_np = clean.detach().cpu().numpy().squeeze()

        min_len = min(reconstructed_np.shape[0], clean_np.shape[0], noisy_np.shape[0])
        reconstructed_np = reconstructed_np[:min_len]
        noisy_np = noisy_np[:min_len]
        clean_np = clean_np[:min_len]

        sample_dir = os.path.join(audio_root, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        reconstructed_path = os.path.join(sample_dir, "reconstructed.wav")
        noisy_path = os.path.join(sample_dir, "noisy.wav")
        clean_path = os.path.join(sample_dir, "clean.wav")
        spectrogram_path = os.path.join(sample_dir, "spectrogram.png")

        sf.write(reconstructed_path, reconstructed_np, sample_rate)
        sf.write(noisy_path, noisy_np, sample_rate)
        sf.write(clean_path, clean_np, sample_rate)
        save_spectrograms(noisy_np, clean_np, reconstructed_np, sample_rate, spectrogram_path)
        if gflops is not None:
            update_metrics_csv(Path(sample_dir), {"GFLOPS": gflops})
        num_done += 1

    # Minimal summary file for traceability.
    with open(os.path.join(output_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"Inference completed. n={num_done}\n")

    print(f"Inference completed. n={num_done}")
    print(f"Outputs: {output_dir}")


def main(
    exp_dir,
    checkpoint_path=None,
    cfg_path=None,
    exp_name=None,
    tiny=False,
    max_eval_samples=None,
    dataset_cfg_path=None,
    num_steps=None,
    decoding_policy=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg_path is not None:
        cfg = OmegaConf.load(cfg_path)
    else:
        cfg = OmegaConf.load(os.path.join(exp_dir, ".hydra/config.yaml"))

    OmegaConf.set_struct(cfg, False)
    model_cfg = cfg.model.model if "model" in cfg.model else cfg.model
    target = str(model_cfg.get("_target_", ""))
    is_marse = target.endswith("MARSE_Model")
    if num_steps is not None and ("num_steps" in model_cfg or is_marse):
        model_cfg.num_steps = int(num_steps)
    if decoding_policy is not None and ("decoding_policy" in model_cfg or is_marse):
        model_cfg.decoding_policy = str(decoding_policy)
    OmegaConf.set_struct(cfg, True)

    seed = int(getattr(getattr(cfg, "train", None), "seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if dataset_cfg_path is not None:
        dataset_cfg = OmegaConf.load(dataset_cfg_path).test_dataset
    else:
        dataset_cfg = cfg.test_dataset if "test_dataset" in cfg else cfg.val_dataset

    if "labeled_data_dirs" in dataset_cfg:
        labeled_data_dirs = dataset_cfg.labeled_data_dirs
    else:
        labeled_data_dirs = []

    labeled_data_list = []
    for labeled_data_dir in labeled_data_dirs:
        noisy_wav_list = librosa.util.find_files(labeled_data_dir["noisy_dir"], ext="wav")
        clean_wav_list = librosa.util.find_files(labeled_data_dir["clean_dir"], ext="wav")
        labeled_data_list.extend(
            list(
                zip(
                    noisy_wav_list,
                    clean_wav_list,
                    [f"{labeled_data_dir['name']}_{p.split('/')[-1].split('.')[0]}" for p in noisy_wav_list],
                )
            )
        )

    if "model" in cfg.model:
        OmegaConf.set_struct(cfg.model, False)
        cfg.model.dim = cfg.model.model.dim
        OmegaConf.set_struct(cfg.model, True)
    model = hydra.utils.instantiate(cfg.model.model if "model" in cfg.model else cfg.model, device=device)
    model = model.to(device)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        if hasattr(model, "module"):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")

    sample_rate = cfg.model.dac_cfg.sample_rate if "dac_cfg" in cfg.model else cfg.model.model.dac_cfg.sample_rate
    output_tag = exp_name if exp_name is not None else os.path.basename(checkpoint_path).split(".")[0]
    output_dir = os.path.join(exp_dir, f"{output_tag}")
    run_inference(
        model,
        labeled_data_list,
        sample_rate=sample_rate,
        output_dir=output_dir,
        tiny=tiny,
        max_eval_samples=max_eval_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=False, default=None)
    parser.add_argument("--cfg_path", type=str, required=False, default=None)
    parser.add_argument("--exp_name", type=str, required=False, default=None)
    parser.add_argument("--dataset_cfg_path", type=str, required=False, default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        required=False,
        default=None,
        help="Infer only the first N samples (default: all).",
    )
    parser.add_argument("--num_steps", type=int, required=False, default=None)
    parser.add_argument(
        "--decoding_policy",
        type=str,
        required=False,
        default=None,
        choices=["random", "causal", "oracle"],
        help="Override MARSE decoding policy at inference time.",
    )
    _args = parser.parse_args()
    main(
        _args.exp_dir,
        _args.checkpoint_path,
        _args.cfg_path,
        _args.exp_name,
        _args.tiny,
        _args.max_eval_samples,
        _args.dataset_cfg_path,
        _args.num_steps,
        _args.decoding_policy,
    )
