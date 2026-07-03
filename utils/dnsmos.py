#!/usr/bin/env python3
"""Compute DNSMOS for each sample and write values to metrics.csv."""

from __future__ import annotations

import argparse
import os
import random

from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np
import torch
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

try:
    from utils.metric_csv_common import iter_sample_dirs, load_audio, update_metrics_csv
except:
    from metric_csv_common import iter_sample_dirs, load_audio, update_metrics_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_exp_dir", required=True, type=str)
    parser.add_argument("--cfg_path", type=str, required=False, default=None)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_items", default=None, type=int)
    parser.add_argument("--eval_bound", default=False, type=bool)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.cfg_path is not None:
        cfg = OmegaConf.load(args.cfg_path)
    else:
        cfg = OmegaConf.load(os.path.join(args.test_exp_dir, "../.hydra/config.yaml"))

    seed = int(getattr(getattr(cfg, "train", None), "seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    dnsmos_device = "cpu" if device.type == "cpu" else f"{device.type}:0"
    dnsmos = DeepNoiseSuppressionMeanOpinionScore(fs=16000, personalized=False, device=dnsmos_device)

    for idx, sample_dir in enumerate(tqdm(iter_sample_dirs(args.test_exp_dir), desc="dnsmos computation")):
        if args.max_items is not None and idx >= int(args.max_items):
            break
        try:
            if not args.eval_bound:
                enhanced, _ = load_audio(sample_dir / "reconstructed.wav", target_sr=16000)
                audio = torch.from_numpy(enhanced).float().to(device)
                scores = dnsmos(audio).detach().cpu().numpy()
                update_metrics_csv(
                    sample_dir,
                    {
                        "DNSMOS_P808": float(scores[0]),
                        "DNSMOS_SIG": float(scores[1]),
                        "DNSMOS_BAK": float(scores[2]),
                        "DNSMOS_OVRL": float(scores[3]),
                    },
                )
            else:
                enhanced, _ = load_audio(sample_dir / "reconstructed.wav", target_sr=16000)
                audio = torch.from_numpy(enhanced).float().to(device)
                scores = dnsmos(audio).detach().cpu().numpy()
                noisy, _ = load_audio(sample_dir / "noisy.wav", target_sr=16000)
                audio_n = torch.from_numpy(noisy).float().to(device)
                scores_n = dnsmos(audio_n).detach().cpu().numpy()
                clean, _ = load_audio(sample_dir / "clean.wav", target_sr=16000)
                audio_c = torch.from_numpy(clean).float().to(device)
                scores_c = dnsmos(audio_c).detach().cpu().numpy()
                update_metrics_csv(
                    sample_dir,
                    {
                        "DNSMOS_P808": float(scores[0]),
                        "DNSMOS_SIG": float(scores[1]),
                        "DNSMOS_BAK": float(scores[2]),
                        "DNSMOS_OVRL": float(scores[3]),
                        "noisy_DNSMOS_P808": float(scores_n[0]),
                        "noisy_DNSMOS_SIG": float(scores_n[1]),
                        "noisy_DNSMOS_BAK": float(scores_n[2]),
                        "noisy_DNSMOS_OVRL": float(scores_n[3]),
                        "clean_DNSMOS_P808": float(scores_c[0]),
                        "clean_DNSMOS_SIG": float(scores_c[1]),
                        "clean_DNSMOS_BAK": float(scores_c[2]),
                        "clean_DNSMOS_OVRL": float(scores_c[3]),
                    },
                )
        except Exception as e:
            print(f"[WARN] {sample_dir.name}: {e}")


if __name__ == "__main__":
    main()
