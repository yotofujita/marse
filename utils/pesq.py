#!/usr/bin/env python3
"""Compute PESQ for each sample and write values to metrics.csv."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
import argparse
import os
import random

from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np
import librosa

try:
    from utils.metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv
except:
    from metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv


def _load_pesq_func():
    """Load `pesq.pesq` while avoiding self-import from this script file."""
    this_dir = str(Path(__file__).resolve().parent)
    old_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p and str(Path(p).resolve()) != this_dir]
        module = importlib.import_module("pesq")
        return getattr(module, "pesq")
    finally:
        sys.path = old_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_exp_dir", required=True, type=str)
    parser.add_argument("--cfg_path", type=str, required=False, default=None)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_items", default=None, type=int)
    args = parser.parse_args()


    if args.cfg_path is not None:
        cfg = OmegaConf.load(args.cfg_path)
    else:
        cfg = OmegaConf.load(os.path.join(args.test_exp_dir, "../.hydra/config.yaml"))

    seed = int(getattr(getattr(cfg, "train", None), "seed", 0))
    np.random.seed(seed)
    random.seed(seed)

    pesq_func = _load_pesq_func()

    for idx, sample_dir in enumerate(tqdm(iter_sample_dirs(args.test_exp_dir), desc="pesq computation")):
        if args.max_items is not None and idx >= int(args.max_items):
            break
        try:
            enhanced, clean, sr = load_aligned_pair(sample_dir)
            if sr not in (8000, 16000):
                enhanced = librosa.resample(enhanced, orig_sr=sr, target_sr=16000)
                clean = librosa.resample(clean, orig_sr=sr, target_sr=16000)
                sr = 16000
            mode = "wb" if sr == 16000 else "nb"
            score = pesq_func(sr, clean, enhanced, mode)
            update_metrics_csv(sample_dir, {"PESQ": float(score)})
        except Exception as e:
            print(f"[WARN] {sample_dir.name}: {e}")


if __name__ == "__main__":
    main()
