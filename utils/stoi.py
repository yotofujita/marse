#!/usr/bin/env python3
"""Compute ESTOI for each sample and write values to metrics.csv."""

from __future__ import annotations

import argparse
import os
import random

from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np

try:
    from utils.metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv
    from utils.metrics import compute_stoi
except:
    from metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv
    from metrics import compute_stoi

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

    for idx, sample_dir in enumerate(tqdm(iter_sample_dirs(args.test_exp_dir), desc="stoi computation")):
        if args.max_items is not None and idx >= int(args.max_items):
            break
        try:
            enhanced, clean, sr = load_aligned_pair(sample_dir)
            score = compute_stoi(enhanced, clean, sample_rate=sr, extended=True)
            update_metrics_csv(sample_dir, {"ESTOI": float(score)})
        except Exception as e:
            print(f"[WARN] {sample_dir.name}: {e}")


if __name__ == "__main__":
    main()
