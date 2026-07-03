#!/usr/bin/env python3
"""Aggregate per-sample metrics.csv files and write test-level results.csv."""

from __future__ import annotations

import argparse
import os
import random

from tqdm import tqdm
from omegaconf import OmegaConf

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _safe_stats(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std())


def _iter_sample_dirs(test_exp_dir: Path):
    audiosamples_dir = test_exp_dir / "audiosamples"
    if not audiosamples_dir.is_dir():
        raise FileNotFoundError(f"audiosamples directory not found: {audiosamples_dir}")
    for path in sorted(audiosamples_dir.iterdir()):
        if path.is_dir():
            yield path


def _read_metrics_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            val = row[1].strip()
            if not key or key.lower() == "metric":
                continue
            try:
                out[key] = float(val)
            except ValueError:
                continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_exp_dir", required=True, type=str)
    parser.add_argument("--cfg_path", type=str, required=False, default=None)
    parser.add_argument("--metrics_name", default="metrics.csv", type=str)
    parser.add_argument("--out_name", default="results.csv", type=str)
    parser.add_argument("--eval_bound", default=False, type=bool)
    args = parser.parse_args()

    if args.cfg_path is not None:
        cfg = OmegaConf.load(args.cfg_path)
    else:
        cfg = OmegaConf.load(os.path.join(args.test_exp_dir, "../.hydra/config.yaml"))

    seed = int(getattr(getattr(cfg, "train", None), "seed", 0))
    np.random.seed(seed)
    random.seed(seed)

    test_dir = Path(args.test_exp_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test_exp_dir not found: {test_dir}")

    # Keep exactly the same output metric order/labels as test.py.
    if not args.eval_bound:
        ordered_metrics = [
            ("DNSMOS_SIG", "DNSMOS_P835_SIG"),
            ("DNSMOS_BAK", "DNSMOS_P835_BAK"),
            ("DNSMOS_OVRL", "DNSMOS_P835_OVRL"),
            ("DNSMOS_P808", "DNSMOS_P808"),
            ("CosSim", "CosSim(%)"),
            ("dWER", "dWER(%)"),
            ("PESQ", "PESQ"),
            ("ESTOI", "ESTOI"),
            ("SI_SDR", "SI-SDR(dB)"),
            ("GFLOPS", "GFLOPs"),
        ]
    else:
        ordered_metrics = [
            ("DNSMOS_SIG", "DNSMOS_P835_SIG"),
            ("DNSMOS_BAK", "DNSMOS_P835_BAK"),
            ("DNSMOS_OVRL", "DNSMOS_P835_OVRL"),
            ("DNSMOS_P808", "DNSMOS_P808"),
            ("noisy_DNSMOS_SIG", "noisy_DNSMOS_P835_SIG"),
            ("noisy_DNSMOS_BAK", "noisy_DNSMOS_P835_BAK"),
            ("noisy_DNSMOS_OVRL", "noisy_DNSMOS_P835_OVRL"),
            ("noisy_DNSMOS_P808", "noisy_DNSMOS_P808"),
            ("clean_DNSMOS_SIG", "clean_DNSMOS_P835_SIG"),
            ("clean_DNSMOS_BAK", "clean_DNSMOS_P835_BAK"),
            ("clean_DNSMOS_OVRL", "clean_DNSMOS_P835_OVRL"),
            ("clean_DNSMOS_P808", "clean_DNSMOS_P808"),
            ("CosSim", "CosSim(%)"),
            ("dWER", "dWER(%)"),
            ("noisy_dWER", "noisy_dWER(%)"),
            ("PESQ", "PESQ"),
            ("ESTOI", "ESTOI"),
            ("SI_SDR", "SI-SDR(dB)"),
            ("GFLOPS", "GFLOPs"),
        ]

    metric_values: Dict[str, List[float]] = {k: [] for k, _ in ordered_metrics}

    for sample_dir in _iter_sample_dirs(test_dir):
        per_sample = _read_metrics_csv(sample_dir / args.metrics_name)
        for key in metric_values.keys():
            if key in per_sample:
                metric_values[key].append(float(per_sample[key]))

    rows = [["Metric", "Mean", "Std"]]
    for key, label in ordered_metrics:
        mean, std = _safe_stats(metric_values[key])
        if mean is None:
            rows.append([label, "N/A", "N/A"])
        else:
            rows.append([label, f"{mean:.4f}", f"{std:.4f}"])

    out_path = test_dir / args.out_name
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
