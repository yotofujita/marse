"""Common helpers for per-sample metric CSV scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import librosa
import numpy as np


def iter_sample_dirs(test_exp_dir: str) -> Iterable[Path]:
    test_dir = Path(test_exp_dir)
    audio_dir = test_dir / "audiosamples"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audiosamples directory not found: {audio_dir}")
    for path in sorted(audio_dir.iterdir()):
        if path.is_dir():
            yield path


def load_audio(path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, int(sr)


def load_aligned_pair(
    sample_dir: Path,
    *,
    enhanced_name: str = "reconstructed.wav",
    clean_name: str = "clean.wav",
    noisy_name: str = "noisy.wav",
    target_sr: Optional[int] = None,
    eval_bound: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    enhanced_path = sample_dir / enhanced_name
    clean_path = sample_dir / clean_name
    noisy_path = sample_dir / noisy_name
    if not enhanced_path.exists() or not clean_path.exists() or not noisy_path.exists():
        raise FileNotFoundError(f"missing wavs under {sample_dir}")

    enhanced, sr_e = load_audio(enhanced_path, target_sr=target_sr)
    clean, sr_c = load_audio(clean_path, target_sr=target_sr)
    if eval_bound:
        noisy, sr_n = load_audio(noisy_path, target_sr=target_sr)
    if sr_e != sr_c:
        enhanced = librosa.resample(enhanced, orig_sr=sr_e, target_sr=sr_c)
        sr_e = sr_c

    if not eval_bound:
        n = min(len(enhanced), len(clean))
        if n <= 0:
            raise ValueError(f"empty aligned pair under {sample_dir}")
        return enhanced[:n], clean[:n], int(sr_e)
    else:
        n = min(len(enhanced), len(clean), len(noisy))
        if n <= 0:
            raise ValueError(f"empty aligned pair under {sample_dir}")
        return enhanced[:n], clean[:n], noisy[:n], int(sr_e)


def read_metrics_csv(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
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


def write_metrics_csv(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics.keys()):
            value = metrics[key]
            writer.writerow([key, f"{float(value):.10f}"])


def update_metrics_csv(sample_dir: Path, updates: Dict[str, float]) -> None:
    csv_path = sample_dir / "metrics.csv"
    data = read_metrics_csv(csv_path)
    data.update(updates)
    write_metrics_csv(csv_path, data)
