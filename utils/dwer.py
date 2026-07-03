#!/usr/bin/env python3
"""Compute dWER for each sample and write values to metrics.csv."""

from __future__ import annotations

import argparse
import os
import random

from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np
import torch
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    from utils.metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv
except:
    from metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv


def _transcribe(audio, processor, model, device) -> str:
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


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
    random.seed(seed)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    for idx, sample_dir in enumerate(tqdm(iter_sample_dirs(args.test_exp_dir), desc="dwer computation")):
        if args.max_items is not None and idx >= int(args.max_items):
            break
        try:
            if not args.eval_bound:
                enhanced, clean, _ = load_aligned_pair(sample_dir, target_sr=16000)
                enhanced_text = _transcribe(enhanced, processor, model, device)
                clean_text = _transcribe(clean, processor, model, device)
                score = (wer(clean_text, enhanced_text) - wer(clean_text, clean_text)) * 100.0
                update_metrics_csv(sample_dir, {"dWER": float(score)})
            else:
                enhanced, clean, noisy, _ = load_aligned_pair(sample_dir, target_sr=16000, eval_bound=True)
                enhanced_text = _transcribe(enhanced, processor, model, device)
                noisy_text = _transcribe(noisy, processor, model, device)
                clean_text = _transcribe(clean, processor, model, device)
                score = (wer(clean_text, enhanced_text) - wer(clean_text, clean_text)) * 100.0
                score_n = (wer(clean_text, noisy_text) - wer(clean_text, clean_text)) * 100.0
                update_metrics_csv(sample_dir, {"dWER": float(score), "noisy_dWER": float(score_n)})
        except Exception as e:
            print(f"[WARN] {sample_dir.name}: {e}")

if __name__ == "__main__":
    main()
