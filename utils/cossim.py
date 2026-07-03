#!/usr/bin/env python3
"""Compute CosSim for each sample and write values to metrics.csv."""

from __future__ import annotations

import argparse
import os
import random

from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

try:
    from utils.metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv
except:
    from metric_csv_common import iter_sample_dirs, load_aligned_pair, update_metrics_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_exp_dir", required=True, type=str)
    parser.add_argument("--cfg_path", type=str, required=False, default=None)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max_items", default=None, type=int)
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

    extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-sv").to(device).eval()

    for idx, sample_dir in enumerate(tqdm(iter_sample_dirs(args.test_exp_dir), desc="cossim computation")):
        if args.max_items is not None and idx >= int(args.max_items):
            break
        try:
            enhanced, clean, _ = load_aligned_pair(sample_dir, target_sr=16000)
            enhanced_inputs = extractor(enhanced, sampling_rate=16000, return_tensors="pt", padding=True)
            clean_inputs = extractor(clean, sampling_rate=16000, return_tensors="pt", padding=True)
            enhanced_inputs = {k: v.to(device) for k, v in enhanced_inputs.items()}
            clean_inputs = {k: v.to(device) for k, v in clean_inputs.items()}
            with torch.no_grad():
                enhanced_emb = model(**enhanced_inputs).embeddings
                clean_emb = model(**clean_inputs).embeddings
            enhanced_emb = F.normalize(enhanced_emb, p=2, dim=1)
            clean_emb = F.normalize(clean_emb, p=2, dim=1)
            cos_sim = F.cosine_similarity(enhanced_emb, clean_emb, dim=1).item() * 100.0
            update_metrics_csv(sample_dir, {"CosSim": float(cos_sim)})
        except Exception as e:
            print(f"[WARN] {sample_dir.name}: {e}")


if __name__ == "__main__":
    main()
