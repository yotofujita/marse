import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

try:
    from .architectures import Conformer
    from .io_representations import DACRepresentation
    from .modules.datasets.labeled_single_channel_se_dataset import LabeledSingleChannelSESample
    from .modules.supervised_single_channel_se_module import (
        SingleChannelSEOutput,
        SupervisedSingleChannelSEModule,
    )
except ImportError:
    from architectures import Conformer
    from io_representations import DACRepresentation
    from modules.datasets.labeled_single_channel_se_dataset import LabeledSingleChannelSESample
    from modules.supervised_single_channel_se_module import SingleChannelSEOutput, SupervisedSingleChannelSEModule


def env_flag(name: str, default: bool = False) -> bool:
    import os

    value = os.environ.get(name, None)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def FeedForward(*, dim: int, mult: int = 4, dropout: float = 0.0) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim),
    )


class C_AR_Model(SupervisedSingleChannelSEModule):
    def __init__(
        self,
        *,
        input_dim: int,
        dim: int,
        max_seq_len: int,
        dac_cfg: dict,
        conformer_cfg: dict,
        device: str = "cuda",
        debug: bool = False,
    ):
        super().__init__()

        self.dac_representation = DACRepresentation(**dac_cfg, device=device)
        self.device = device

        self.max_seq_len = max_seq_len
        self.dim = dim
        self.input_dim = input_dim
        self.debug = debug

        # PE_t: allow up to (noisy_len + 1 + clean_len) where both noisy/clean lengths are <= max_seq_len.
        self.spatial_pos_emb = nn.Embedding(int(2 * max_seq_len) + 1, self.dim)
        self.spatial_start_token = nn.Parameter(torch.randn(self.dim))

        # C-AR uses a causal Conformer.
        self.noise_transformer = Conformer(**conformer_cfg, causal=True)

        self.input_layer = nn.Linear(self.input_dim, self.dim)
        self.output_layer = nn.Linear(self.dim, self.input_dim)

    def _forward(
        self,
        noisy_embeds: torch.Tensor,
        clean_embeds_quant: torch.Tensor,
        return_loss: bool = False,
        clean_embeds_gt: torch.Tensor = None,
    ):
        batch_size, noisy_len, _ = noisy_embeds.shape
        clean_len = clean_embeds_quant.shape[1]
        device = noisy_embeds.device

        tokens_noisy = self.input_layer(noisy_embeds)
        tokens_clean = self.input_layer(clean_embeds_quant)

        spatial_pos = self.spatial_pos_emb(torch.arange(noisy_len + 1 + clean_len, device=device))
        spatial_tokens = torch.cat(
            (
                tokens_noisy,
                repeat(self.spatial_start_token, "d -> b 1 d", b=batch_size),
                tokens_clean,
            ),
            dim=1,
        )
        spatial_tokens = spatial_tokens + spatial_pos

        out = self.noise_transformer(spatial_tokens)
        out = out[:, noisy_len:, :]  # keep [<start>, clean_tokens]
        out_embeds = self.output_layer(out)

        if not return_loss:
            return out_embeds

        preds = out_embeds[:, :-1]  # predict clean_0..clean_{t-1}
        mse_loss = F.mse_loss(preds, clean_embeds_gt)
        loss = 0.25 * mse_loss
        return loss, {"mse_loss": mse_loss}

    def _generate_autoregressive(self, noisy_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = noisy_embeds.shape
        device = noisy_embeds.device

        generated = torch.empty((batch_size, 0, embed_dim), dtype=noisy_embeds.dtype, device=device)
        for _ in range(seq_len):
            out_embeds = self._forward(noisy_embeds, generated, return_loss=False)  # (B, 1 + t, D)
            next_embed = out_embeds[:, -1:, :]
            next_embed_quant = self.dac_representation.quantize(
                next_embed.transpose(2, 1)
            ).transpose(2, 1)
            generated = torch.cat([generated, next_embed_quant], dim=1)
        return generated

    # ===== SingleChannelSEModel interface =====

    def forward(self, batch: LabeledSingleChannelSESample):
        noisy_wav = batch["noisy_wav"].to(self.device)
        clean_wav = batch["clean_wav"].to(self.device)

        noisy_embeds = self.dac_representation.encode(noisy_wav)
        clean_embeds = self.dac_representation.encode(clean_wav)
        clean_embeds_quantized = self.dac_representation.quantize(clean_embeds)
        noisy_embeds = rearrange(noisy_embeds, "b d t -> b t d")
        clean_embeds = rearrange(clean_embeds, "b d t -> b t d")
        clean_embeds_quantized = rearrange(clean_embeds_quantized, "b d t -> b t d")

        # Teacher forcing uses the full clean embedding sequence, as in the original implementation.
        loss, loss_dict = self._forward(
            noisy_embeds,
            clean_embeds_quantized,
            return_loss=True,
            clean_embeds_gt=clean_embeds,
        )
        return loss, loss_dict

    @torch.no_grad()
    def separate_batch(self, batch: LabeledSingleChannelSESample) -> SingleChannelSEOutput:
        noisy_wav = batch["noisy_wav"].to(self.device)
        clean_wav = batch["clean_wav"].to(self.device)
        name = batch["metadata"].get("name", "unknown")

        noisy_embeds = self.dac_representation.encode(noisy_wav)
        clean_embeds = self.dac_representation.encode(clean_wav)
        noisy_embeds = rearrange(noisy_embeds, "b d t -> b t d")
        clean_embeds = rearrange(clean_embeds, "b d t -> b t d")

        generated = self._generate_autoregressive(noisy_embeds)  # (B, T, D)

        generated = rearrange(generated, "b t d -> b d t")
        noisy_embeds = rearrange(noisy_embeds, "b t d -> b d t")
        clean_embeds = rearrange(clean_embeds, "b t d -> b d t")

        reconstructed = self.dac_representation.decode(generated, quantized=True)
        noisy = self.dac_representation.decode(noisy_embeds)
        clean = self.dac_representation.decode(clean_embeds)

        return {
            "reconstructed": reconstructed,
            "noisy": noisy,
            "clean": clean,
            "metadata": {"name": name},
        }
