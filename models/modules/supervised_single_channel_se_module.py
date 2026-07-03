"""Abstract model classes for source separation."""

from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
from .datasets import LabeledSingleChannelSESample


SingleChannelSEOutput = {
    "reconstructed": torch.Tensor, # shape (B, T) or (B, 1, T)
    "noisy": torch.Tensor, # shape (B, T) or (B, 1, T)
    "clean": torch.Tensor, # shape (B, T) or (B, 1, T)
    "metadata": dict, # shape (B,)
}

class SupervisedSingleChannelSEModule(nn.Module, ABC):
    """
    Intermediate abstract class for single-channel single-speaker speech enhancement

    train.py / eval.py only call this interface,
    and model-specific processing (continuous/discrete representation, AR/NAR, etc.) is encapsulated within each model class.
    """

    @abstractmethod
    def forward(self, batch: LabeledSingleChannelSESample):
        """
        Loss calculation for training

        Args:
            batch: LabeledSingleChannelSESample
        Returns:
            loss: torch.Tensor, shape (1,)
        """
        raise NotImplementedError

    @abstractmethod
    def separate_batch(self, batch: LabeledSingleChannelSESample) -> SingleChannelSEOutput:
        """
        Separation processing for inference

        Args:
            batch: LabeledSingleChannelSESample
        Returns:
            SingleChannelSEOutput: Dictionary of separation results
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def separate_batch_long(self, batch: LabeledSingleChannelSESample) -> SingleChannelSEOutput:
        noisy_wav = batch["noisy_wav"].to(self.device)
        clean_wav = batch["clean_wav"].to(self.device)
        name = batch["metadata"].get("name", "unknown")

        # `test.py` provides waveforms as (channels, samples); training provides (batch, channels, samples).
        if noisy_wav.dim() == 3:
            assert noisy_wav.shape[0] == 1, "Batch size must be 1 for long audio processing"
            noisy_wav = noisy_wav.squeeze(0)
        elif noisy_wav.dim() == 1:
            noisy_wav = noisy_wav.unsqueeze(0)
        if clean_wav.dim() == 3:
            assert clean_wav.shape[0] == 1, "Batch size must be 1 for long audio processing"
            clean_wav = clean_wav.squeeze(0)
        elif clean_wav.dim() == 1:
            clean_wav = clean_wav.unsqueeze(0)

        assert noisy_wav.dim() == 2 and noisy_wav.shape[0] == 1, "Expected noisy_wav as (1, n_samples)"
        assert clean_wav.dim() == 2 and clean_wav.shape[0] == 1, "Expected clean_wav as (1, n_samples)"

        total_len = min(noisy_wav.shape[-1], clean_wav.shape[-1])
        seq_len_samples = int(self.max_seq_len * self.dac_representation.dac_model.hop_length)
        rem = 0

        if total_len < seq_len_samples:
            # Circular (wrap-around) padding: repeat waveform content instead of padding with zeros.
            repeat_times = int(math.ceil(seq_len_samples / total_len))
            noisy_wav = noisy_wav.repeat(1, repeat_times)[..., :seq_len_samples]
            clean_wav = clean_wav.repeat(1, repeat_times)[..., :seq_len_samples]
            n_blocks = 1
        else:
            full_blocks = total_len // seq_len_samples
            rem = int(total_len - full_blocks * seq_len_samples)

            if rem == 0:
                # No padding needed: exact multiple of `seq_len_samples`.
                n_blocks = int(full_blocks)
                noisy_wav = noisy_wav[..., :total_len]
                clean_wav = clean_wav[..., :total_len]
            else:
                # Tail-only overlap-add style processing:
                # - Run the first `full_blocks` consecutive blocks as-is.
                # - Run one additional window from the end (`[-seq_len_samples:]`).
                # - Keep only the last `rem` samples from the tail window output.
                n_blocks = int(full_blocks + 1)
                prefix_len = int(full_blocks * seq_len_samples)
                noisy_wav = torch.cat([noisy_wav[..., :prefix_len], noisy_wav[..., -seq_len_samples:]], dim=-1)
                clean_wav = torch.cat([clean_wav[..., :prefix_len], clean_wav[..., -seq_len_samples:]], dim=-1)

        noisy_wav_stacked = noisy_wav.reshape(1, n_blocks, seq_len_samples).permute(1, 0, 2)
        clean_wav_stacked = clean_wav.reshape(1, n_blocks, seq_len_samples).permute(1, 0, 2)

        output = self.separate_batch(
            {
                "noisy_wav": noisy_wav_stacked,
                "clean_wav": clean_wav_stacked,
                "metadata": {"name": f"{name}_stacked"},
            }
        )

        reconstructed_blocks = output["reconstructed"]
        noisy_blocks = output["noisy"]
        clean_blocks = output["clean"]

        if total_len >= seq_len_samples and rem > 0:
            reconstructed = torch.cat(
                [reconstructed_blocks[:-1].reshape(-1), reconstructed_blocks[-1].reshape(-1)[-rem:]],
                dim=0,
            )
            noisy = torch.cat([noisy_blocks[:-1].reshape(-1), noisy_blocks[-1].reshape(-1)[-rem:]], dim=0)
            clean = torch.cat([clean_blocks[:-1].reshape(-1), clean_blocks[-1].reshape(-1)[-rem:]], dim=0)
        else:
            reconstructed = reconstructed_blocks.flatten()[:total_len]
            noisy = noisy_blocks.flatten()[:total_len]
            clean = clean_blocks.flatten()[:total_len]

        return {
            "reconstructed": reconstructed,
            "noisy": noisy,
            "clean": clean,
            "metadata": {"name": name},
        }

