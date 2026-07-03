"""DAC (Descript Audio Codec) I/O representation implementation."""

import torch
from einops import rearrange
import dac


class DACRepresentation:
    """
    Implementation of I/O representation using DAC
    
    Encodes/decodes and tokenizes/detokenizes single-channel waveforms with DAC
    """
    
    def __init__(self, dac_model_type: str, n_quantizers: int, sample_rate: int = 16000, device: str = "cuda"):
        """
        Args:
            dac_model_type: DAC model type
            n_quantizers: Number of quantization levels
            sample_rate: Sampling rate
            device: Device
        """
        dac_model_path = dac.utils.download(model_type=dac_model_type)
        self.dac_model = dac.DAC.load(dac_model_path)
        self.dac_model.encoder.to(device)
        self.dac_model.quantizer.to(device)
        self.dac_model.decoder.to(device)
        self.dac_model.eval()
        
        self.n_quantizers = n_quantizers
        self.sample_rate = sample_rate
        self.device = device
        self._representation_dim = self.dac_model.encoder.enc_dim

    @torch.no_grad()
    def quantize(self, representation: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to single-channel waveform
        
        Args:
            representation: torch.Tensor, shape (batch, dim, T) or (dim, T)
        Returns:
            representation_q: torch.Tensor, shape (batch, dim, T) or (dim, T)
        """

        # Add batch dimension if not present
        if representation.dim() == 2:
            representation = representation.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        representation_q = self.dac_model.quantizer(representation, self.n_quantizers)[0]  # (batch, dim, T)
        
        if squeeze_batch:
            representation_q = representation_q.squeeze(0)
        
        return representation_q
    
    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode single-channel waveform to latent representation
        
        Args:
            waveform: torch.Tensor, shape (batch, n_channels, n_samples) or (n_channels, n_samples)
        Returns:
            representation: torch.Tensor, shape (batch, dim, T) or (dim, T)
        """

        assert waveform.shape[1] == 1, "DACRepresentation only supports single-channel input (n_channels=1)"

        # Add batch dimension if not present
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
    
        # DAC preprocess and encode
        preprocessed = self.dac_model.preprocess(waveform, self.sample_rate)
        representation = self.dac_model.encoder(preprocessed)  # (batch, dim, T)

        if squeeze_batch:
            representation = representation.squeeze(0)

        return representation
    
    @torch.no_grad()
    def decode(self, representation: torch.Tensor, quantized=False) -> torch.Tensor:
        """
        Decode latent representation to single-channel waveform
        
        Args:
            representation: torch.Tensor, shape (batch, dim, T) or (dim, T)
        Returns:
            waveform: torch.Tensor, shape (batch, n_channels, n_samples) or (n_channels, n_samples)
        """

        # Add batch dimension if not present
        if representation.dim() == 2:
            representation = representation.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        if not quantized:
            representation = self.dac_model.quantizer(representation, self.n_quantizers)[0]  # (batch, dim, T)
        decoded = self.dac_model.decode(representation)  # (batch, n_channels, n_samples)
        
        if squeeze_batch:
            decoded = decoded.squeeze(0)
        
        return decoded
    
    @torch.no_grad()
    def tokenize(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert single-channel waveform to tokens
        
        Args:
            waveform: torch.Tensor, shape (batch, n_channels, n_samples) or (n_channels, n_samples)
        Returns:
            tokens: torch.Tensor, shape (batch, Nq, T) or (Nq, T)
        """

        assert waveform.shape[1] == 1, "DACRepresentation only supports single-channel input (n_channels=1)"

        # Add batch dimension if not present
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False
    
        # DAC preprocess and encode
        preprocessed = self.dac_model.preprocess(waveform, self.sample_rate)
        tokens, _ = self.dac_model.encode(preprocessed, self.n_quantizers)  # (batch, Nq, T)
        
        if squeeze_batch:
            tokens = tokens.squeeze(0)
        
        return tokens
    
    @torch.no_grad()
    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens to latent representation (single-channel only)
        
        Args:
            tokens: torch.Tensor, shape (batch, Nq, T) or (Nq, T)
        Returns:
            decoded: torch.Tensor, shape (batch, n_channels, n_samples) or (n_channels, n_samples)
        """

        # Add batch dimension if not present
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        z = self.quantizer.from_codes(tokens)[0]
        decoded = self.dac_model.decode(z)  # (batch, n_channels, n_samples)
        
        if squeeze_batch:
            decoded = decoded.squeeze(0)
        
        return decoded
    
    def get_representation_dim(self) -> int:
        """Return representation dimension (per channel)"""
        return self._representation_dim
    
    def get_sample_rate(self) -> int:
        """Return sampling rate"""
        return self.sample_rate


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf
    import torchaudio
    dac_representation = DACRepresentation(dac_model_type="16khz", n_quantizers=12, sample_rate=16000, device="cuda")
    waveform, _ = torchaudio.load("waveform.wav")
    waveform = waveform.unsqueeze(0).to("cuda")

    # representation = dac_representation.encode(waveform)
    # print(representation.shape)
    # waveform_reproduced = dac_representation.decode(representation)
    # print(waveform_reproduced.shape)
    # sf.write("waveform_reproduced.wav", waveform_reproduced.detach().cpu().numpy().flatten(), dac_representation.get_sample_rate())

    tokens = dac_representation.tokenize(waveform)
    print(tokens.shape)
    waveform_reproduced = dac_representation.detokenize(tokens)
    print(waveform_reproduced.shape)
    sf.write("waveform_reproduced.wav", waveform_reproduced.detach().cpu().numpy().flatten(), dac_representation.get_sample_rate())

    waveform_cpu = waveform.detach().cpu().numpy().flatten()
    waveform_reproduced_cpu = waveform_reproduced.detach().cpu().numpy().flatten()
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.specgram(waveform_cpu, Fs=dac_representation.get_sample_rate(), NFFT=512, noverlap=256, cmap="magma")
    plt.title("Original Waveform Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Intensity [dB]")
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    plt.specgram(waveform_reproduced_cpu, Fs=dac_representation.get_sample_rate(), NFFT=512, noverlap=256, cmap="magma")
    plt.title("Waveform Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Intensity [dB]")
    plt.tight_layout()
    plt.savefig("waveform_spectrogram.png")