import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


def spectrogram_db(sig, sample_rate, n_fft=1024, hop_length=256, eps=1e-10):
    f, t, Zxx = stft(
        sig,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    S = np.abs(Zxx)
    S_db = 20 * np.log10(S + eps)   # amplitude -> dB
    return f, t, S_db


def save_spectrograms(noisy, clean, reconstructed, sample_rate, save_path, figsize=(16, 8), n_fft=512, hop_length=128):
    """
    Function to draw and save spectrograms

    Args:
        noisy (np.ndarray or torch.Tensor): Noisy speech waveform (1D or 2D)
        clean (np.ndarray or torch.Tensor): Clean speech waveform (1D or 2D)
        reconstructed (np.ndarray or torch.Tensor): Enhanced speech waveform (1D or 2D)
        sample_rate (int): Sampling rate
        save_path (str): Output file path
        figsize (tuple): Figure size
        n_fft (int): FFT size
        hop_length (int): Hop length
    """
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().squeeze().numpy()
        return np.array(x).squeeze()

    noisy = to_numpy(noisy)
    clean = to_numpy(clean)
    reconstructed = to_numpy(reconstructed)

    f, t, S_noisy = spectrogram_db(noisy, sample_rate)
    _, _, S_clean = spectrogram_db(clean, sample_rate)
    _, _, S_recon = spectrogram_db(reconstructed, sample_rate)

    # Use shared limits so colors mean the same thing in every panel
    vmax = max(S_noisy.max(), S_clean.max(), S_recon.max())
    vmin = vmax - 80   # show 80 dB range

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True, sharey=True)

    im0 = axes[0].pcolormesh(t, f, S_noisy, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title("Noisy")

    axes[1].pcolormesh(t, f, S_clean, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("Clean")

    axes[2].pcolormesh(t, f, S_recon, shading="gouraud", cmap="magma", vmin=vmin, vmax=vmax)
    axes[2].set_title("Reconstructed")

    # Difference plot
    diff = np.abs(S_recon - S_clean)
    axes[3].pcolormesh(t, f, diff, shading="gouraud", cmap="magma", vmin=0, vmax=np.max(diff))
    axes[3].set_title("Reconstructed - Clean (dB difference)")

    for ax in axes:
        ax.set_ylabel("Frequency [Hz]")
    axes[-1].set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()