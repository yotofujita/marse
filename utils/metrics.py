"""Evaluation metrics utilities."""

import numpy as np
import torch
from pystoi import stoi


def compute_sisdr(estimate, reference):
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    
    Args:
        estimate: Estimated signal (numpy array or torch.Tensor)
        reference: Reference signal (numpy array or torch.Tensor)
    Returns:
        SI-SDR value (dB)
    """
    if isinstance(estimate, torch.Tensor):
        estimate = estimate.detach().cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().cpu().numpy()
    
    eps = np.finfo(estimate.dtype).eps
    alpha = (np.sum(estimate * reference) + eps) / (np.sum(np.abs(reference) ** 2) + eps)
    sisdr = 10 * np.log10(
        (np.sum(np.abs(alpha * reference) ** 2) + eps) /
        (np.sum(np.abs(alpha * reference - estimate) ** 2) + eps)
    )
    return sisdr


def compute_stoi(estimate, reference, sample_rate: int = 16000, extended: bool = True):
    """
    Compute Short-Time Objective Intelligibility (STOI)
    
    Args:
        estimate: Estimated signal (numpy array or torch.Tensor)
        reference: Reference signal (numpy array or torch.Tensor)
        sample_rate: Sampling rate
        extended: Whether to use Extended STOI
    Returns:
        STOI value
    """
    if isinstance(estimate, torch.Tensor):
        estimate = estimate.detach().cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().cpu().numpy()
    
    return stoi(reference, estimate, sample_rate, extended=extended)


def par_count(model):
    """
    Calculate the number of model parameters
    
    Args:
        model: PyTorch model
    Returns:
        Number of parameters
    """
    parcount = 0
    for p in model.parameters():
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        parcount += nn
    return parcount

