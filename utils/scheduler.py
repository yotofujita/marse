"""Learning rate scheduler utilities."""

import numpy as np


def cosine_decay(epoch, warmup_epochs, total_epochs):
    """
    Learning rate scheduler function with cosine decay and warmup
    
    Args:
        epoch: Current epoch
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
    Returns:
        Learning rate scale factor
    """
    if epoch < warmup_epochs:
        return min((epoch + 1) / warmup_epochs, 1.0)  # Warmup phase (Linear)
    else:
        cosine_epoch = epoch - warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * cosine_epoch / (total_epochs - warmup_epochs)))

