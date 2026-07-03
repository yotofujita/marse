"""Distributed training utilities."""

import os
import torch
from torch.distributed import init_process_group


def ddp_setup(rank: int, world_size: int):
    """
    Setup for Distributed Data Parallel (DDP)
    
    Args:
        rank: Unique identifier for each process
        world_size: Total number of processes
    """
    # If torchrun is used, environment variables are already set
    if "RANK" in os.environ:
        # Use environment variables set by torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # torchrun sets MASTER_ADDR and MASTER_PORT automatically
        # init_process_group can infer rank and world_size from env vars
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        init_process_group(backend="nccl")
    else:
        # Manual setup - set environment variables
        torch.cuda.set_device(rank)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

