#!/usr/bin/env python3
"""Training script using Hydra configuration."""

import os
import time
import random
import torch
import torch.multiprocessing as mp
import numpy as np
import tqdm
import soundfile as sf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import hydra
from omegaconf import DictConfig
from torch.utils.data import Subset

from torch.amp import autocast

from utils.distributed import ddp_setup
from utils.metrics import compute_sisdr, compute_stoi, par_count
from utils.scheduler import cosine_decay

from utils.visualization import save_spectrograms


COMET_API_KEY = '0RQBTe7VRVAteoC6t35f6MupZ'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def process_validation_batch(model, batch, use_bfloat16: bool):
    """Process a single validation batch"""
    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        if hasattr(model, 'module'):
            output = model.module.separate_batch(batch)
        else:
            output = model.separate_batch(batch)
    
    reconstructed = output['reconstructed']
    noisy = output['noisy']
    clean = output['clean']
    name = output['metadata']['name']
    
    if reconstructed.dim() == 3 and reconstructed.shape[1] == 1:
        reconstructed = reconstructed.squeeze(1)
    
    reconstructed_np = reconstructed.detach().float().cpu().numpy().squeeze()
    noisy_np = noisy.detach().float().cpu().numpy().squeeze()
    clean_np = clean.detach().float().cpu().numpy().squeeze()
    
    return reconstructed_np, noisy_np, clean_np, name


def save_validation_samples(reconstructed_j, noisy_j, clean_j, name_j, epoch, audio_output_dir, sample_rate):
    """Save validation audio samples and spectrograms"""
    os.makedirs(f"{audio_output_dir}/{name_j}/{epoch + 1}ep", exist_ok=True)
    sf.write(f"{audio_output_dir}/{name_j}/{epoch + 1}ep/enhanced.wav", reconstructed_j, sample_rate)
    sf.write(f"{audio_output_dir}/{name_j}/{epoch + 1}ep/noisy.wav", noisy_j, sample_rate)
    sf.write(f"{audio_output_dir}/{name_j}/{epoch + 1}ep/clean.wav", clean_j, sample_rate)
    save_spectrograms(noisy_j, clean_j, reconstructed_j, sample_rate, f"{audio_output_dir}/{name_j}/{epoch + 1}ep/spectrogram.png")


def validate(model, val_loader, epoch, exp, sample_rate=16000, audio_output_dir="output_audio", use_bfloat16=False, device="cuda"):
    """
    Execute validation
    
    Args:
        model: Source separation model
        val_loader: Validation data loader
        epoch: Epoch number
        sample_rate: Sampling rate
        audio_output_dir: Audio output directory
        use_bfloat16: Whether to use bfloat16 mixed precision
        device: Device
    """
    model.eval()
    os.makedirs(audio_output_dir, exist_ok=True)
    
    SI_SDR_list = []
    ESTOI_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")):
            reconstructed_np, noisy_np, clean_np, name = process_validation_batch(model, batch, use_bfloat16)

            for j, (reconstructed_j, noisy_j, clean_j, name_j) in enumerate(zip(reconstructed_np, noisy_np, clean_np, name)):
                try:
                    sisdr = compute_sisdr(reconstructed_j, clean_j)
                    estoi = compute_stoi(reconstructed_j, clean_j, sample_rate)
                    SI_SDR_list.append(sisdr)
                    ESTOI_list.append(estoi)

                    # Save samples from first batch
                    if i == 0 and j < 10:
                        save_validation_samples(
                            reconstructed_j, noisy_j, clean_j, name_j,
                            epoch, audio_output_dir, sample_rate
                        )
                except Exception as e:
                    print(f"STOI computation failed for batch {i}, sample {j}: {e}")
                    break
    
    mean_sisdr = np.mean(SI_SDR_list)
    mean_estoi = np.mean(ESTOI_list)
    
    print(f"  SI-SDR: {mean_sisdr:.4f} for Epoch {epoch + 1}")
    print(f"  ESTOI: {mean_estoi:.4f} for Epoch {epoch + 1}")

    exp.log_metrics({
        'Validation SI-SDR': mean_sisdr,
        'Validation ESTOI': mean_estoi
    }, step=epoch + 1)
    
    return mean_sisdr, mean_estoi


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, model_params=None, name=None, scheduler=None):
    """Save checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if name is not None:
        path = f"{checkpoint_dir}/{name}.pt"
    else:
        path = f"{checkpoint_dir}/{epoch + 1}ep.pt"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if model_params:
        checkpoint['model_params'] = model_params
    
    torch.save(checkpoint, path)
    print(f"Epoch {epoch + 1} | Checkpoint saved at {path}")


def setup_device(rank: int, world_size: int, cfg: DictConfig, is_distributed: bool):
    """Setup device for training"""
    if is_distributed:
        ddp_setup(rank, world_size)
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda")
    return device


def setup_experiment_logging(cfg: DictConfig, rank: int):
    """Setup experiment logging (Comet.ml or dummy)"""
    if cfg.comet_logging and rank == 0:
        import comet_ml
        import datetime
        exp = comet_ml.Experiment(api_key=COMET_API_KEY, project_name=cfg.project_name)
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        exp.set_name(f"{cfg.dataset_name}__{cfg.model_name}__{now}")
    else:
        class dummy_exp:
            def log_parameters(self, params):
                pass
            def log_metrics(self, metrics, step):
                pass
            def end(self):
                pass
        exp = dummy_exp()
    return exp


def create_dataloaders(cfg: DictConfig, world_size: int, g: torch.Generator, is_distributed: bool):
    """Create training and validation dataloaders"""
    val_dataset = hydra.utils.instantiate(cfg.val_dataset)
    if cfg.mini:
        mini_num_samples = cfg.train.get("mini_num_samples", None)
        if mini_num_samples is None:
            mini_num_samples = cfg.train.batch_size * 8
        val_dataset = Subset(val_dataset, list(np.arange(mini_num_samples)))
        train_dataset = val_dataset
    else:
        train_dataset = hydra.utils.instantiate(cfg.train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True if not is_distributed else False,
        num_workers=cfg.train.num_workers,
        worker_init_fn=seed_worker if cfg.train.num_workers > 0 else None, generator=g,
        sampler=DistributedSampler(train_dataset) if is_distributed else None,
        pin_memory=True,
        persistent_workers=cfg.train.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        sampler=DistributedSampler(val_dataset) if is_distributed else None,
        pin_memory=True,
        persistent_workers=cfg.train.num_workers > 0
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def setup_model(cfg: DictConfig, device: torch.device, rank: int, is_distributed: bool):
    """Setup and configure model"""
    model = hydra.utils.instantiate(cfg.model, device=device)
    model = model.to(device)
    
    if cfg.train.get('channels_last', False):
        model = model.to(memory_format=torch.channels_last)
    
    if hasattr(torch, 'compile') and cfg.train.get('compile_model', False):
        if rank == 0:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')
    
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    return model


def create_scheduler(optimizer, train_loader, gradient_accumulation_steps, cfg: DictConfig, rank: int):
    """Create learning rate scheduler with step-based scheduling"""
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    if len(train_loader) % gradient_accumulation_steps != 0:
        steps_per_epoch += 1
    
    total_steps = steps_per_epoch * cfg.train.num_epochs
    warmup_steps = steps_per_epoch * cfg.train.warmup_epochs
    
    if rank == 0:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
    
    def lr_lambda(step):
        """Lambda function for learning rate scheduling
        step: last_epoch from LambdaLR (starts at -1, becomes 0 after first step)
        """
        # Convert to 0-indexed step count
        current_step = step + 1
        
        if current_step <= warmup_steps:
            # Linear warmup from 0 to 1
            return current_step / warmup_steps if warmup_steps > 0 else 1.0
        else:
            # Cosine decay from 1 to 0.5
            cosine_step = current_step - warmup_steps
            cosine_total = total_steps - warmup_steps
            if cosine_total > 0:
                return 0.5 * (1 + np.cos(np.pi * cosine_step / cosine_total))
            else:
                return 0.5
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, steps_per_epoch


def load_checkpoint(
    model, optimizer, scheduler, cfg: DictConfig, device: torch.device,
    steps_per_epoch: int, rank: int
):
    """Load checkpoint if specified"""
    start_epoch = 0
    if cfg.train.checkpoint_path is not None:
        checkpoint = torch.load(cfg.train.checkpoint_path, map_location=device, weights_only=False)
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if rank == 0:
                print(f"Restored scheduler state from checkpoint (last_epoch: {scheduler.last_epoch})")
        else:
            steps_to_advance = start_epoch * steps_per_epoch
            for _ in range(steps_to_advance):
                scheduler.step()
            if rank == 0:
                print(f"Scheduler state not found in checkpoint. Advanced scheduler by {steps_to_advance} steps (epoch {start_epoch})")
    
    return start_epoch


def clip_gradients(model, grad_clip_norm: float, rank: int):
    """Clip gradients if specified"""
    if grad_clip_norm is not None:
        if hasattr(model, 'module'):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        
        if rank == 0 and grad_norm > grad_clip_norm:
            print(f"Warning: Gradient norm clipped: {grad_norm:.4f}")
        return grad_norm
    return None


def handle_remaining_gradients(
    model, optimizer, scheduler, train_loader, gradient_accumulation_steps,
    cfg: DictConfig, rank: int
):
    """Handle remaining gradients if dataset size is not divisible by accumulation steps"""
    if len(train_loader) % gradient_accumulation_steps != 0:
        clip_gradients(model, cfg.train.grad_clip_norm, rank)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


def train_epoch(
    model, train_loader, optimizer, scheduler, gradient_accumulation_steps,
    cfg: DictConfig, use_bfloat16: bool, rank: int
):
    """Train for one epoch"""
    model.train()
    # Store tensors instead of scalars to avoid GPU-CPU synchronization
    train_losses_tensor = []
    loss_components_tensor = {}
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch")):
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            loss, loss_dict = model(batch)
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Store tensor (detached) instead of calling .item() immediately
        # This avoids GPU-CPU synchronization during training
        train_losses_tensor.append((loss.detach() * gradient_accumulation_steps))
        
        # Store loss component tensors (detached)
        for key, value in loss_dict.items():
            if loss_components_tensor.get(key) is None:
                loss_components_tensor[key] = []
            loss_components_tensor[key].append(value.detach())
        
        # Update weights every N accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            clip_gradients(model, cfg.train.grad_clip_norm, rank)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    
    # Handle remaining gradients
    handle_remaining_gradients(
        model, optimizer, scheduler, train_loader,
        gradient_accumulation_steps, cfg, rank
    )
    
    # Convert all tensors to CPU/float in one batch operation (much faster)
    if train_losses_tensor:
        train_losses = torch.stack(train_losses_tensor).cpu().float().numpy().tolist()
    else:
        train_losses = []
    
    loss_components = {}
    for key, tensor_list in loss_components_tensor.items():
        if tensor_list:
            loss_components[key] = torch.stack(tensor_list).cpu().float().numpy().tolist()
        else:
            loss_components[key] = []
    
    return train_losses, loss_components


def log_training_metrics(exp, train_losses, loss_components, epoch: int, rank: int, scheduler: torch.optim.lr_scheduler.LambdaLR):
    """Log training metrics"""
    print(f"Average Training Loss: {np.mean(train_losses):.4f}")
    exp.log_metrics({'Training Loss': np.mean(train_losses)}, step=epoch + 1)
    for key, value in loss_components.items():
        print(f"{key}: {np.mean(value)}")
        exp.log_metrics({f"Training {key}": np.mean(value)}, step=epoch + 1)
    exp.log_metrics({'Learning rate': scheduler.get_last_lr()[0]}, step=epoch + 1)


def append_train_log(message: str, log_path: str, rank: int):
    if rank != 0:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def main_worker(rank: int, world_size: int, cfg: DictConfig, g: torch.Generator, is_distributed: bool):
    """DDP worker function - main training orchestration"""
    # Setup device and distributed training
    device = setup_device(rank, world_size, cfg, is_distributed)
    
    # Setup experiment logging
    exp = setup_experiment_logging(cfg, rank)
    log_path = os.path.join(os.getcwd(), "train.log")
    if rank == 0:
        append_train_log(f"Training start: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_path, rank)
    
    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(cfg, world_size, g, is_distributed)
    
    # Setup model
    model = setup_model(cfg, device, rank, is_distributed)
    
    # Get gradient accumulation configuration
    gradient_accumulation_steps = cfg.train.get('gradient_accumulation_steps', 1)
    effective_batch_size = cfg.train.batch_size * world_size * gradient_accumulation_steps
    
    if rank == 0:
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size} (per-GPU: {cfg.train.batch_size}, GPUs: {world_size}, accumulation: {gradient_accumulation_steps})")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate_base,
        betas=(0.9, 0.95),
        weight_decay=cfg.train.weight_decay
    )
    
    # Create scheduler
    scheduler, steps_per_epoch = create_scheduler(
        optimizer, train_loader, gradient_accumulation_steps, cfg, rank
    )
    
    # Load checkpoint if specified
    start_epoch = load_checkpoint(model, optimizer, scheduler, cfg, device, steps_per_epoch, rank)
    
    # Display model information
    if rank == 0:
        param_count = par_count(model.module if hasattr(model, 'module') else model)
        print(f"Number of Model Parameters: {param_count / 1e6:.2f}M")
        if cfg.train.get('use_bfloat16', False):
            print(f"Using bfloat16 mixed precision training")
    
    use_bfloat16 = cfg.train.get('use_bfloat16', False)
    
    # Training loop
    total_start = time.time()
    for epoch in range(start_epoch, cfg.train.num_epochs):
        if rank == 0:
            print(f"Epoch {epoch + 1}/{cfg.train.num_epochs}")
        epoch_start = time.time()
        
        # Set epoch for DistributedSampler to ensure proper shuffling
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_losses, loss_components = train_epoch(
            model, train_loader, optimizer, scheduler,
            gradient_accumulation_steps, cfg, use_bfloat16, rank
        )
        
        # Validation
        if (epoch + 1) % cfg.train.gen_every == 0:
            if rank == 0:
                validate(
                    model, val_loader, epoch, exp,
                    sample_rate=cfg.model.dac_cfg.sample_rate,
                    audio_output_dir=cfg.paths.audio_output_dir,
                    use_bfloat16=use_bfloat16,
                    device=device,
                )
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, cfg.paths.checkpoint_dir,
                scheduler=scheduler
            )
        
        # Log training metrics
        log_training_metrics(exp, train_losses, loss_components, epoch, rank, scheduler)
        epoch_elapsed = time.time() - epoch_start
        append_train_log(f"Epoch {epoch + 1} time: {epoch_elapsed:.2f} sec", log_path, rank)
    
    if is_distributed:
        destroy_process_group()


@hydra.main(config_path="configs", config_name="mgse")
def main(cfg: DictConfig):
    """Main function"""
    # Set seed for reproducibility
    seed = cfg.train.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Check if running via torchrun (environment variables are set)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if rank == 0:
            print("Running via torchrun")
        main_worker(rank, world_size, cfg, g, is_distributed=True)
    else:
        print("Running via single GPU")
        # Single GPU training
        main_worker(0, 1, cfg, g, is_distributed=False)

if __name__ == "__main__":
    main()
