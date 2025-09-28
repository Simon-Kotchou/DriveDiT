# drivedit/training/self_forcing.py
"""
Self-Forcing training implementation for DriveDiT.
Based on "Self-Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os
import gc
from collections import defaultdict
from pathlib import Path
import json
import time

try:
    from .distributed import (
        DistributedManager, MemoryMonitor, CheckpointManager,
        setup_distributed_model, reduce_dict
    )
except ImportError:
    from distributed import (
        DistributedManager, MemoryMonitor, CheckpointManager,
        setup_distributed_model, reduce_dict
    )

class SelfForcingTrainer:
    """
    Self-forcing trainer with distributed training, memory optimization,
    and efficient checkpointing for large-scale video generation.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_dir: str = "./checkpoints",
        max_memory_gb: float = 12.0,
        mixed_precision: bool = True,
        distributed: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Training setup
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_memory_gb = max_memory_gb
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        
        # Memory management
        self.memory_monitor = MemoryMonitor(max_memory_gb)
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Distributed setup
        self.dist_manager = DistributedManager()
        if distributed:
            self.model = setup_distributed_model(
                model, 
                self.dist_manager.local_rank,
                find_unused_parameters=True
            )
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, keep_checkpoints=5
        )
        
        # Loss weights optimized for self-forcing
        self.loss_weights = {
            'reconstruction': 1.0,
            'temporal_consistency': 0.3,
            'flow_matching': 0.5,
            'jepa_contrastive': 0.2
        }
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Performance tracking
        self.step_times = []
        self.throughput_tracker = defaultdict(list)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with memory optimization and distributed support.
        """
        self.model.train()
        epoch_losses = defaultdict(list)
        
        # Set distributed sampler epoch
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(self.epoch)
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Epoch {self.epoch}",
            disable=not self.dist_manager.is_master
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start = time.time()
            
            # Move batch to device
            batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Memory check and cleanup
            if self.memory_monitor.should_cleanup():
                freed = self.memory_monitor.cleanup()
                if self.dist_manager.is_master:
                    print(f"Memory cleanup: freed {freed:.2f}GB")
            
            # Training step with mixed precision
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                losses = self._self_forcing_step(batch)
                total_loss = self._compute_total_loss(losses)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Reduce losses across processes
            if self.dist_manager.world_size > 1:
                reduced_losses = reduce_dict(losses, self.dist_manager.world_size)
                reduced_total_loss = total_loss.item() / self.dist_manager.world_size
            else:
                reduced_losses = {k: v.item() if torch.is_tensor(v) else v 
                                for k, v in losses.items()}
                reduced_total_loss = total_loss.item()
            
            # Logging and tracking
            if self.dist_manager.is_master:
                for key, value in reduced_losses.items():
                    epoch_losses[key].append(value)
                
                # Performance tracking
                step_time = time.time() - step_start
                self.step_times.append(step_time)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{reduced_total_loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    'mem': f"{self.memory_monitor.update()['usage_percent']:.1f}%",
                    'time': f"{step_time:.2f}s"
                })
            
            # Checkpointing
            if (self.global_step + 1) % self.checkpoint_frequency == 0:
                self._save_checkpoint(reduced_total_loss)
            
            self.global_step += 1
        
        self.epoch += 1
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def _self_forcing_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute optimized self-forcing training step with memory efficiency.
        """
        frames = batch['frames']  # [B, T, C, H, W]
        B, T, _, H, W = frames.shape
        
        # Split sequence: context + prediction horizon
        context_length = min(8, T // 2)  # Adaptive context length
        prediction_horizon = min(16, T - context_length)
        
        context_frames = frames[:, :context_length]
        target_frames = frames[:, context_length:context_length + prediction_horizon]
        
        # Simple control simulation if not available
        if 'controls' in batch:
            controls = batch['controls']
        else:
            # Generate dummy controls for training
            controls = torch.zeros(B, T, 6, device=frames.device)
        
        losses = {}
        
        # Simplified autoregressive generation without complex memory systems
        # Focus on core self-forcing principle
        
        # Encode context frames
        with torch.no_grad():
            # Use simple frame encoding for demonstration
            context_features = self._encode_frames(context_frames)
        
        # Simplified autoregressive generation for core self-forcing
        generated_frames = []
        current_input = context_frames[:, -1]  # Last context frame
        
        for t in range(prediction_horizon):
            # Predict next frame using current input (self-forcing principle)
            next_frame = self._predict_next_frame(current_input)
            generated_frames.append(next_frame)
            
            # CRITICAL: Use predicted frame as input for next step
            current_input = next_frame
        
        # Compute losses
        if generated_frames:
            pred_frames = torch.stack(generated_frames, dim=1)  # [B, T, C, H, W]
            
            # 1. Reconstruction loss
            losses['reconstruction'] = F.mse_loss(pred_frames, target_frames)
            
            # 2. Temporal consistency loss
            if pred_frames.shape[1] > 1:
                pred_diffs = pred_frames[:, 1:] - pred_frames[:, :-1]
                target_diffs = target_frames[:, 1:] - target_frames[:, :-1]
                losses['temporal_consistency'] = F.mse_loss(pred_diffs, target_diffs)
            
            # 3. Perceptual loss (simplified)
            losses['perceptual'] = self._compute_perceptual_loss(pred_frames, target_frames)
        
        return losses
    
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Simple frame encoding for demonstration."""
        # Placeholder: replace with actual VAE encoder
        return F.avg_pool2d(frames.flatten(0, 1), 8).view(
            frames.shape[0], frames.shape[1], -1
        )
    
    def _predict_next_frame(self, current_frame: torch.Tensor) -> torch.Tensor:
        """Predict next frame using model."""
        # Placeholder: replace with actual model forward pass
        # For now, add small noise to simulate prediction
        noise = torch.randn_like(current_frame) * 0.01
        return torch.clamp(current_frame + noise, 0, 1)
    
    def _compute_perceptual_loss(
        self, 
        pred_frames: torch.Tensor, 
        target_frames: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss using VGG features."""
        # Simplified perceptual loss
        pred_gray = pred_frames.mean(dim=2, keepdim=True)
        target_gray = target_frames.mean(dim=2, keepdim=True)
        return F.l1_loss(pred_gray, target_gray)
    
    def _save_checkpoint(self, loss: float) -> None:
        """Save training checkpoint."""
        if not self.dist_manager.is_master:
            return
        
        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss
        
        self.checkpoint_manager.save_checkpoint(
            model=self.model.module if hasattr(self.model, 'module') else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=loss,
            is_best=is_best
        )
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load training checkpoint."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return self.checkpoint_manager.load_checkpoint(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=checkpoint_path
        )
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training performance statistics."""
        if not self.step_times:
            return {}
        
        memory_stats = self.memory_monitor.update()
        
        return {
            'avg_step_time': np.mean(self.step_times[-100:]),  # Last 100 steps
            'steps_per_sec': 1.0 / np.mean(self.step_times[-100:]),
            **memory_stats
        }
    
    def print_training_summary(self) -> None:
        """Print comprehensive training summary."""
        if not self.dist_manager.is_master:
            return
        
        stats = self.get_training_stats()
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Epoch: {self.epoch}")
        print(f"Global Step: {self.global_step}")
        print(f"Best Loss: {self.best_loss:.6f}")
        
        if stats:
            print(f"\nPerformance:")
            print(f"  Steps/sec: {stats.get('steps_per_sec', 0):.2f}")
            print(f"  Avg step time: {stats.get('avg_step_time', 0):.3f}s")
            print(f"  Memory usage: {stats.get('usage_percent', 0):.1f}%")
            print(f"  Peak memory: {stats.get('peak_gb', 0):.2f}GB")
        
        print("="*60 + "\n")
    
    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine all losses with appropriate weights."""
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        for loss_name, loss_value in losses.items():
            if loss_name in self.loss_weights:
                weight = self.loss_weights[loss_name]
                total_loss = total_loss + weight * loss_value
        
        return total_loss
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics with simple print statements."""
        if not self.dist_manager.is_master:
            return
        
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {self.global_step} | {metric_str}")
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Simple validation with basic metrics."""
        self.model.eval()
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(
                val_dataloader, 
                desc="Validation",
                disable=not self.dist_manager.is_master
            ):
                batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Simple validation step
                losses = self._self_forcing_step(batch)
                
                # Reduce losses across processes
                if self.dist_manager.world_size > 1:
                    reduced_losses = reduce_dict(losses, self.dist_manager.world_size)
                else:
                    reduced_losses = {k: v.item() if torch.is_tensor(v) else v 
                                    for k, v in losses.items()}
                
                if self.dist_manager.is_master:
                    for key, value in reduced_losses.items():
                        val_losses[key].append(value)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in val_losses.items()}
        
        if self.dist_manager.is_master:
            self.log_metrics({f'val_{k}': v for k, v in avg_metrics.items()})
        
        return avg_metrics
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'dist_manager'):
            self.dist_manager.cleanup()


def create_simple_model(input_channels: int = 3, hidden_dim: int = 256) -> torch.nn.Module:
    """Create a simple model for demonstration."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_channels, hidden_dim // 4, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(hidden_dim // 2, input_channels, 3, padding=1),
        torch.nn.Sigmoid()
    )


def main(rank: int = 0, world_size: int = 1):
    """Main training function with distributed support."""
    
    # Setup distributed training if multi-GPU
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    try:
        # Create simple model for demonstration
        model = create_simple_model()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        
        # Create trainer
        trainer = SelfForcingTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir="./checkpoints",
            max_memory_gb=12.0,
            mixed_precision=True,
            distributed=(world_size > 1)
        )
        
        # Load data
        from ..data.pipeline import create_efficient_pipeline
        
        try:
            train_loader, val_loader, _ = create_efficient_pipeline(
                video_dir="./data/videos",
                batch_size=4,
                sequence_length=16,
                image_size=(256, 256),
                cache_dir="./data/cache",
                num_workers=4
            )
        except Exception as e:
            print(f"Warning: Could not load real data ({e}), using dummy data")
            train_loader = create_dummy_dataloader(batch_size=4)
            val_loader = create_dummy_dataloader(batch_size=4)
        
        # Training loop
        num_epochs = 100
        
        for epoch in range(num_epochs):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_losses = trainer.train_epoch(train_loader)
            
            if rank == 0:
                trainer.log_metrics(train_losses)
            
            # Validate
            if epoch % 5 == 0:
                val_losses = trainer.validate(val_loader)
                if rank == 0:
                    print(f"Validation: {val_losses}")
            
            # Print summary periodically
            if epoch % 10 == 0 and rank == 0:
                trainer.print_training_summary()
    
    finally:
        if hasattr(trainer, 'cleanup'):
            trainer.cleanup()
        
        if world_size > 1:
            torch.distributed.destroy_process_group()


def create_dummy_dataloader(batch_size: int = 4, num_batches: int = 100):
    """Create dummy dataloader for testing."""
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return num_batches
        
        def __getitem__(self, idx):
            return {
                'frames': torch.randn(16, 3, 256, 256),  # [T, C, H, W]
                'controls': torch.randn(16, 6)  # [T, control_dim]
            }
    
    return torch.utils.data.DataLoader(
        DummyDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DriveDiT Self-Forcing Training')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    args = parser.parse_args()
    
    if args.distributed and args.gpus > 1:
        # Multi-GPU distributed training
        import torch.multiprocessing as mp
        mp.spawn(main, args=(args.gpus,), nprocs=args.gpus, join=True)
    else:
        # Single GPU training
        main(0, 1)