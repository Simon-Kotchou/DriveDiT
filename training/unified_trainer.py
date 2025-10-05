"""
Unified training pipeline integrating all methodologies:
- Self-forcing training with curriculum learning
- Flow matching and distillation  
- Modular components (control, JEPA, depth, memory)
- Large-scale distributed training capabilities

This replaces the scattered training approaches with a single, cohesive system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
import random
import time
import logging
from dataclasses import dataclass
from pathlib import Path

from .losses import UnifiedLoss
from .distributed import MemoryMonitor, CheckpointManager, DistributedManager
from ..config.config import DriveDiTConfig
from ..models.world_model import WorldModel


@dataclass
class TrainingConfig:
    """Unified training configuration."""
    # Basic training
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 1000
    
    # Self-forcing curriculum (integrated comma.ai insights)
    enable_curriculum: bool = True
    initial_sequence_length: int = 4
    final_sequence_length: int = 32
    curriculum_warmup_steps: int = 10000
    
    initial_self_forcing_ratio: float = 0.0
    final_self_forcing_ratio: float = 0.8
    self_forcing_warmup_steps: int = 5000
    
    # Flow matching
    enable_flow_matching: bool = True
    flow_matching_weight: float = 1.0
    num_flow_steps: int = 4
    
    # Training stability
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    checkpoint_every: int = 1000
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    
    # Memory management
    max_memory_gb: float = 16.0
    memory_cleanup_threshold: float = 0.8


class CurriculumScheduler:
    """Manages curriculum learning schedule."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.step = 0
    
    def update(self, step: int):
        """Update current training step."""
        self.step = step
    
    def get_sequence_length(self) -> int:
        """Get current sequence length."""
        if not self.config.enable_curriculum:
            return self.config.final_sequence_length
        
        progress = min(1.0, self.step / self.config.curriculum_warmup_steps)
        progress = self._smooth_schedule(progress)
        
        length = (
            self.config.initial_sequence_length + 
            progress * (self.config.final_sequence_length - self.config.initial_sequence_length)
        )
        return int(length)
    
    def get_self_forcing_ratio(self) -> float:
        """Get current self-forcing ratio."""
        progress = min(1.0, self.step / self.config.self_forcing_warmup_steps)
        progress = self._smooth_schedule(progress)
        
        ratio = (
            self.config.initial_self_forcing_ratio + 
            progress * (self.config.final_self_forcing_ratio - self.config.initial_self_forcing_ratio)
        )
        return ratio
    
    def _smooth_schedule(self, progress: float) -> float:
        """Apply smooth scheduling (cosine)."""
        return 0.5 * (1 - math.cos(math.pi * progress))


class UnifiedTrainer:
    """
    Unified trainer that combines all training methodologies:
    - Self-forcing with curriculum learning (comma.ai insights)
    - Flow matching and distillation
    - Distributed training
    - Memory management
    """
    
    def __init__(
        self,
        model: WorldModel,
        config: TrainingConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Training components
        self.curriculum = CurriculumScheduler(config)
        self.loss_fn = UnifiedLoss(config)
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        
        # Distributed training
        self.distributed_manager = DistributedManager() if config.distributed else None
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager("./checkpoints")
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'sequence_length': [],
            'self_forcing_ratio': [],
            'memory_usage': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        self.logger.info(f"Initialized UnifiedTrainer with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Update curriculum
            self.curriculum.update(self.global_step)
            
            # Adapt batch to current curriculum
            batch = self._adapt_batch_sequence_length(batch)
            
            # Training step
            losses = self._training_step(batch, optimizer)
            epoch_losses.append(losses['total'])
            
            # Logging
            if self.global_step % 100 == 0:
                self._log_metrics(losses)
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_every == 0:
                self._save_checkpoint(losses['total'], optimizer, scheduler)
            
            # Memory management
            if self.memory_monitor.should_cleanup():
                self._cleanup_memory()
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step()
            
            self.global_step += 1
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.epoch += 1
        
        return avg_loss
    
    def _training_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Single training step with unified methodology."""
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Get current curriculum parameters
        seq_len = self.curriculum.get_sequence_length()
        sf_ratio = self.curriculum.get_self_forcing_ratio()
        
        optimizer.zero_grad()
        
        if self.scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                predictions = self._forward_pass(batch, sf_ratio)
                losses = self._compute_losses(predictions, batch, sf_ratio)
            
            self.scaler.scale(losses['total']).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard training
            predictions = self._forward_pass(batch, sf_ratio)
            losses = self._compute_losses(predictions, batch, sf_ratio)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
        
        # Update metrics
        self._update_metrics(losses, seq_len, sf_ratio)
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor], sf_ratio: float) -> Dict[str, torch.Tensor]:
        """Forward pass with self-forcing."""
        frames = batch['frames']  # [B, T, C, H, W]
        controls = batch.get('controls')  # [B, T, control_dim] (optional)
        depth = batch.get('depth')  # [B, T, 1, H, W] (optional)
        
        B, T = frames.shape[:2]
        
        # Use self-forcing mode for training
        outputs = self.model(
            frames=frames,
            controls=controls,
            depth=depth,
            mode="self_forcing",
            self_forcing_ratio=sf_ratio
        )
        
        return outputs
    
    def _compute_losses(
        self, 
        predictions: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor],
        sf_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses using unified loss function."""
        
        # Prepare targets
        targets = {
            'frames': batch['frames'],
            'controls': batch.get('controls'),
            'depth': batch.get('depth')
        }
        
        # Add flow matching targets if enabled
        if self.config.enable_flow_matching and 'predictions' in predictions:
            targets['flow_target'] = self._compute_flow_target(predictions, targets)
        
        # Compute losses
        losses = self.loss_fn(predictions, targets, self.global_step)
        
        return losses
    
    def _compute_flow_target(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute flow matching target."""
        if 'predictions' in predictions and 'frames' in targets:
            pred_frames = predictions['predictions']
            target_frames = targets['frames']
            
            # Simple flow target: difference between prediction and target
            return target_frames - pred_frames
        
        return torch.zeros_like(predictions.get('predictions', targets['frames']))
    
    def _adapt_batch_sequence_length(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt batch to current curriculum sequence length."""
        target_length = self.curriculum.get_sequence_length()
        adapted_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                current_length = value.size(1)
                if current_length > target_length:
                    # Random crop
                    start_idx = random.randint(0, current_length - target_length)
                    adapted_batch[key] = value[:, start_idx:start_idx + target_length]
                elif current_length < target_length:
                    # Repeat last element
                    repeats = target_length - current_length
                    last_elements = value[:, -1:].repeat(1, repeats, *([1] * (value.dim() - 2)))
                    adapted_batch[key] = torch.cat([value, last_elements], dim=1)
                else:
                    adapted_batch[key] = value
            else:
                adapted_batch[key] = value
        
        return adapted_batch
    
    def _update_metrics(self, losses: Dict[str, torch.Tensor], seq_len: int, sf_ratio: float):
        """Update training metrics."""
        self.metrics['train_loss'].append(losses['total'].item() if torch.is_tensor(losses['total']) else losses['total'])
        self.metrics['sequence_length'].append(seq_len)
        self.metrics['self_forcing_ratio'].append(sf_ratio)
        
        # Memory usage
        memory_stats = self.memory_monitor.update()
        self.metrics['memory_usage'].append(memory_stats['usage_percent'])
        
        # Keep only recent history
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-500:]
    
    def _log_metrics(self, losses: Dict[str, float]):
        """Log current metrics."""
        seq_len = self.curriculum.get_sequence_length()
        sf_ratio = self.curriculum.get_self_forcing_ratio()
        memory_stats = self.memory_monitor.update()
        
        self.logger.info(
            f"Step {self.global_step}: Loss={losses['total']:.4f}, "
            f"SeqLen={seq_len}, SF_Ratio={sf_ratio:.3f}, "
            f"Memory={memory_stats['usage_percent']:.1f}%"
        )
        
        # Detailed loss breakdown
        if len(losses) > 1:
            loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses.items() if k != 'total'])
            self.logger.info(f"  Losses: {loss_str}")
    
    def _save_checkpoint(self, current_loss: float, optimizer: torch.optim.Optimizer, scheduler=None):
        """Save training checkpoint."""
        is_best = current_loss < self.best_loss
        if is_best:
            self.best_loss = current_loss
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=current_loss,
            metrics=self.get_current_metrics(),
            is_best=is_best
        )
        
        if is_best:
            self.logger.info(f"New best model saved: {checkpoint_path}")
    
    def _cleanup_memory(self):
        """Clean up memory when needed."""
        freed_gb = self.memory_monitor.cleanup()
        if freed_gb > 0:
            self.logger.info(f"Freed {freed_gb:.2f}GB of memory")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        metrics = {}
        
        for key, values in self.metrics.items():
            if values:
                metrics[f'avg_{key}'] = sum(values[-100:]) / len(values[-100:])  # Last 100 steps
                metrics[f'current_{key}'] = values[-1]
        
        metrics['global_step'] = self.global_step
        metrics['epoch'] = self.epoch
        metrics['best_loss'] = self.best_loss
        
        return metrics
    
    def load_checkpoint(self, checkpoint_path: str, optimizer: torch.optim.Optimizer, scheduler=None):
        """Load training checkpoint."""
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            self.model, optimizer, scheduler, checkpoint_path
        )
        
        if checkpoint_info:
            self.global_step = checkpoint_info.get('step', 0)
            self.epoch = checkpoint_info.get('epoch', 0)
            self.best_loss = checkpoint_info.get('loss', float('inf'))
            
            self.logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        
        return checkpoint_info
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                batch = self._adapt_batch_sequence_length(batch)
                
                predictions = self.model(
                    frames=batch['frames'],
                    controls=batch.get('controls'),
                    depth=batch.get('depth'),
                    mode="train"  # Use standard training mode for evaluation
                )
                
                losses = self._compute_losses(predictions, batch, 0.0)  # No self-forcing for eval
                total_loss += losses['total'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.model.train()
        
        return {'eval_loss': avg_loss}


def create_unified_trainer(
    model_config: DriveDiTConfig,
    training_config: TrainingConfig,
    device: str = "cuda"
) -> UnifiedTrainer:
    """Create unified trainer with world model."""
    
    # Create world model
    world_model = WorldModel(model_config)
    
    # Create trainer
    trainer = UnifiedTrainer(world_model, training_config, device)
    
    return trainer


if __name__ == "__main__":
    # Test unified trainer
    from ..config.config import get_research_config
    
    model_config = get_research_config()
    training_config = TrainingConfig(
        batch_size=2,
        max_steps=1000,
        enable_curriculum=True,
        enable_flow_matching=True
    )
    
    trainer = create_unified_trainer(model_config, training_config)
    
    # Test batch
    batch = {
        'frames': torch.randn(2, 8, 3, 64, 64),
        'controls': torch.randn(2, 8, 6) if model_config.control.enabled else None
    }
    
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=training_config.learning_rate)
    
    # Test training step
    losses = trainer._training_step(batch, optimizer)
    print(f"Training step completed: {losses}")
    
    # Test metrics
    metrics = trainer.get_current_metrics()
    print(f"Current metrics: {metrics}")
    
    print("Unified trainer test completed successfully!")