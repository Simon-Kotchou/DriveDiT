"""
Distributed training utilities for DriveDiT.
Zero-dependency implementation using only PyTorch's native distributed capabilities.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import os
import socket
from contextlib import contextmanager
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DistributedManager:
    """Manages distributed training setup and teardown."""
    
    def __init__(self):
        self.is_initialized = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_master = True
    
    def setup(self, rank: int, world_size: int, backend: str = 'nccl') -> None:
        """Initialize distributed training."""
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count()
        self.is_master = rank == 0
        self.is_initialized = True
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        
        logger.info(f"Distributed training initialized: rank {rank}/{world_size}")
    
    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce operation across processes."""
        if self.is_initialized and self.world_size > 1:
            dist.all_reduce(tensor, op)
        return tensor
    
    def gather(self, tensor: torch.Tensor) -> Optional[list]:
        """Gather tensors from all processes to master."""
        if not self.is_initialized or self.world_size == 1:
            return [tensor] if self.is_master else None
        
        gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)] if self.is_master else None
        dist.gather(tensor, gather_list, dst=0)
        return gather_list if self.is_master else None


class MemoryMonitor:
    """Monitor and manage GPU memory usage during training."""
    
    def __init__(self, max_memory_gb: float = 12.0):
        self.max_memory_gb = max_memory_gb
        self.peak_memory = 0.0
        self.allocated_memory = 0.0
        self.cleanup_threshold = max_memory_gb * 0.8  # Trigger cleanup at 80%
    
    def update(self) -> Dict[str, float]:
        """Update memory statistics."""
        if torch.cuda.is_available():
            self.allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.peak_memory = max(self.peak_memory, self.allocated_memory)
            
            return {
                'allocated_gb': self.allocated_memory,
                'peak_gb': self.peak_memory,
                'usage_percent': (self.allocated_memory / self.max_memory_gb) * 100
            }
        return {'allocated_gb': 0, 'peak_gb': 0, 'usage_percent': 0}
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        return self.allocated_memory > self.cleanup_threshold
    
    def cleanup(self) -> float:
        """Force memory cleanup and return freed memory."""
        if torch.cuda.is_available():
            before = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            after = torch.cuda.memory_allocated() / 1024**3
            freed = before - after
            self.allocated_memory = after
            return freed
        return 0.0
    
    def reset_peak(self) -> None:
        """Reset peak memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0.0


class CheckpointManager:
    """Manage model checkpointing with automatic cleanup."""
    
    def __init__(self, checkpoint_dir: str, keep_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.keep_checkpoints = keep_checkpoints
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        step: int,
        loss: float,
        metrics: Dict[str, float] = None,
        is_best: bool = False
    ) -> str:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics or {}
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """Load training checkpoint."""
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
            else:
                checkpoint_path = self._get_latest_checkpoint()
        
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return {}
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        
        return {
            'epoch': checkpoint['epoch'],
            'step': checkpoint['step'],
            'loss': checkpoint['loss'],
            'metrics': checkpoint.get('metrics', {})
        }
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoints[-1])
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) <= self.keep_checkpoints:
            return
        
        # Sort by step number and remove oldest
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        for checkpoint in checkpoints[:-self.keep_checkpoints]:
            checkpoint.unlink()


@contextmanager
def distributed_context(rank: int, world_size: int, backend: str = 'nccl'):
    """Context manager for distributed training."""
    dist_manager = DistributedManager()
    try:
        dist_manager.setup(rank, world_size, backend)
        yield dist_manager
    finally:
        dist_manager.cleanup()


def setup_distributed_model(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False
) -> torch.nn.Module:
    """Wrap model for distributed training."""
    model = model.to(device_id)
    return DDP(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters
    )


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """Create dataloader with distributed sampler."""
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def reduce_dict(input_dict: Dict[str, torch.Tensor], world_size: int) -> Dict[str, float]:
    """Reduce dictionary of tensors across all processes."""
    if world_size == 1:
        return {k: v.item() if torch.is_tensor(v) else v for k, v in input_dict.items()}
    
    reduced_dict = {}
    for key, value in input_dict.items():
        if torch.is_tensor(value):
            reduced_tensor = value.clone()
            dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
            reduced_dict[key] = (reduced_tensor / world_size).item()
        else:
            reduced_dict[key] = value
    
    return reduced_dict


def get_available_port() -> int:
    """Get an available port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def launch_distributed(
    train_func,
    world_size: int,
    backend: str = 'nccl',
    master_addr: str = 'localhost',
    master_port: Optional[int] = None
):
    """Launch distributed training across multiple processes."""
    if master_port is None:
        master_port = get_available_port()
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    mp.spawn(
        train_func,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


# Example usage for distributed training
def example_distributed_training(rank: int, world_size: int):
    """Example of how to use distributed training utilities."""
    with distributed_context(rank, world_size) as dist_manager:
        # Setup model
        model = MyModel()
        model = setup_distributed_model(model, dist_manager.local_rank)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dataset and dataloader
        dataset = MyDataset()
        dataloader = create_distributed_dataloader(dataset, batch_size=32)
        
        # Training loop
        for epoch in range(num_epochs):
            # Set epoch for distributed sampler
            dataloader.sampler.set_epoch(epoch)
            
            for batch in dataloader:
                # Training step
                loss = training_step(model, batch)
                
                # Reduce loss across processes
                reduced_losses = reduce_dict({'loss': loss}, world_size)
                
                if dist_manager.is_master:
                    print(f"Epoch {epoch}, Loss: {reduced_losses['loss']:.4f}")


if __name__ == "__main__":
    # Example: launch 4-GPU distributed training
    world_size = torch.cuda.device_count()
    if world_size > 1:
        launch_distributed(example_distributed_training, world_size)
    else:
        example_distributed_training(0, 1)