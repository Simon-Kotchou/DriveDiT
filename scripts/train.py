#!/usr/bin/env python3
"""
Production training script for DriveDiT with full pipeline integration.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.pipeline import create_efficient_pipeline, MemoryMonitor
from training.self_forcing import SelfForcingTrainer, create_simple_model
from training.distributed import launch_distributed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DriveDiT Training')
    
    # Model arguments
    parser.add_argument('--model_dim', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--sequence_length', type=int, default=16, help='Video sequence length')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/videos', help='Data directory')
    parser.add_argument('--cache_dir', type=str, default='./data/cache', help='Cache directory')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    
    # System arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--max_memory_gb', type=float, default=12.0, help='Max GPU memory (GB)')
    
    # Distributed arguments
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--world_size', type=int, default=None, help='World size for distributed training')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port')
    
    # Validation arguments
    parser.add_argument('--val_frequency', type=int, default=5, help='Validation frequency (epochs)')
    parser.add_argument('--checkpoint_frequency', type=int, default=1000, help='Checkpoint frequency (steps)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--resume_best', action='store_true', help='Resume from best checkpoint')
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories."""
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)


def save_config(args, config_path: str):
    """Save training configuration."""
    config = vars(args)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_model(args) -> torch.nn.Module:
    """Create model based on arguments."""
    # For now, use simple model - replace with actual DriveDiT when available
    return create_simple_model(
        input_channels=3,
        hidden_dim=args.model_dim
    )


def create_dataloaders(args):
    """Create training and validation dataloaders."""
    try:
        train_loader, val_loader, _ = create_efficient_pipeline(
            video_dir=args.data_dir,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            image_size=(args.image_size, args.image_size),
            cache_dir=args.cache_dir,
            num_workers=args.num_workers
        )
        print(f"Loaded data from {args.data_dir}")
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Warning: Could not load real data ({e})")
        print("Using dummy data for testing")
        return create_dummy_dataloaders(args)


def create_dummy_dataloaders(args):
    """Create dummy dataloaders for testing."""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'frames': torch.randn(args.sequence_length, 3, args.image_size, args.image_size),
                'controls': torch.randn(args.sequence_length, 6)
            }
    
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, args):
    """Create optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * 1000,  # Approximate steps per epoch
        eta_min=args.learning_rate * 0.01
    )
    
    return optimizer, scheduler


def train_worker(rank: int, world_size: int, args):
    """Training worker function for distributed training."""
    
    # Setup distributed training
    if world_size > 1:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    try:
        # Create model
        model = create_model(args)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(model, args)
        
        # Create trainer
        trainer = SelfForcingTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=args.checkpoint_dir,
            max_memory_gb=args.max_memory_gb,
            mixed_precision=args.mixed_precision,
            distributed=(world_size > 1)
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume or args.resume_best:
            checkpoint_info = trainer.load_checkpoint(
                checkpoint_path=args.resume,
                load_best=args.resume_best
            )
            if checkpoint_info:
                start_epoch = checkpoint_info.get('epoch', 0)
                if rank == 0:
                    print(f"Resumed from epoch {start_epoch}")
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(args)
        
        # Training loop
        if rank == 0:
            print(f"Starting training for {args.num_epochs} epochs")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        for epoch in range(start_epoch, args.num_epochs):
            if rank == 0:
                print(f"\\nEpoch {epoch + 1}/{args.num_epochs}")
            
            # Train epoch
            train_losses = trainer.train_epoch(train_loader)
            
            if rank == 0:
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
                print(f"Train | {loss_str}")
            
            # Validation
            if (epoch + 1) % args.val_frequency == 0:
                val_losses = trainer.validate(val_loader)
                if rank == 0:
                    loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
                    print(f"Val   | {loss_str}")
            
            # Print training summary
            if (epoch + 1) % 10 == 0 and rank == 0:
                trainer.print_training_summary()
        
        if rank == 0:
            print("Training completed!")
            trainer.print_training_summary()
    
    finally:
        if hasattr(trainer, 'cleanup'):
            trainer.cleanup()
        
        if world_size > 1:
            torch.distributed.destroy_process_group()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup directories and save config
    setup_directories(args)
    config_path = Path(args.log_dir) / 'config.json'
    save_config(args, config_path)
    
    # Determine world size
    if args.world_size is None:
        args.world_size = torch.cuda.device_count() if args.distributed else 1
    
    print("DriveDiT Training Configuration:")
    print("=" * 50)
    for key, value in vars(args).items():
        print(f"{key:20}: {value}")
    print("=" * 50)
    
    # Launch training
    if args.distributed and args.world_size > 1:
        print(f"Launching distributed training on {args.world_size} GPUs")
        mp.spawn(
            train_worker,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        print("Launching single GPU training")
        train_worker(0, 1, args)


if __name__ == "__main__":
    main()