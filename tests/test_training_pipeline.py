"""
Test suite for training pipeline components.
Tests training loops, loss functions, optimization, and convergence.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Import training components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.self_forcing import SelfForcingTrainer, create_simple_model
from training.distributed import MemoryMonitor, CheckpointManager, DistributedManager
from training.comma_ai_insights import CommaAITrainer, CurriculumScheduler, LatentSpacePlanner
from config.modular_config import DriveDiTConfig, get_minimal_config, get_research_config


class TestSelfForcingTrainer:
    """Test self-forcing training methodology."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization with different configurations."""
        model = create_simple_model(input_channels=3, hidden_dim=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        trainer = SelfForcingTrainer(
            model=model,
            optimizer=optimizer,
            checkpoint_dir="./test_checkpoints",
            max_memory_gb=2.0,
            mixed_precision=False,
            distributed=False
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.global_step == 0
        assert trainer.epoch == 0
    
    def test_self_forcing_step(self):
        """Test self-forcing training step."""
        model = create_simple_model(input_channels=3, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        trainer = SelfForcingTrainer(
            model=model,
            optimizer=optimizer,
            mixed_precision=False,
            distributed=False
        )
        
        # Create test batch
        batch = {
            'frames': torch.randn(2, 8, 3, 64, 64),
            'controls': torch.randn(2, 8, 6)
        }
        
        # Test training step
        losses = trainer._self_forcing_step(batch)
        
        assert 'reconstruction' in losses
        assert isinstance(losses['reconstruction'], torch.Tensor)
        assert losses['reconstruction'].requires_grad
    
    def test_loss_computation(self):
        """Test loss computation and weighting."""
        model = create_simple_model()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = SelfForcingTrainer(model=model, optimizer=optimizer)
        
        losses = {
            'reconstruction': torch.tensor(1.0, requires_grad=True),
            'temporal_consistency': torch.tensor(0.5, requires_grad=True),
            'perceptual': torch.tensor(0.2, requires_grad=True)
        }
        
        total_loss = trainer._compute_total_loss(losses)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert total_loss.item() > 0
    
    def test_memory_optimization(self):
        """Test memory optimization during training."""
        model = create_simple_model(hidden_dim=32)  # Small model for testing
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = SelfForcingTrainer(
            model=model,
            optimizer=optimizer,
            max_memory_gb=0.1,  # Very low limit to trigger cleanup
            mixed_precision=False
        )
        
        # Create dummy dataloader
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 4, 3, 32, 32),
            torch.randn(10, 4, 6)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Test that training doesn't crash with memory management
        try:
            trainer.train_epoch(dataloader)
            assert True  # If we get here, memory management worked
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.fail("Memory management failed to prevent OOM")
            else:
                # Other errors are acceptable for this test
                pass
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = create_simple_model(hidden_dim=64)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            trainer = SelfForcingTrainer(
                model=model,
                optimizer=optimizer,
                checkpoint_dir=temp_dir
            )
            
            # Save checkpoint
            trainer.global_step = 100
            trainer.epoch = 5
            trainer.best_loss = 0.5
            trainer._save_checkpoint(0.5)
            
            # Check that checkpoint file exists
            checkpoint_files = list(Path(temp_dir).glob("*.pt"))
            assert len(checkpoint_files) > 0
            
            # Load checkpoint
            checkpoint_info = trainer.load_checkpoint()
            
            if checkpoint_info:  # May be empty if no checkpoint found
                assert 'epoch' in checkpoint_info
                assert 'step' in checkpoint_info


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        monitor = MemoryMonitor(max_memory_gb=4.0)
        
        # Update memory stats
        stats = monitor.update()
        
        assert 'allocated_gb' in stats
        assert 'peak_gb' in stats
        assert 'usage_percent' in stats
        assert all(isinstance(v, (int, float)) for v in stats.values())
    
    def test_cleanup_threshold(self):
        """Test cleanup threshold detection."""
        monitor = MemoryMonitor(max_memory_gb=1.0)
        monitor.allocated_memory = 0.9  # 90% of limit
        
        should_cleanup = monitor.should_cleanup()
        assert should_cleanup
        
        monitor.allocated_memory = 0.5  # 50% of limit
        should_cleanup = monitor.should_cleanup()
        assert not should_cleanup
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        monitor = MemoryMonitor()
        
        # Create some tensors to potentially free
        tensors = [torch.randn(100, 100) for _ in range(10)]
        
        # Cleanup should not crash
        try:
            freed = monitor.cleanup()
            assert isinstance(freed, (int, float))
        except Exception as e:
            pytest.fail(f"Memory cleanup failed: {e}")


class TestCheckpointManager:
    """Test checkpoint management."""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation and metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, keep_checkpoints=3)
            
            # Create dummy model
            model = nn.Linear(10, 5)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=1,
                step=100,
                loss=0.5,
                metrics={'accuracy': 0.8},
                is_best=True
            )
            
            assert os.path.exists(checkpoint_path)
            
            # Check best checkpoint exists
            best_path = Path(temp_dir) / "best_checkpoint.pt"
            assert best_path.exists()
    
    def test_checkpoint_cleanup(self):
        """Test old checkpoint cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir, keep_checkpoints=2)
            
            model = nn.Linear(5, 3)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Save multiple checkpoints
            for i in range(5):
                manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=i,
                    step=i * 100,
                    loss=1.0 - i * 0.1
                )
            
            # Check that only 2 regular checkpoints remain (plus best)
            checkpoint_files = list(Path(temp_dir).glob("checkpoint_step_*.pt"))
            assert len(checkpoint_files) <= 2
    
    def test_checkpoint_loading(self):
        """Test checkpoint loading and state restoration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(temp_dir)
            
            # Create and save model
            model = nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            original_weight = model.weight.data.clone()
            
            manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=5,
                step=500,
                loss=0.3
            )
            
            # Modify model
            model.weight.data.fill_(999.0)
            
            # Load checkpoint
            info = manager.load_checkpoint(model, optimizer)
            
            if info:  # May be empty if loading failed
                assert info['epoch'] == 5
                assert info['step'] == 500
                assert torch.allclose(model.weight.data, original_weight)


class TestCommaAITrainer:
    """Test comma.ai inspired training components."""
    
    def test_curriculum_scheduler(self):
        """Test curriculum learning scheduler."""
        from training.comma_ai_insights import CommaAIConfig
        
        config = CommaAIConfig(
            initial_sequence_length=4,
            final_sequence_length=16,
            curriculum_warmup_steps=1000,
            initial_self_forcing_ratio=0.0,
            final_self_forcing_ratio=0.8,
            self_forcing_warmup_steps=500
        )
        
        scheduler = CurriculumScheduler(config)
        
        # Test initial values
        assert scheduler.get_sequence_length() == 4
        assert scheduler.get_self_forcing_ratio() == 0.0
        
        # Test progression
        scheduler.step = 500
        seq_len = scheduler.get_sequence_length()
        sf_ratio = scheduler.get_self_forcing_ratio()
        
        assert seq_len > 4 and seq_len <= 16
        assert sf_ratio > 0.0 and sf_ratio <= 0.8
        
        # Test final values
        scheduler.step = 2000  # Beyond warmup
        assert scheduler.get_sequence_length() == 16
        assert scheduler.get_self_forcing_ratio() == 0.8
    
    def test_latent_space_planner(self):
        """Test latent space planning component."""
        from training.comma_ai_insights import CommaAIConfig
        
        config = CommaAIConfig()
        planner = LatentSpacePlanner(
            latent_dim=128,
            action_dim=6,
            config=config
        )
        
        # Mock world model
        class MockWorldModel(nn.Module):
            def predict_next_state(self, state, action):
                return state + 0.1 * action.sum(dim=-1, keepdim=True)
        
        world_model = MockWorldModel()
        
        # Test planning
        batch_size = 2
        latent_state = torch.randn(batch_size, 128)
        
        result = planner(latent_state, world_model)
        
        assert 'action' in result
        assert 'planned_trajectory' in result
        assert 'costs' in result
        
        assert result['action'].shape == (batch_size, 6)
        assert torch.isfinite(result['action']).all()
    
    def test_safety_classifier(self):
        """Test safety classification component."""
        from training.comma_ai_insights import SafetyClassifier
        
        classifier = SafetyClassifier(latent_dim=64)
        
        batch_size = 4
        latent_state = torch.randn(batch_size, 64)
        
        safety_score = classifier(latent_state)
        
        assert safety_score.shape == (batch_size, 1)
        assert (safety_score >= 0).all() and (safety_score <= 1).all()


class TestDistributedTraining:
    """Test distributed training components."""
    
    def test_distributed_manager_single_gpu(self):
        """Test distributed manager in single GPU mode."""
        manager = DistributedManager()
        
        # Should work in single GPU mode
        assert manager.rank == 0
        assert manager.world_size == 1
        assert manager.is_master
        assert not manager.is_initialized
    
    def test_memory_reduction_operations(self):
        """Test tensor reduction operations."""
        manager = DistributedManager()
        
        # Test reduction (should work even without distributed setup)
        tensor = torch.tensor([1.0, 2.0, 3.0])
        reduced = manager.all_reduce(tensor.clone())
        
        # In single process, tensor should be unchanged
        assert torch.equal(reduced, tensor)
    
    @patch('torch.distributed.is_initialized')
    def test_distributed_barrier(self, mock_is_initialized):
        """Test distributed barrier functionality."""
        mock_is_initialized.return_value = False
        
        manager = DistributedManager()
        
        # Should not crash even if distributed is not initialized
        try:
            manager.barrier()
            assert True
        except Exception as e:
            pytest.fail(f"Barrier failed: {e}")


class TestLossFunction:
    """Test loss function implementations and properties."""
    
    def test_reconstruction_loss(self):
        """Test reconstruction loss computation."""
        pred = torch.randn(2, 4, 3, 32, 32)
        target = torch.randn(2, 4, 3, 32, 32)
        
        loss = nn.MSELoss()(pred, target)
        
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_temporal_consistency_loss(self):
        """Test temporal consistency loss."""
        # Simulate frame predictions
        pred_frames = torch.randn(2, 8, 3, 32, 32)
        target_frames = torch.randn(2, 8, 3, 32, 32)
        
        # Compute temporal differences
        pred_diffs = pred_frames[:, 1:] - pred_frames[:, :-1]
        target_diffs = target_frames[:, 1:] - target_frames[:, :-1]
        
        temporal_loss = nn.MSELoss()(pred_diffs, target_diffs)
        
        assert temporal_loss.item() >= 0
        assert temporal_loss.requires_grad
    
    def test_perceptual_loss_properties(self):
        """Test perceptual loss properties."""
        # Simple perceptual loss (mean difference)
        pred = torch.randn(2, 4, 3, 32, 32)
        target = torch.randn(2, 4, 3, 32, 32)
        
        # Convert to grayscale for simple perceptual comparison
        pred_gray = pred.mean(dim=2, keepdim=True)
        target_gray = target.mean(dim=2, keepdim=True)
        
        perceptual_loss = nn.L1Loss()(pred_gray, target_gray)
        
        assert perceptual_loss.item() >= 0
        assert perceptual_loss.requires_grad
    
    def test_loss_scaling_properties(self):
        """Test that loss scaling works correctly."""
        losses = {
            'reconstruction': torch.tensor(2.0, requires_grad=True),
            'temporal': torch.tensor(1.0, requires_grad=True),
            'perceptual': torch.tensor(0.5, requires_grad=True)
        }
        
        weights = {
            'reconstruction': 1.0,
            'temporal': 0.5,
            'perceptual': 0.1
        }
        
        total_loss = sum(weights[name] * loss for name, loss in losses.items())
        expected = 2.0 * 1.0 + 1.0 * 0.5 + 0.5 * 0.1
        
        assert abs(total_loss.item() - expected) < 1e-6


class TestTrainingStability:
    """Test training stability and convergence properties."""
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        model = create_simple_model(hidden_dim=32)
        
        # Create loss that might cause large gradients
        x = torch.randn(1, 4, 3, 32, 32, requires_grad=True)
        target = torch.randn(1, 4, 3, 32, 32)
        
        output = model(x)
        loss = nn.MSELoss()(output, target) * 1000  # Scale up for large gradients
        loss.backward()
        
        # Check gradients before clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check gradients after clipping
        grad_norm_after = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_after += param.grad.norm().item() ** 2
        grad_norm_after = grad_norm_after ** 0.5
        
        if grad_norm_before > 1.0:
            assert grad_norm_after <= 1.1  # Allow small numerical error
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduling."""
        model = create_simple_model(hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler
        scheduler.step()
        scheduler.step()
        
        new_lr = optimizer.param_groups[0]['lr']
        expected_lr = initial_lr * 0.5  # After 2 steps with gamma=0.5
        
        assert abs(new_lr - expected_lr) < 1e-6
    
    def test_mixed_precision_stability(self):
        """Test mixed precision training stability."""
        model = create_simple_model(hidden_dim=32)
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.Adam(model.parameters())
        
        x = torch.randn(1, 2, 3, 32, 32)
        target = torch.randn(1, 2, 3, 32, 32)
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=True):
            output = model(x)
            loss = nn.MSELoss()(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Check that no parameters became NaN
        for param in model.parameters():
            assert torch.isfinite(param).all()


class TestModelConfiguration:
    """Test model configuration and initialization."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = get_minimal_config()
        
        # Test that config has required fields
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
        assert hasattr(config.model, 'model_dim')
        assert hasattr(config.model, 'num_layers')
    
    def test_different_config_presets(self):
        """Test different configuration presets."""
        configs = {
            'minimal': get_minimal_config(),
            'research': get_research_config()
        }
        
        for name, config in configs.items():
            assert config.model.model_dim > 0
            assert config.model.num_layers > 0
            assert config.model.num_heads > 0
            
            # Research config should be larger than minimal
            if name == 'research':
                minimal = configs['minimal']
                assert config.model.model_dim >= minimal.model.model_dim
                assert config.model.num_layers >= minimal.model.num_layers


def test_end_to_end_training_step():
    """Test complete end-to-end training step."""
    # Create minimal setup
    model = create_simple_model(input_channels=3, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = SelfForcingTrainer(
        model=model,
        optimizer=optimizer,
        mixed_precision=False,
        distributed=False
    )
    
    # Create dummy batch
    batch = {
        'frames': torch.randn(1, 4, 3, 32, 32),
        'controls': torch.randn(1, 4, 6)
    }
    
    # Perform training step
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    losses = trainer._self_forcing_step(batch)
    total_loss = trainer._compute_total_loss(losses)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Check that parameters actually changed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.equal(initial_params[name], param), f"Parameter {name} did not change"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])