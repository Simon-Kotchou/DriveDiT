"""
Integration tests for training components and pipelines.
Tests complete training workflows and component interactions.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List
import tempfile
from pathlib import Path

# Import training modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.losses import (
    ReconstructionLoss, TemporalConsistencyLoss, DepthConsistencyLoss,
    JEPAContrastiveLoss, FlowMatchingLoss, CompositeLoss
)
from training.distill import DistillationTrainer
from models.vae3d import VAE3D
from models.dit_student import DiTStudent
from models.dit_teacher import DiTTeacher
from config.model_config import get_small_config


class TestLossFunctions:
    """Test loss function implementations."""
    
    def test_reconstruction_loss(self, device):
        """Test reconstruction loss computation."""
        recon_loss = ReconstructionLoss(
            pixel_weight=1.0,
            perceptual_weight=0.1,
            ssim_weight=0.1
        )
        
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred_frames = torch.randn(B, T, C, H, W, device=device)
        target_frames = torch.randn(B, T, C, H, W, device=device)
        
        losses = recon_loss(pred_frames, target_frames)
        
        assert 'pixel' in losses
        assert isinstance(losses['pixel'], torch.Tensor)
        assert torch.isfinite(losses['pixel'])
        assert losses['pixel'] >= 0
    
    def test_reconstruction_loss_with_mask(self, device):
        """Test reconstruction loss with mask."""
        recon_loss = ReconstructionLoss()
        
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred_frames = torch.randn(B, T, C, H, W, device=device)
        target_frames = torch.randn(B, T, C, H, W, device=device)
        mask = torch.randint(0, 2, (B, T, 1, H, W), device=device).float()
        
        losses = recon_loss(pred_frames, target_frames, mask=mask)
        
        assert 'pixel' in losses
        assert torch.isfinite(losses['pixel'])
    
    def test_temporal_consistency_loss(self, device):
        """Test temporal consistency loss."""
        temporal_loss = TemporalConsistencyLoss()
        
        B, T, C, H, W = 2, 6, 3, 32, 32
        pred_frames = torch.randn(B, T, C, H, W, device=device)
        target_frames = torch.randn(B, T, C, H, W, device=device)
        
        losses = temporal_loss(pred_frames, target_frames)
        
        assert 'frame_diff' in losses
        assert torch.isfinite(losses['frame_diff'])
        assert losses['frame_diff'] >= 0
    
    def test_depth_consistency_loss(self, device):
        """Test depth consistency loss."""
        depth_loss = DepthConsistencyLoss()
        
        B, T, C, H, W = 2, 4, 3, 32, 32
        pred_frames = torch.randn(B, T, C, H, W, device=device)
        target_depth = torch.randn(B, T, 1, H, W, device=device).abs()  # Positive depths
        
        losses = depth_loss(pred_frames, target_depth)
        
        assert 'depth' in losses
        assert torch.isfinite(losses['depth'])
        assert losses['depth'] >= 0
    
    def test_jepa_contrastive_loss(self, device):
        """Test JEPA contrastive loss."""
        jepa_loss = JEPAContrastiveLoss(temperature=0.1)
        
        B, T, C, H, W = 2, 8, 16, 8, 8
        pred_latents = torch.randn(B, T, C, H, W, device=device)
        target_latents = torch.randn(B, T, C, H, W, device=device)
        
        loss = jepa_loss(pred_latents, target_latents)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)
        assert loss >= 0
    
    def test_flow_matching_loss(self, device):
        """Test flow matching loss."""
        flow_loss = FlowMatchingLoss()
        
        B, T, C, H, W = 2, 4, 8, 8, 8
        flow_pred = torch.randn(B, T, C, H, W, device=device)
        z_current = torch.randn(B, T, C, H, W, device=device)
        z_target = torch.randn(B, T, C, H, W, device=device)
        timesteps = torch.randint(0, 100, (B,), device=device)
        noise_schedule = torch.linspace(0.1, 1.0, 100, device=device)
        
        loss = flow_loss(flow_pred, z_current, z_target, timesteps, noise_schedule)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss >= 0
    
    def test_composite_loss(self, device):
        """Test composite loss combination."""
        loss_weights = {
            'reconstruction': 1.0,
            'temporal': 0.5,
            'depth': 0.1,
            'contrastive': 0.2
        }
        
        composite_loss = CompositeLoss(loss_weights)
        
        # Create individual losses
        losses = {
            'reconstruction': torch.tensor(2.0, device=device),
            'temporal': torch.tensor(1.5, device=device),
            'depth': torch.tensor(0.8, device=device),
            'contrastive': torch.tensor(3.2, device=device)
        }
        
        total_loss, weights_used = composite_loss(losses)
        
        expected_loss = 1.0*2.0 + 0.5*1.5 + 0.1*0.8 + 0.2*3.2
        
        assert torch.allclose(total_loss, torch.tensor(expected_loss, device=device))
        assert len(weights_used) == len(loss_weights)
    
    def test_composite_loss_with_uncertainty_weighting(self, device):
        """Test composite loss with uncertainty weighting."""
        loss_weights = {
            'loss1': 1.0,
            'loss2': 1.0
        }
        
        composite_loss = CompositeLoss(
            loss_weights, 
            uncertainty_weighting=True
        ).to(device)
        
        losses = {
            'loss1': torch.tensor(2.0, device=device),
            'loss2': torch.tensor(1.0, device=device)
        }
        
        # Forward pass (uncertainty weights are learnable)
        total_loss, weights_used = composite_loss(losses)
        
        assert isinstance(total_loss, torch.Tensor)
        assert torch.isfinite(total_loss)
        assert total_loss.requires_grad  # Should require gradients for uncertainty weights


class TestDistillationTraining:
    """Test distillation training functionality."""
    
    @pytest.fixture
    def distillation_setup(self, device):
        """Setup for distillation tests."""
        config = get_small_config()
        
        # Create models
        teacher = DiTTeacher(
            latent_dim=config.teacher.latent_dim,
            d_model=config.teacher.d_model,
            n_layers=config.teacher.n_layers,
            n_heads=config.teacher.n_heads,
            num_diffusion_steps=50  # Smaller for testing
        ).to(device)
        
        student = DiTStudent(
            latent_dim=config.student.latent_dim,
            d_model=config.student.d_model,
            n_layers=config.student.n_layers,
            n_heads=config.student.n_heads
        ).to(device)
        
        vae = VAE3D(
            in_channels=config.vae.in_channels,
            latent_dim=config.vae.latent_dim,
            hidden_dims=[32, 64, 128],  # Smaller for testing
            num_res_blocks=1
        ).to(device)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        return teacher, student, vae, optimizer
    
    def test_distillation_trainer_initialization(self, distillation_setup, device):
        """Test distillation trainer initialization."""
        teacher, student, vae, optimizer = distillation_setup
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            vae_model=vae,
            optimizer=optimizer,
            device=device,
            use_wandb=False
        )
        
        # Check that teacher and VAE are frozen
        assert not any(p.requires_grad for p in trainer.teacher_model.parameters())
        assert not any(p.requires_grad for p in trainer.vae_model.parameters())
        
        # Check that student is trainable
        assert any(p.requires_grad for p in trainer.student_model.parameters())
    
    def test_distillation_step(self, distillation_setup, device):
        """Test single distillation step."""
        teacher, student, vae, optimizer = distillation_setup
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            vae_model=vae,
            optimizer=optimizer,
            device=device,
            use_wandb=False
        )
        
        # Create sample data
        B, T, C, H, W = 2, 4, 3, 32, 32
        frames = torch.randn(B, T, C, H, W, device=device)
        controls = torch.randn(B, T, 4, device=device)
        
        # Test distillation step
        losses = trainer._distillation_step(frames, controls)
        
        assert isinstance(losses, dict)
        assert len(losses) > 0
        
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert torch.isfinite(loss_value)
            assert loss_value >= 0
    
    def test_distillation_backward_pass(self, distillation_setup, device):
        """Test distillation backward pass."""
        teacher, student, vae, optimizer = distillation_setup
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            vae_model=vae,
            optimizer=optimizer,
            device=device,
            use_wandb=False
        )
        
        # Create sample data
        B, T, C, H, W = 1, 2, 3, 16, 16  # Small for testing
        frames = torch.randn(B, T, C, H, W, device=device)
        
        # Forward pass
        losses = trainer._distillation_step(frames)
        total_loss = trainer._compute_total_loss(losses)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients
        for name, param in student.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
    
    @pytest.mark.slow
    def test_distillation_training_epoch(self, distillation_setup, device):
        """Test training for one epoch."""
        teacher, student, vae, optimizer = distillation_setup
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            vae_model=vae,
            optimizer=optimizer,
            device=device,
            use_wandb=False
        )
        
        # Create mock dataloader
        class MockDataLoader:
            def __init__(self, num_batches=3):
                self.num_batches = num_batches
                self.batch_data = {
                    'frames': torch.randn(1, 2, 3, 16, 16, device=device),
                    'controls': torch.randn(1, 2, 4, device=device)
                }
            
            def __iter__(self):
                for _ in range(self.num_batches):
                    yield self.batch_data
            
            def __len__(self):
                return self.num_batches
        
        dataloader = MockDataLoader(num_batches=2)  # Small for testing
        
        # Train for one epoch
        epoch_losses = trainer.train_epoch(dataloader)
        
        assert isinstance(epoch_losses, dict)
        assert len(epoch_losses) > 0
        
        for loss_name, avg_loss in epoch_losses.items():
            assert isinstance(avg_loss, float)
            assert avg_loss >= 0
    
    def test_distillation_checkpoint_save_load(self, distillation_setup, device, temp_dir):
        """Test checkpoint saving and loading."""
        teacher, student, vae, optimizer = distillation_setup
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            vae_model=vae,
            optimizer=optimizer,
            device=device,
            use_wandb=False
        )
        
        # Save checkpoint
        checkpoint_path = temp_dir / "distill_checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), best_metric=0.5)
        
        assert checkpoint_path.exists()
        
        # Create new trainer and load
        new_student = DiTStudent(
            latent_dim=student.latent_dim,
            d_model=student.d_model,
            n_layers=student.n_layers,
            n_heads=student.n_heads
        ).to(device)
        
        new_optimizer = torch.optim.Adam(new_student.parameters(), lr=1e-4)
        
        new_trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=new_student,
            vae_model=vae,
            optimizer=new_optimizer,
            device=device,
            use_wandb=False
        )
        
        new_trainer.load_checkpoint(str(checkpoint_path))
        
        # Check that student models have same parameters
        for p1, p2 in zip(student.parameters(), new_student.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)


class TestTrainingPipelines:
    """Test complete training pipelines."""
    
    def test_vae_training_step(self, device):
        """Test VAE training step."""
        config = get_small_config()
        
        vae = VAE3D(
            in_channels=config.vae.in_channels,
            latent_dim=config.vae.latent_dim,
            hidden_dims=[32, 64],  # Very small for testing
            num_res_blocks=1
        ).to(device)
        
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        
        # Training data
        B, T, C, H, W = 1, 2, 3, 16, 16
        frames = torch.randn(B, T, C, H, W, device=device)
        
        # Training step
        vae.train()
        optimizer.zero_grad()
        
        recon, mean, logvar = vae(frames)
        loss = vae.loss_function(recon, frames, mean, logvar)
        
        loss.backward()
        optimizer.step()
        
        # Check that loss is reasonable
        assert torch.isfinite(loss)
        assert loss > 0
        
        # Check that parameters were updated
        for param in vae.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_dit_student_training_step(self, device):
        """Test DiT student training step."""
        config = get_small_config()
        
        student = DiTStudent(
            latent_dim=config.student.latent_dim,
            d_model=config.student.d_model,
            n_layers=config.student.n_layers,
            n_heads=config.student.n_heads,
            max_seq_len=64  # Smaller for testing
        ).to(device)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        # Training data
        B, T, latent_dim = 1, 4, config.student.latent_dim
        tokens = torch.randn(B, T, latent_dim, device=device)
        target_tokens = torch.randn(B, T, latent_dim, device=device)
        
        # Training step
        student.train()
        loss = student.compute_loss(tokens, target_tokens)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check training step
        assert torch.isfinite(loss)
        assert loss >= 0
    
    def test_end_to_end_training_compatibility(self, device):
        """Test end-to-end training compatibility."""
        config = get_small_config()
        
        # Create all models
        vae = VAE3D(
            in_channels=config.vae.in_channels,
            latent_dim=config.vae.latent_dim,
            hidden_dims=[16, 32],  # Very small
            num_res_blocks=1
        ).to(device)
        
        student = DiTStudent(
            latent_dim=config.student.latent_dim,
            d_model=64,  # Very small
            n_layers=2,
            n_heads=4
        ).to(device)
        
        teacher = DiTTeacher(
            latent_dim=config.teacher.latent_dim,
            d_model=64,  # Very small
            n_layers=2,
            n_heads=4,
            num_diffusion_steps=10  # Very small
        ).to(device)
        
        # Test data flow
        B, T, C, H, W = 1, 2, 3, 16, 16
        frames = torch.randn(B, T, C, H, W, device=device)
        
        # 1. VAE encode
        with torch.no_grad():
            mean, logvar = vae.encode(frames)
            latents = vae.reparameterize(mean, logvar)
        
        # 2. Student process
        B_lat, T_lat, C_lat, H_lat, W_lat = latents.shape
        tokens = latents.view(B_lat, T_lat, -1)
        
        student_output, _ = student(tokens)
        processed_latents = student_output.view(B_lat, T_lat, C_lat, H_lat, W_lat)
        
        # 3. VAE decode
        with torch.no_grad():
            reconstructed = vae.decode(processed_latents)
        
        # 4. Teacher loss
        teacher_losses = teacher.compute_loss(latents)
        
        # Check everything works
        assert reconstructed.shape == frames.shape
        assert torch.isfinite(reconstructed).all()
        assert torch.isfinite(teacher_losses['total_loss'])
    
    @pytest.mark.slow
    def test_memory_efficiency_during_training(self, device):
        """Test memory efficiency during training."""
        if device.type == 'cpu':
            pytest.skip("Memory test requires GPU")
        
        config = get_small_config()
        
        # Create model
        student = DiTStudent(
            latent_dim=config.student.latent_dim,
            d_model=128,
            n_layers=4,
            n_heads=8
        ).to(device)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        # Measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Training loop
        for step in range(5):
            # Create batch
            tokens = torch.randn(2, 8, config.student.latent_dim, device=device)
            target = torch.randn_like(tokens)
            
            # Forward pass
            output, _ = student(tokens)
            loss = nn.MSELoss()(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clean up
            del tokens, target, output, loss
            
            if step % 2 == 0:
                torch.cuda.empty_cache()
        
        # Final memory check
        final_memory = torch.cuda.memory_allocated(device)
        memory_growth = final_memory - initial_memory
        
        # Should not grow excessively (allow some growth for model state)
        max_allowed_growth = 50 * 1024 * 1024  # 50MB
        assert memory_growth < max_allowed_growth, f"Memory grew by {memory_growth / 1024 / 1024:.1f}MB"


# Performance benchmarks
class TestTrainingPerformance:
    """Test training performance characteristics."""
    
    @pytest.mark.performance
    def test_loss_computation_speed(self, device, performance_timer):
        """Test loss computation performance."""
        # Setup
        B, T, C, H, W = 4, 8, 3, 64, 64
        pred_frames = torch.randn(B, T, C, H, W, device=device)
        target_frames = torch.randn(B, T, C, H, W, device=device)
        
        recon_loss = ReconstructionLoss()
        
        # Warm up
        for _ in range(3):
            losses = recon_loss(pred_frames, target_frames)
        
        # Benchmark
        performance_timer.start()
        for _ in range(10):
            losses = recon_loss(pred_frames, target_frames)
        elapsed_time = performance_timer.stop()
        
        # Should be reasonably fast
        avg_time_per_iteration = elapsed_time / 10
        assert avg_time_per_iteration < 0.1, f"Loss computation too slow: {avg_time_per_iteration:.3f}s"
    
    @pytest.mark.performance
    def test_distillation_step_speed(self, device, performance_timer):
        """Test distillation step performance."""
        if device.type == 'cpu':
            pytest.skip("Performance test requires GPU")
        
        config = get_small_config()
        
        # Create minimal models for performance testing
        teacher = DiTTeacher(
            latent_dim=config.teacher.latent_dim,
            d_model=256,
            n_layers=4,
            n_heads=8,
            num_diffusion_steps=20
        ).to(device)
        
        student = DiTStudent(
            latent_dim=config.student.latent_dim,
            d_model=256,
            n_layers=4,
            n_heads=8
        ).to(device)
        
        vae = VAE3D(
            in_channels=config.vae.in_channels,
            latent_dim=config.vae.latent_dim,
            hidden_dims=[64, 128],
            num_res_blocks=1
        ).to(device)
        
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        
        trainer = DistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            vae_model=vae,
            optimizer=optimizer,
            device=device,
            use_wandb=False
        )
        
        # Test data
        B, T, C, H, W = 2, 4, 3, 32, 32
        frames = torch.randn(B, T, C, H, W, device=device)
        
        # Warm up
        for _ in range(2):
            losses = trainer._distillation_step(frames)
        
        # Benchmark
        performance_timer.start()
        for _ in range(5):
            losses = trainer._distillation_step(frames)
        elapsed_time = performance_timer.stop()
        
        avg_time_per_step = elapsed_time / 5
        # Should complete in reasonable time
        assert avg_time_per_step < 2.0, f"Distillation step too slow: {avg_time_per_step:.3f}s"