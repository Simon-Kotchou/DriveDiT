"""
Unit tests for model components (VAE3D, DiTStudent, DiTTeacher).
Tests model architecture, forward passes, and mathematical properties.
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple
import math

# Import models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.vae3d import VAE3D, Encoder3D, Decoder3D, causal_conv3d
from models.dit_student import DiTStudent, TokenEmbedding, MemoryBank
from models.dit_teacher import DiTTeacher, TeacherEmbedding


class TestVAE3D:
    """Test 3D Causal VAE implementation."""
    
    def test_vae_forward_pass(self, vae_model, sample_video_data, device):
        """Test VAE forward pass."""
        # Use smaller data for testing
        frames = sample_video_data[:, :4, :, :32, :32]  # [B, T, C, H, W]
        
        # Forward pass
        recon, mean, logvar = vae_model(frames)
        
        # Check shapes
        assert recon.shape == frames.shape
        assert mean.shape[0] == frames.shape[0]  # Batch dimension
        assert logvar.shape == mean.shape
        assert torch.isfinite(recon).all()
        assert torch.isfinite(mean).all()
        assert torch.isfinite(logvar).all()
    
    def test_vae_encode_decode_consistency(self, vae_model, sample_video_data, device):
        """Test encode-decode consistency."""
        frames = sample_video_data[:, :4, :, :32, :32]
        
        # Encode
        mean, logvar = vae_model.encode(frames)
        z = vae_model.reparameterize(mean, logvar)
        
        # Decode
        recon = vae_model.decode(z)
        
        # Check shapes match
        assert recon.shape == frames.shape
        assert torch.isfinite(recon).all()
    
    def test_vae_loss_computation(self, vae_model, sample_video_data, device):
        """Test VAE loss computation."""
        frames = sample_video_data[:, :4, :, :32, :32]
        
        recon, mean, logvar = vae_model(frames)
        loss = vae_model.loss_function(recon, frames, mean, logvar)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert torch.isfinite(loss)
        assert loss >= 0  # Loss should be non-negative
    
    def test_causal_conv3d(self, device):
        """Test causal 3D convolution."""
        in_channels, out_channels = 3, 16
        kernel_size = (3, 3, 3)
        
        conv = causal_conv3d(in_channels, out_channels, kernel_size).to(device)
        
        B, C, T, H, W = 2, 3, 8, 16, 16
        x = torch.randn(B, C, T, H, W, device=device)
        
        output = conv(x)
        
        # Check output shape
        expected_shape = (B, out_channels, T, H, W)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()
    
    def test_vae_deterministic_mode(self, vae_model, sample_video_data, device):
        """Test VAE in deterministic mode (eval)."""
        frames = sample_video_data[:, :4, :, :32, :32]
        
        vae_model.eval()
        
        with torch.no_grad():
            # Multiple forward passes should give same result in eval mode
            mean1, logvar1 = vae_model.encode(frames)
            z1 = vae_model.reparameterize(mean1, logvar1)
            
            mean2, logvar2 = vae_model.encode(frames)
            z2 = vae_model.reparameterize(mean2, logvar2)
            
            # In eval mode, reparameterize should return mean
            assert torch.allclose(z1, mean1, atol=1e-6)
            assert torch.allclose(z2, mean2, atol=1e-6)
    
    def test_vae_latent_space_properties(self, vae_model, sample_video_data, device):
        """Test latent space properties."""
        frames = sample_video_data[:, :4, :, :32, :32]
        
        mean, logvar = vae_model.encode(frames)
        
        # Check that logvar is reasonable (not too extreme)
        assert torch.all(logvar > -10)  # Not too negative
        assert torch.all(logvar < 10)   # Not too positive
        
        # Check that mean has reasonable magnitude
        assert torch.all(torch.abs(mean) < 100)  # Not exploding
    
    @pytest.mark.slow
    def test_vae_gradient_flow(self, vae_model, sample_video_data, device):
        """Test gradient flow through VAE."""
        frames = sample_video_data[:, :4, :, :32, :32]
        frames.requires_grad_(True)
        
        vae_model.train()
        
        recon, mean, logvar = vae_model(frames)
        loss = vae_model.loss_function(recon, frames, mean, logvar)
        
        loss.backward()
        
        # Check that gradients exist and are finite
        for name, param in vae_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestDiTStudent:
    """Test DiT Student model implementation."""
    
    def test_dit_student_forward_pass(self, dit_student_model, sample_latent_data, device):
        """Test DiT student forward pass."""
        # Use smaller latent data
        latents = sample_latent_data[:, :4, :4, :4, :4]  # [B, T, C, H, W]
        
        # Flatten for token input
        B, T, C, H, W = latents.shape
        tokens = latents.view(B, T, -1)  # [B, T, C*H*W]
        
        output, _ = dit_student_model(tokens)
        
        assert output.shape == tokens.shape
        assert torch.isfinite(output).all()
    
    def test_dit_student_with_kv_cache(self, dit_student_model, sample_latent_data, device):
        """Test DiT student with KV caching."""
        latents = sample_latent_data[:, :4, :4, :4, :4]
        B, T, C, H, W = latents.shape
        tokens = latents.view(B, T, -1)
        
        # First forward pass with cache
        output1, kv_cache = dit_student_model(tokens, use_cache=True)
        
        # Second forward pass with new token
        new_token = torch.randn(B, 1, tokens.size(-1), device=device)
        output2, new_kv_cache = dit_student_model(
            new_token, kv_cache=kv_cache, use_cache=True
        )
        
        assert output1.shape == tokens.shape
        assert output2.shape == new_token.shape
        assert kv_cache is not None
        assert new_kv_cache is not None
    
    def test_dit_student_generation(self, dit_student_model, sample_latent_data, device):
        """Test DiT student generation capabilities."""
        context = sample_latent_data[:, :2, :4, :4, :4]  # [B, T_ctx, C, H, W]
        B, T_ctx, C, H, W = context.shape
        
        # Generate sequence
        generated = dit_student_model.generate(
            context.view(B, T_ctx, -1),
            max_new_tokens=4,
            temperature=0.8
        )
        
        expected_length = T_ctx + 4
        assert generated.shape == (B, expected_length, C * H * W)
        assert torch.isfinite(generated).all()
    
    def test_token_embedding(self, device):
        """Test token embedding component."""
        latent_dim = 64
        d_model = 256
        vocab_size = 1000
        max_seq_len = 128
        
        token_embed = TokenEmbedding(
            latent_dim, d_model, vocab_size, max_seq_len
        ).to(device)
        
        # Test continuous tokens
        B, T = 2, 8
        latent_tokens = torch.randn(B, T, latent_dim, device=device)
        embedded = token_embed(latent_tokens=latent_tokens)
        
        assert embedded.shape == (B, T, d_model)
        assert torch.isfinite(embedded).all()
        
        # Test discrete tokens
        discrete_tokens = torch.randint(0, vocab_size, (B, T), device=device)
        embedded_discrete = token_embed(discrete_tokens=discrete_tokens)
        
        assert embedded_discrete.shape == (B, T, d_model)
        assert torch.isfinite(embedded_discrete).all()
    
    def test_memory_bank(self, device):
        """Test memory bank functionality."""
        d_model = 256
        max_memory = 100
        
        memory_bank = MemoryBank(
            d_model=d_model,
            max_spatial_memory=max_memory
        ).to(device) if hasattr(MemoryBank, 'to') else MemoryBank(
            d_model=d_model,
            max_spatial_memory=max_memory
        )
        
        # Add some memories
        for i in range(10):
            new_tokens = torch.randn(2, 16, d_model, device=device)
            importance = torch.rand(2, 16, device=device)
            
            if hasattr(memory_bank, 'update'):
                memory_bank.update(new_tokens, importance)
        
        # Retrieve memories
        if hasattr(memory_bank, 'retrieve'):
            query = torch.randn(2, 4, d_model, device=device)
            retrieved = memory_bank.retrieve(query, top_k=32)
            
            assert retrieved.shape[0] == query.shape[0]  # Batch dimension
            assert retrieved.shape[2] == d_model  # Feature dimension
            assert torch.isfinite(retrieved).all()


class TestDiTTeacher:
    """Test DiT Teacher model implementation."""
    
    def test_dit_teacher_forward_pass(self, dit_teacher_model, sample_latent_data, device):
        """Test DiT teacher forward pass."""
        latents = sample_latent_data[:, :4, :4, :4, :4]  # [B, T, C, H, W]
        B, T, C, H, W = latents.shape
        
        # Teacher expects noisy latents and timesteps
        timesteps = torch.randint(0, 100, (B,), device=device)
        
        # Reshape for teacher input
        teacher_input = latents.view(B, T * C, H, W)  # Flatten time and channels
        
        flow_pred = dit_teacher_model(teacher_input.unsqueeze(2), timesteps)  # Add time dim
        
        assert flow_pred.shape == teacher_input.unsqueeze(2).shape
        assert torch.isfinite(flow_pred).all()
    
    def test_dit_teacher_noise_addition(self, dit_teacher_model, sample_latent_data, device):
        """Test teacher noise addition functionality."""
        latents = sample_latent_data[:, :4, :4, :4, :4]
        B, T, C, H, W = latents.shape
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 100, (B,), device=device)
        
        noisy_latents = dit_teacher_model.add_noise(latents, noise, timesteps)
        
        assert noisy_latents.shape == latents.shape
        assert torch.isfinite(noisy_latents).all()
        
        # Noisy latents should be different from clean latents
        assert not torch.allclose(noisy_latents, latents, atol=1e-6)
    
    def test_dit_teacher_loss_computation(self, dit_teacher_model, sample_latent_data, device):
        """Test teacher loss computation."""
        latents = sample_latent_data[:, :4, :4, :4, :4]
        
        losses = dit_teacher_model.compute_loss(latents)
        
        assert 'total_loss' in losses
        assert isinstance(losses['total_loss'], torch.Tensor)
        assert losses['total_loss'].dim() == 0  # Scalar
        assert torch.isfinite(losses['total_loss'])
        assert losses['total_loss'] >= 0
    
    def test_dit_teacher_sampling(self, dit_teacher_model, device):
        """Test teacher sampling capabilities."""
        B, T, C, H, W = 1, 4, 4, 4, 4
        shape = (B, T, C, H, W)
        
        # Test with fewer steps for speed
        samples = dit_teacher_model.sample(
            shape=shape,
            num_steps=5,  # Reduced for testing
            guidance_scale=1.0
        )
        
        assert samples.shape == shape
        assert torch.isfinite(samples).all()
    
    def test_teacher_embedding(self, device):
        """Test teacher embedding component."""
        latent_dim = 64
        d_model = 256
        max_seq_len = 128
        time_embed_dim = 128
        
        teacher_embed = TeacherEmbedding(
            latent_dim, d_model, max_seq_len, time_embed_dim
        ).to(device)
        
        B, T = 2, 8
        latent_tokens = torch.randn(B, T, latent_dim, device=device)
        timesteps = torch.randint(0, 100, (B,), device=device)
        
        embedded = teacher_embed(latent_tokens, timesteps)
        
        assert embedded.shape == (B, T, d_model)
        assert torch.isfinite(embedded).all()
    
    def test_teacher_timestep_embedding(self, device):
        """Test timestep embedding functionality."""
        time_embed_dim = 256
        
        teacher_embed = TeacherEmbedding(64, 512, 128, time_embed_dim).to(device)
        
        B = 4
        timesteps = torch.randint(0, 1000, (B,), device=device)
        
        time_emb = teacher_embed.get_timestep_embedding(timesteps)
        
        assert time_emb.shape == (B, time_embed_dim)
        assert torch.isfinite(time_emb).all()
        
        # Test that different timesteps give different embeddings
        t1 = torch.tensor([0], device=device)
        t2 = torch.tensor([500], device=device)
        
        emb1 = teacher_embed.get_timestep_embedding(t1)
        emb2 = teacher_embed.get_timestep_embedding(t2)
        
        assert not torch.allclose(emb1, emb2, atol=1e-6)


class TestModelInteractions:
    """Test interactions between different models."""
    
    def test_vae_dit_compatibility(self, vae_model, dit_student_model, sample_video_data, device):
        """Test compatibility between VAE and DiT models."""
        frames = sample_video_data[:, :4, :, :32, :32]
        
        # Encode with VAE
        mean, logvar = vae_model.encode(frames)
        latents = vae_model.reparameterize(mean, logvar)
        
        # Prepare for DiT
        B, T, C, H, W = latents.shape
        tokens = latents.view(B, T, -1)
        
        # Forward through DiT
        dit_output, _ = dit_student_model(tokens)
        
        # Reshape back to latent space
        dit_latents = dit_output.view(B, T, C, H, W)
        
        # Decode with VAE
        reconstructed = vae_model.decode(dit_latents)
        
        assert reconstructed.shape == frames.shape
        assert torch.isfinite(reconstructed).all()
    
    def test_teacher_student_distillation_compatibility(self, dit_teacher_model, dit_student_model, sample_latent_data, device):
        """Test teacher-student distillation compatibility."""
        latents = sample_latent_data[:, :4, :4, :4, :4]
        
        # Teacher forward pass
        losses = dit_teacher_model.compute_loss(latents)
        teacher_loss = losses['total_loss']
        
        # Student forward pass
        B, T, C, H, W = latents.shape
        student_tokens = latents.view(B, T, -1)
        student_output, _ = dit_student_model(student_tokens)
        
        # Compute simple student loss
        target = torch.randn_like(student_tokens)
        student_loss = F.mse_loss(student_output, target)
        
        # Both should be finite scalars
        assert torch.isfinite(teacher_loss)
        assert torch.isfinite(student_loss)
        assert teacher_loss.dim() == 0
        assert student_loss.dim() == 0
    
    @pytest.mark.slow
    def test_full_pipeline_forward_pass(self, vae_model, dit_student_model, sample_video_data, device):
        """Test full pipeline: VAE encode -> DiT process -> VAE decode."""
        frames = sample_video_data[:, :4, :, :32, :32]
        
        # Full forward pass
        with torch.no_grad():
            # 1. Encode frames
            mean, logvar = vae_model.encode(frames)
            latents = vae_model.reparameterize(mean, logvar)
            
            # 2. Process with DiT
            B, T, C, H, W = latents.shape
            tokens = latents.view(B, T, -1)
            processed_tokens, _ = dit_student_model(tokens)
            processed_latents = processed_tokens.view(B, T, C, H, W)
            
            # 3. Decode back to frames
            reconstructed = vae_model.decode(processed_latents)
        
        # Check final output
        assert reconstructed.shape == frames.shape
        assert torch.isfinite(reconstructed).all()
        
        # Should be different from input (model is processing)
        assert not torch.allclose(reconstructed, frames, atol=1e-3)


# Edge cases and error conditions
class TestModelEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_sequence_handling(self, dit_student_model, device):
        """Test handling of empty sequences."""
        # Empty sequence
        tokens = torch.empty(1, 0, 64, device=device)
        
        output, _ = dit_student_model(tokens)
        
        assert output.shape == (1, 0, 64)
    
    def test_single_token_sequence(self, dit_student_model, device):
        """Test handling of single token sequences."""
        tokens = torch.randn(1, 1, 64, device=device)
        
        output, _ = dit_student_model(tokens)
        
        assert output.shape == tokens.shape
        assert torch.isfinite(output).all()
    
    def test_model_device_consistency(self, vae_model, dit_student_model, device):
        """Test that models maintain device consistency."""
        # Check that all parameters are on the expected device
        for model in [vae_model, dit_student_model]:
            for param in model.parameters():
                assert param.device == device
    
    def test_model_parameter_count(self, vae_model, dit_student_model, dit_teacher_model):
        """Test parameter counting and model sizes."""
        from utils.model_utils import count_parameters
        
        # All models should have reasonable parameter counts
        vae_params = count_parameters(vae_model)
        student_params = count_parameters(dit_student_model)
        teacher_params = count_parameters(dit_teacher_model)
        
        assert vae_params > 0
        assert student_params > 0
        assert teacher_params > 0
        
        # Teacher and student should have similar parameter counts
        # (allowing for some difference in architecture)
        ratio = student_params / teacher_params
        assert 0.5 < ratio < 2.0, f"Parameter count ratio {ratio} seems unreasonable"