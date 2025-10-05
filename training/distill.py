"""
Distillation training implementation for DriveDiT.
KL + flow matching distillation from bidirectional teacher to causal student.
Implements 4→1 step distillation for efficient inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import wandb
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dit_teacher import DiTTeacher
from models.dit_student import DiTStudent
from models.vae3d import VAE3D
from blocks.flow_match import FlowMatchingSampler, FlowLoss, ConsistencyTraining


class DistillationTrainer:
    """
    Distillation trainer for teaching causal student from bidirectional teacher.
    
    Key components:
    1. Flow matching distillation (4→1 step)
    2. KL divergence on latent distributions
    3. Consistency training across noise levels
    4. Multi-scale feature matching
    """
    
    def __init__(
        self,
        teacher_model: DiTTeacher,
        student_model: DiTStudent,
        vae_model: VAE3D,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        use_wandb: bool = True,
        distill_config: Optional[Dict] = None
    ):
        self.teacher_model = teacher_model.eval()  # Teacher in eval mode
        self.student_model = student_model
        self.vae_model = vae_model.eval()  # VAE frozen during distillation
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_wandb = use_wandb
        
        # Distillation configuration
        default_config = {
            'flow_steps': 4,
            'sigma_min': 0.002,
            'sigma_max': 80.0,
            'distill_weight': 1.0,
            'kl_weight': 0.1,
            'consistency_weight': 0.1,
            'feature_weight': 0.05,
            'guidance_scale': 1.0,
            'temperature': 1.0,
            'clip_grad_norm': 1.0
        }
        self.config = {**default_config, **(distill_config or {})}
        
        # Loss components
        self.flow_loss = FlowLoss(loss_type='mse', reduction='mean')
        self.consistency_loss = ConsistencyTraining(
            num_scales=4,
            consistency_weight=self.config['consistency_weight']
        )
        
        # Flow sampler for teacher trajectory
        self.flow_sampler = FlowMatchingSampler(
            num_steps=self.config['flow_steps'],
            sigma_min=self.config['sigma_min'],
            sigma_max=self.config['sigma_max'],
            guidance_scale=self.config['guidance_scale']
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Freeze teacher and VAE
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        for param in self.vae_model.parameters():
            param.requires_grad = False
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train student for one epoch using distillation.
        
        Args:
            dataloader: Training data with video sequences
        
        Returns:
            Dictionary of average losses for the epoch
        """
        self.student_model.train()
        epoch_losses = defaultdict(list)
        
        progress_bar = tqdm(dataloader, desc=f"Distillation Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            frames = batch['frames'].to(self.device)  # [B, T, 3, H, W]
            controls = batch.get('controls', None)
            if controls is not None:
                controls = controls.to(self.device)  # [B, T, 4]
            
            # Distillation step
            losses = self._distillation_step(frames, controls)
            
            # Compute total loss
            total_loss = self._compute_total_loss(losses)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), 
                    self.config['clip_grad_norm']
                )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            for key, value in losses.items():
                epoch_losses[key].append(value.item() if torch.is_tensor(value) else value)
            
            if self.use_wandb and self.global_step % 100 == 0:
                self._log_metrics(losses, total_loss)
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'distill': f"{losses.get('distillation', 0):.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        self.epoch += 1
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def _distillation_step(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one distillation training step.
        
        Args:
            frames: Video frames [B, T, 3, H, W]
            controls: Control signals [B, T, 4]
        
        Returns:
            Dictionary of losses
        """
        B, T, C, H, W = frames.shape
        
        # Encode frames to latents using frozen VAE
        with torch.no_grad():
            # Reshape for VAE: [B, T, C, H, W] -> [B, C, T, H, W]
            frames_vae = frames.permute(0, 2, 1, 3, 4)
            mean, logvar = self.vae_model.encode(frames_vae)
            latents = self.vae_model.reparameterize(mean, logvar)
            # Back to [B, T, C, H//8, W//8]
            latents = latents.permute(0, 2, 1, 3, 4)
        
        losses = {}
        
        # 1. Flow Matching Distillation
        flow_losses = self._compute_flow_distillation(latents, controls)
        losses.update(flow_losses)
        
        # 2. KL Divergence on latent distributions
        kl_loss = self._compute_kl_distillation(latents, controls)
        losses['kl_divergence'] = kl_loss
        
        # 3. Consistency training
        consistency_loss = self._compute_consistency_loss(latents, controls)
        losses['consistency'] = consistency_loss
        
        # 4. Feature matching (optional)
        if self.config['feature_weight'] > 0:
            feature_loss = self._compute_feature_matching(latents, controls)
            losses['feature_matching'] = feature_loss
        
        return losses
    
    def _compute_flow_distillation(
        self,
        latents: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching distillation loss.
        Teacher uses 4 steps, student learns to match in 1 step.
        
        Args:
            latents: Latent representations [B, T, C, H, W]
            controls: Control signals [B, T, 4]
        
        Returns:
            Dictionary of flow-related losses
        """
        B, T, C, H, W = latents.shape
        device = latents.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.teacher_model.num_diffusion_steps,
            (B,), device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Create noisy latents
        with torch.no_grad():
            # Get noise schedule
            sigmas = self.flow_sampler.sigmas
            sigma_t = sigmas[timesteps].view(B, 1, 1, 1, 1)
            noisy_latents = latents + sigma_t * noise
        
        # Teacher flow prediction (bidirectional)
        with torch.no_grad():
            # Reshape for teacher: [B, T, C, H, W] -> [B*T, C, H, W]
            teacher_input = noisy_latents.view(B*T, C, H, W)
            teacher_timesteps = timesteps.repeat_interleave(T)
            
            teacher_flow = self.teacher_model(
                noisy_latents=teacher_input.unsqueeze(1),  # Add seq dim
                timesteps=teacher_timesteps
            ).squeeze(1)  # Remove seq dim
            
            # Reshape back: [B*T, C, H, W] -> [B, T, C, H, W]
            teacher_flow = teacher_flow.view(B, T, C, H, W)
        
        # Student flow prediction (causal)
        # Prepare student input tokens
        student_tokens = self._prepare_student_tokens(noisy_latents, controls)
        
        student_output, _ = self.student_model(
            tokens=student_tokens,
            use_cache=False
        )
        
        # Reshape student output to match latent dimensions
        student_flow = student_output.view(B, T, C, H, W)
        
        # Flow matching loss
        flow_loss = self.flow_loss(
            flow_pred=student_flow,
            z_i=noisy_latents,
            z_next=latents,  # Target is clean latents
            d_sigma=sigma_t.squeeze()
        )
        
        return {'flow_matching': flow_loss}
    
    def _compute_kl_distillation(
        self,
        latents: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence between teacher and student distributions.
        
        Args:
            latents: Clean latent representations [B, T, C, H, W]
            controls: Control signals [B, T, 4]
        
        Returns:
            KL divergence loss
        """
        B, T, C, H, W = latents.shape
        
        # Sample context and target split
        context_len = T // 2
        context_latents = latents[:, :context_len]
        target_latents = latents[:, context_len:]
        
        if controls is not None:
            context_controls = controls[:, :context_len]
            target_controls = controls[:, context_len:]
        else:
            context_controls = target_controls = None
        
        # Teacher predictions (bidirectional)
        with torch.no_grad():
            teacher_tokens = self._prepare_teacher_tokens(latents, controls)
            teacher_logits = self.teacher_model(
                noisy_latents=latents,
                timesteps=torch.zeros(B, device=latents.device).long()
            )
            
            # Convert to log probabilities
            teacher_log_probs = F.log_softmax(
                teacher_logits.flatten(2) / self.config['temperature'], 
                dim=-1
            )
        
        # Student predictions (causal)
        student_tokens = self._prepare_student_tokens(context_latents, context_controls)
        student_logits, _ = self.student_model(
            tokens=student_tokens,
            use_cache=False
        )
        
        # Convert to log probabilities
        student_log_probs = F.log_softmax(
            student_logits.flatten(2) / self.config['temperature'],
            dim=-1
        )
        
        # KL divergence
        teacher_probs = F.softmax(teacher_log_probs, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        )
        
        return kl_loss
    
    def _compute_consistency_loss(
        self,
        latents: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute consistency loss across different noise levels.
        
        Args:
            latents: Clean latent representations [B, T, C, H, W]
            controls: Control signals [B, T, 4]
        
        Returns:
            Consistency loss
        """
        # Prepare tokens for consistency training
        student_tokens = self._prepare_student_tokens(latents, controls)
        
        # Use consistency training module
        consistency_loss = self.consistency_loss(
            flow_model=lambda x, sigma, ctx: self.student_model(x)[0],
            z=student_tokens,
            context=controls
        )
        
        return consistency_loss
    
    def _compute_feature_matching(
        self,
        latents: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute feature matching loss between teacher and student hidden states.
        
        Args:
            latents: Latent representations [B, T, C, H, W]
            controls: Control signals [B, T, 4]
        
        Returns:
            Feature matching loss
        """
        B, T, C, H, W = latents.shape
        
        # Teacher features (bidirectional)
        with torch.no_grad():
            teacher_input = latents.view(B*T, C, H, W).unsqueeze(1)
            teacher_timesteps = torch.zeros(B*T, device=latents.device).long()
            
            # Get intermediate features from teacher
            teacher_features = []
            x = self.teacher_model.embed(teacher_input.squeeze(1), teacher_timesteps)
            
            for block in self.teacher_model.blocks[::4]:  # Sample every 4th layer
                x, _ = block(x)
                teacher_features.append(x.detach())
        
        # Student features (causal)
        student_tokens = self._prepare_student_tokens(latents, controls)
        student_features = []
        
        # Forward through student with feature collection
        _, hidden_states = self.student_model(
            tokens=student_tokens,
            return_hidden=True
        )
        
        # Sample corresponding layers
        for i in range(0, len(hidden_states), len(hidden_states)//len(teacher_features)):
            if i < len(hidden_states):
                student_features.append(hidden_states[i])
        
        # Compute feature matching loss
        feature_loss = 0.0
        for teacher_feat, student_feat in zip(teacher_features, student_features):
            # Match feature dimensions if needed
            if teacher_feat.shape != student_feat.shape:
                min_seq = min(teacher_feat.size(1), student_feat.size(1))
                teacher_feat = teacher_feat[:, :min_seq]
                student_feat = student_feat[:, :min_seq]
            
            feature_loss += F.mse_loss(student_feat, teacher_feat)
        
        return feature_loss / len(teacher_features)
    
    def _prepare_student_tokens(
        self,
        latents: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prepare input tokens for student model.
        
        Args:
            latents: Latent representations [B, T, C, H, W]
            controls: Control signals [B, T, 4]
        
        Returns:
            Student input tokens [B, seq_len, latent_dim]
        """
        B, T, C, H, W = latents.shape
        
        # Flatten spatial dimensions
        latent_tokens = latents.view(B, T, -1)
        
        if controls is not None:
            # Combine latent and control tokens
            # For simplicity, concatenate along feature dimension
            control_expanded = controls.unsqueeze(-1).expand(-1, -1, latent_tokens.size(-1) // 4)
            tokens = torch.cat([latent_tokens, control_expanded], dim=-1)
        else:
            tokens = latent_tokens
        
        return tokens
    
    def _prepare_teacher_tokens(
        self,
        latents: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prepare input tokens for teacher model.
        """
        # Similar to student preparation but for teacher format
        return self._prepare_student_tokens(latents, controls)
    
    def _compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Combine all losses with appropriate weights.
        
        Args:
            losses: Dictionary of individual losses
        
        Returns:
            Total weighted loss
        """
        total_loss = 0.0
        
        # Main distillation losses
        if 'flow_matching' in losses:
            total_loss += self.config['distill_weight'] * losses['flow_matching']
        
        if 'kl_divergence' in losses:
            total_loss += self.config['kl_weight'] * losses['kl_divergence']
        
        if 'consistency' in losses:
            total_loss += self.config['consistency_weight'] * losses['consistency']
        
        if 'feature_matching' in losses:
            total_loss += self.config['feature_weight'] * losses['feature_matching']
        
        return total_loss
    
    def _log_metrics(self, losses: Dict, total_loss: torch.Tensor) -> None:
        """Log training metrics to wandb."""
        if not self.use_wandb:
            return
        
        log_dict = {
            'distill/total_loss': total_loss.item(),
            'distill/learning_rate': self.optimizer.param_groups[0]['lr'],
            'distill/epoch': self.epoch,
            'distill/global_step': self.global_step
        }
        
        for key, value in losses.items():
            log_dict[f'distill/{key}'] = value.item() if torch.is_tensor(value) else value
        
        wandb.log(log_dict, step=self.global_step)
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate student model on held-out data.
        
        Args:
            val_dataloader: Validation data loader
        
        Returns:
            Dictionary of validation metrics
        """
        self.student_model.eval()
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                frames = batch['frames'].to(self.device)
                controls = batch.get('controls', None)
                if controls is not None:
                    controls = controls.to(self.device)
                
                # Compute validation losses
                losses = self._distillation_step(frames, controls)
                total_loss = self._compute_total_loss(losses)
                
                # Record metrics
                val_metrics['total_loss'].append(total_loss.item())
                for key, value in losses.items():
                    val_metrics[key].append(value.item() if torch.is_tensor(value) else value)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        if self.use_wandb:
            wandb.log({f'val_distill/{k}': v for k, v in avg_metrics.items()}, 
                     step=self.global_step)
        
        self.student_model.train()
        return avg_metrics
    
    def save_checkpoint(self, filepath: str, best_metric: Optional[float] = None):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            best_metric: Best validation metric achieved
        """
        checkpoint = {
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_metric': best_metric
        }
        
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from {filepath}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


def test_distillation():
    """Test function for distillation trainer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    teacher = DiTTeacher(
        latent_dim=8,
        d_model=256,
        n_layers=4,
        n_heads=8,
        num_diffusion_steps=100
    ).to(device)
    
    student = DiTStudent(
        latent_dim=8,
        d_model=256,
        n_layers=4,
        n_heads=8
    ).to(device)
    
    vae = VAE3D(
        in_channels=3,
        latent_dim=8,
        hidden_dims=[32, 64, 128],
        num_res_blocks=1
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    # Create trainer
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        vae_model=vae,
        optimizer=optimizer,
        device=device,
        use_wandb=False
    )
    
    # Test data
    B, T, C, H, W = 2, 8, 3, 64, 64
    frames = torch.randn(B, T, C, H, W).to(device)
    controls = torch.randn(B, T, 4).to(device)
    
    # Test distillation step
    losses = trainer._distillation_step(frames, controls)
    total_loss = trainer._compute_total_loss(losses)
    
    print("Distillation test:")
    print(f"Losses: {[(k, v.item()) for k, v in losses.items()]}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Test backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("Backward pass successful!")


if __name__ == "__main__":
    test_distillation()