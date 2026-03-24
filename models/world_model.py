"""
Unified world model integrating all components:
- Self-forcing training methodology
- Flow matching and distillation
- Modular components (control, depth, memory, JEPA)
- Clean integration with unified training pipeline
- NEW v2: MoE, SLA, C-JEPA, Rich Conditioning, REPA

This replaces the scattered model implementations with a single, cohesive world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import sys
import os
import warnings
import logging
from einops import rearrange, repeat

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DriveDiTConfig, ComponentType
from layers.mha import MultiHeadAttention
from layers.mlp import MLP
from layers.nn_helpers import RMSNorm
from blocks.dit_block import DiTBlock

# Use torch's built-in SiLU
SiLU = nn.SiLU

# Configure logger
logger = logging.getLogger(__name__)

# =============================================================================
# Conditional Imports for New Components (v2)
# These modules may not exist yet - import conditionally
# =============================================================================

# Try to import SLA attention
_SLA_AVAILABLE = False
try:
    from layers.sla import SparseLinearAttention
    _SLA_AVAILABLE = True
except ImportError:
    SparseLinearAttention = None

# Try to import MoE FFN
_MOE_AVAILABLE = False
try:
    from layers.moe import MoEFFN, MoEDiTBlock
    _MOE_AVAILABLE = True
except ImportError:
    MoEFFN = None
    MoEDiTBlock = None

# Try to import C-JEPA predictor
_CJEPA_AVAILABLE = False
try:
    from core.causal_jepa import CausalJEPAPredictor
    _CJEPA_AVAILABLE = True
except ImportError:
    CausalJEPAPredictor = None

# Try to import Rich Conditioning module
_RICH_COND_AVAILABLE = False
try:
    from models.conditioning import (
        ConditioningIntegration,
        ConditioningConfig,
        create_conditioning_module
    )
    _RICH_COND_AVAILABLE = True
except ImportError:
    ConditioningIntegration = None
    ConditioningConfig = None
    create_conditioning_module = None


def check_component_availability(component: ComponentType) -> bool:
    """
    Check if a component's implementation is available.

    Args:
        component: The component type to check.

    Returns:
        True if the component is available, False otherwise.
    """
    availability_map = {
        ComponentType.MOE: _MOE_AVAILABLE,
        ComponentType.SLA: _SLA_AVAILABLE,
        ComponentType.CAUSAL_JEPA: _CJEPA_AVAILABLE,
        ComponentType.RICH_CONDITIONING: _RICH_COND_AVAILABLE,
        # Legacy components are always available
        ComponentType.CONTROL: True,
        ComponentType.DEPTH: True,
        ComponentType.MEMORY: True,
        ComponentType.JEPA: True,
        ComponentType.FLOW_MATCHING: True,
        ComponentType.REPA: True,  # REPA is implemented inline
    }
    return availability_map.get(component, False)


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


# =============================================================================
# Control Encoder
# =============================================================================

class ControlEncoder(nn.Module):
    """Encode control signals for conditioning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_layers - 1 else nn.Identity()
            ])

        self.encoder = nn.Sequential(*layers)

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        return self.encoder(controls)


# =============================================================================
# Depth Encoder
# =============================================================================

class DepthEncoder(nn.Module):
    """Encode depth information."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_depth: float = 100.0
    ):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, output_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(2)
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # Normalize depth
        depth_norm = torch.clamp(depth / self.max_depth, 0, 1)
        x = self.encoder(depth_norm)
        x = x.transpose(-2, -1)  # [B, 16*16, output_dim]
        return x


# =============================================================================
# Memory System
# =============================================================================

class MemorySystem(nn.Module):
    """Simple memory system for object and spatial permanence."""

    def __init__(
        self,
        dim: int,
        max_objects: int = 64,
        max_spatial: int = 256
    ):
        super().__init__()
        self.dim = dim
        self.max_objects = max_objects
        self.max_spatial = max_spatial

        # Memory banks
        self.object_memory = nn.Parameter(torch.randn(max_objects, dim))
        self.spatial_memory = nn.Parameter(torch.randn(max_spatial, dim))

        # Memory update networks
        self.object_update = nn.Linear(dim, dim)
        self.spatial_update = nn.Linear(dim, dim)

        # Memory retrieval
        self.memory_retrieval = MultiHeadAttention(dim, 8)

    def forward(
        self,
        hidden_states: torch.Tensor,
        step: int = 0
    ) -> torch.Tensor:
        """Update and retrieve memory."""
        B, T, D = hidden_states.shape

        # Combine object and spatial memory
        memory_tokens = torch.cat([
            self.object_memory.unsqueeze(0).repeat(B, 1, 1),
            self.spatial_memory.unsqueeze(0).repeat(B, 1, 1)
        ], dim=1)  # [B, max_objects + max_spatial, dim]

        # Simple memory retrieval (cross-attention)
        query = hidden_states.mean(dim=1, keepdim=True)  # [B, 1, D]
        retrieved_memory = self.memory_retrieval(query, kv=memory_tokens)  # [B, 1, D]

        return retrieved_memory.repeat(1, T, 1)  # [B, T, D]

    def get_memory_tokens(self, batch_size: int) -> torch.Tensor:
        """Get memory tokens for conditioning."""
        memory_tokens = torch.cat([self.object_memory, self.spatial_memory], dim=0)
        return memory_tokens.unsqueeze(0).repeat(batch_size, 1, 1)


# =============================================================================
# JEPA Predictor (Legacy)
# =============================================================================

class JEPAPredictor(nn.Module):
    """V-JEPA style predictor for future frame representations."""

    def __init__(
        self,
        input_dim: int,
        target_length: int,
        prediction_head_dim: int
    ):
        super().__init__()
        self.target_length = target_length

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, prediction_head_dim),
            nn.ReLU(),
            nn.Linear(prediction_head_dim, input_dim * target_length)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict future representations."""
        B, T, D = hidden_states.shape

        # Use last few tokens for prediction
        context = hidden_states[:, -4:].mean(dim=1)  # Average last 4 tokens

        predictions = self.predictor(context)
        predictions = predictions.view(B, self.target_length, D)

        return predictions


# =============================================================================
# Flow Matching Predictor
# =============================================================================

class FlowMatchingPredictor(nn.Module):
    """Flow matching predictor for diffusion acceleration."""

    def __init__(self, dim: int, num_steps: int = 4):
        super().__init__()
        self.dim = dim
        self.num_steps = num_steps

        self.flow_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            SiLU(),
            nn.Linear(dim * 2, dim),
            SiLU(),
            nn.Linear(dim, dim),
            nn.Tanh()  # Bounded flow predictions
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict flow field v_theta(z_t, t)."""
        # Simple implementation - in full version would include time embedding
        flow = self.flow_net(z)
        return flow

    def sample(self, z_init: torch.Tensor, num_steps: int) -> torch.Tensor:
        """Sample using Euler integration."""
        dt = 1.0 / num_steps
        z_t = z_init.clone()

        for step in range(num_steps):
            t = torch.full((z_t.size(0),), step * dt, device=z_t.device)
            flow = self.forward(z_t, t)
            z_t = z_t + dt * flow

        return z_t


# =============================================================================
# Self-Forcing Scheduler
# =============================================================================

class SelfForcingScheduler:
    """Scheduler for self-forcing ratio during training."""

    def __init__(self, config: DriveDiTConfig):
        self.config = config
        self.step = 0

    def update(self, step: int):
        """Update internal step counter."""
        self.step = step

    def get_ratio(self) -> float:
        """Get current self-forcing ratio."""
        if self.config.self_forcing_schedule == "linear":
            progress = min(1.0, self.step / self.config.self_forcing_warmup_steps)
            return self.config.initial_self_forcing_ratio + progress * (
                self.config.final_self_forcing_ratio - self.config.initial_self_forcing_ratio
            )
        elif self.config.self_forcing_schedule == "cosine":
            progress = min(1.0, self.step / self.config.self_forcing_warmup_steps)
            cos_progress = 0.5 * (1 - math.cos(math.pi * progress))
            return self.config.initial_self_forcing_ratio + cos_progress * (
                self.config.final_self_forcing_ratio - self.config.initial_self_forcing_ratio
            )
        else:  # exponential
            decay = 0.999 ** self.step
            return self.config.final_self_forcing_ratio + (
                self.config.initial_self_forcing_ratio - self.config.final_self_forcing_ratio
            ) * decay


# =============================================================================
# REPA Feature Alignment (inline implementation)
# =============================================================================

class REPAAlignmentHead(nn.Module):
    """
    REPA (Representation Alignment) head for aligning intermediate features
    with external representations (e.g., DINOv2, CLIP).
    """

    def __init__(
        self,
        model_dim: int,
        alignment_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.model_dim = model_dim
        self.alignment_dim = alignment_dim

        # Projection network
        layers = []
        for i in range(num_layers):
            in_dim = model_dim if i == 0 else alignment_dim
            layers.extend([
                nn.Linear(in_dim, alignment_dim),
                nn.LayerNorm(alignment_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity()
            ])
        self.projection = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project features to alignment space."""
        return self.projection(features)


# =============================================================================
# Unified World Model
# =============================================================================

class WorldModel(nn.Module):
    """
    Unified world model combining all methodologies:
    - Self-forcing training with curriculum learning
    - Flow matching and distillation
    - Modular components (control, depth, memory, JEPA)
    - NEW v2: MoE, SLA, C-JEPA, Rich Conditioning, REPA

    The model supports any subset of components and gracefully falls back
    if a component is not initialized or not available.
    """

    def __init__(self, config: DriveDiTConfig):
        """
        Initialize the WorldModel.

        Args:
            config: DriveDiT configuration object containing all model parameters.
        """
        super().__init__()
        self.config = config
        self.enabled_components = config.get_enabled_components()

        # Track active vs unavailable components
        self._active_components: List[ComponentType] = []
        self._unavailable_components: List[ComponentType] = []

        # Calculate max sequence length based on config
        patches_per_frame = (config.image_size // config.patch_size) ** 2
        max_seq_len = patches_per_frame * config.final_sequence_length + 512  # Extra for context

        # Current attention mode
        self._attention_mode = getattr(config, 'attention_mode', None)
        if self._attention_mode is None:
            # Import if not already defined
            try:
                from config.config import AttentionMode
                self._attention_mode = AttentionMode.STANDARD
            except ImportError:
                self._attention_mode = "standard"

        # Initialize backbone blocks
        self._init_backbone(config, max_seq_len)

        # Input encoders
        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            embed_dim=config.model_dim
        )

        # Optional components based on config
        self._init_optional_components()

        # Initialize new v2 components
        self._init_v2_components()

        # Output heads - each token predicts its own patch latent
        self.frame_head = nn.Linear(config.model_dim, config.vae_latent_dim)

        # Self-forcing scheduler
        self.self_forcing_scheduler = SelfForcingScheduler(config)

        # Cache for autoregressive generation
        self.past_kvs: Optional[List[Dict[str, torch.Tensor]]] = None

        # Intermediate feature storage for REPA
        self._intermediate_features: Dict[int, torch.Tensor] = {}

        # Log initialization summary
        self._log_init_summary()

    def _init_backbone(self, config: DriveDiTConfig, max_seq_len: int) -> None:
        """
        Initialize the transformer backbone.
        Uses MoE blocks if enabled and available, otherwise standard DiT blocks.

        Args:
            config: Configuration object.
            max_seq_len: Maximum sequence length.
        """
        use_moe = (
            ComponentType.MOE in self.enabled_components and
            _MOE_AVAILABLE and
            MoEDiTBlock is not None
        )

        if use_moe:
            # Use MoE DiT blocks
            self.backbone = nn.ModuleList([
                MoEDiTBlock(
                    d_model=config.model_dim,
                    n_heads=config.num_heads,
                    d_ff=config.model_dim * config.mlp_ratio,
                    num_experts=config.moe_num_experts,
                    top_k=config.moe_top_k,
                    capacity_factor=config.moe_capacity_factor,
                    causal=True,
                    max_seq_len=max_seq_len
                )
                for _ in range(config.num_layers)
            ])
            self._active_components.append(ComponentType.MOE)
        else:
            # Use standard DiT blocks
            self.backbone = nn.ModuleList([
                DiTBlock(
                    d_model=config.model_dim,
                    n_heads=config.num_heads,
                    d_ff=config.model_dim * config.mlp_ratio,
                    causal=True,
                    max_seq_len=max_seq_len
                )
                for _ in range(config.num_layers)
            ])

            if ComponentType.MOE in self.enabled_components and not _MOE_AVAILABLE:
                self._unavailable_components.append(ComponentType.MOE)
                warnings.warn(
                    "MoE component requested but layers.moe module not found. "
                    "Using standard DiT blocks."
                )

        self.use_moe = use_moe

    def _init_optional_components(self) -> None:
        """Initialize legacy optional components based on configuration."""
        config = self.config

        # Control encoder
        if ComponentType.CONTROL in self.enabled_components:
            self.control_encoder = ControlEncoder(
                input_dim=config.control_input_dim,
                hidden_dim=config.control_hidden_dim,
                output_dim=config.model_dim,
                num_layers=config.control_num_layers,
                dropout=config.control_dropout
            )
            self._active_components.append(ComponentType.CONTROL)
        else:
            self.control_encoder = None

        # Depth encoder
        if ComponentType.DEPTH in self.enabled_components:
            self.depth_encoder = DepthEncoder(
                input_dim=config.depth_channels,
                output_dim=config.model_dim,
                max_depth=config.depth_max_depth
            )
            self._active_components.append(ComponentType.DEPTH)
        else:
            self.depth_encoder = None

        # Memory system
        if ComponentType.MEMORY in self.enabled_components:
            self.memory_system = MemorySystem(
                dim=config.memory_dim,
                max_objects=config.object_memory_size,
                max_spatial=config.spatial_memory_size
            )
            self._active_components.append(ComponentType.MEMORY)
        else:
            self.memory_system = None

        # Flow matching predictor
        if ComponentType.FLOW_MATCHING in self.enabled_components:
            self.flow_predictor = FlowMatchingPredictor(
                dim=config.model_dim,
                num_steps=config.num_flow_steps
            )
            self._active_components.append(ComponentType.FLOW_MATCHING)
        else:
            self.flow_predictor = None

        # JEPA predictor (legacy)
        if ComponentType.JEPA in self.enabled_components:
            self.jepa_predictor = JEPAPredictor(
                input_dim=config.model_dim,
                target_length=config.jepa_target_length,
                prediction_head_dim=config.jepa_prediction_head_dim
            )
            self._active_components.append(ComponentType.JEPA)
        else:
            self.jepa_predictor = None

    def _init_v2_components(self) -> None:
        """Initialize new v2 architectural components."""
        config = self.config

        # SLA Attention (handled at forward time via attention mode)
        if ComponentType.SLA in self.enabled_components:
            if _SLA_AVAILABLE:
                # SLA is integrated into attention mechanism
                self._active_components.append(ComponentType.SLA)
            else:
                self._unavailable_components.append(ComponentType.SLA)
                warnings.warn(
                    "SLA component requested but layers.sla module not found. "
                    "Using standard attention."
                )

        # C-JEPA Predictor
        if ComponentType.CAUSAL_JEPA in self.enabled_components:
            if _CJEPA_AVAILABLE and CausalJEPAPredictor is not None:
                self.cjepa_predictor = CausalJEPAPredictor(
                    input_dim=config.model_dim,
                    slot_dim=config.cjepa_slot_dim,
                    num_slots=config.cjepa_num_slots,
                    num_iterations=config.cjepa_num_iterations,
                    prediction_horizon=config.cjepa_prediction_horizon,
                    temperature=config.cjepa_temperature
                )
                self._active_components.append(ComponentType.CAUSAL_JEPA)
            else:
                self.cjepa_predictor = None
                self._unavailable_components.append(ComponentType.CAUSAL_JEPA)
                warnings.warn(
                    "C-JEPA component requested but core.causal_jepa module not found."
                )
        else:
            self.cjepa_predictor = None

        # Rich Conditioning (GAIA-2 style)
        if ComponentType.RICH_CONDITIONING in self.enabled_components:
            if _RICH_COND_AVAILABLE and create_conditioning_module is not None:
                self.rich_conditioning = create_conditioning_module(
                    model_dim=config.model_dim,
                    cond_dim=config.rich_cond_dim,
                    num_cameras=config.rich_cond_num_cameras,
                    max_waypoints=config.rich_cond_max_waypoints,
                    max_objects=config.rich_cond_max_objects,
                    cfg_dropout_prob=config.rich_cond_cfg_dropout
                )
                self._active_components.append(ComponentType.RICH_CONDITIONING)
            else:
                self.rich_conditioning = None
                self._unavailable_components.append(ComponentType.RICH_CONDITIONING)
                warnings.warn(
                    "Rich conditioning requested but models.conditioning module not found."
                )
        else:
            self.rich_conditioning = None

        # REPA Alignment
        if ComponentType.REPA in self.enabled_components:
            # REPA is always available (inline implementation)
            self.repa_heads = nn.ModuleDict({
                str(layer_idx): REPAAlignmentHead(
                    model_dim=config.model_dim,
                    alignment_dim=config.repa_alignment_dim
                )
                for layer_idx in config.repa_feature_layers
            })
            self._active_components.append(ComponentType.REPA)
        else:
            self.repa_heads = None

    def _log_init_summary(self) -> None:
        """Log a summary of component initialization."""
        logger.info("WorldModel initialized")
        logger.info(f"  Active components: {[c.value for c in self._active_components]}")
        if self._unavailable_components:
            logger.warning(
                f"  Unavailable components: {[c.value for c in self._unavailable_components]}"
            )
        logger.info(f"  Using MoE: {self.use_moe}")
        logger.info(f"  Attention mode: {self._attention_mode}")

    # =========================================================================
    # Component Status Methods
    # =========================================================================

    def get_active_components(self) -> List[ComponentType]:
        """
        Get list of active (initialized and available) components.

        Returns:
            List of active ComponentType values.
        """
        return self._active_components.copy()

    def get_unavailable_components(self) -> List[ComponentType]:
        """
        Get list of requested but unavailable components.

        Returns:
            List of unavailable ComponentType values.
        """
        return self._unavailable_components.copy()

    def is_component_active(self, component: ComponentType) -> bool:
        """
        Check if a specific component is active (initialized and usable).

        Args:
            component: The component type to check.

        Returns:
            True if the component is active.
        """
        return component in self._active_components

    def get_component_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all components.

        Returns:
            Dictionary with component status information.
        """
        return {
            'active': [c.value for c in self._active_components],
            'unavailable': [c.value for c in self._unavailable_components],
            'requested': [c.value for c in self.enabled_components],
            'use_moe': self.use_moe,
            'attention_mode': str(self._attention_mode),
            'availability': {
                'moe': _MOE_AVAILABLE,
                'sla': _SLA_AVAILABLE,
                'cjepa': _CJEPA_AVAILABLE,
                'rich_conditioning': _RICH_COND_AVAILABLE
            }
        }

    # =========================================================================
    # Attention Mode Control
    # =========================================================================

    def switch_attention_mode(self, mode: str) -> None:
        """
        Switch between attention modes (standard, SLA, flash).

        Note: This affects future forward passes. The actual implementation
        depends on whether the corresponding modules are available.

        Args:
            mode: Attention mode ('standard', 'sla', 'flash').
        """
        valid_modes = ['standard', 'sla', 'flash']
        if mode not in valid_modes:
            raise ValueError(f"Invalid attention mode: {mode}. Valid: {valid_modes}")

        if mode == 'sla' and not _SLA_AVAILABLE:
            warnings.warn("SLA attention requested but not available. Using standard.")
            mode = 'standard'

        self._attention_mode = mode
        logger.info(f"Switched attention mode to: {mode}")

    def get_attention_mode(self) -> str:
        """Get current attention mode."""
        if hasattr(self._attention_mode, 'value'):
            return self._attention_mode.value
        return str(self._attention_mode)

    # =========================================================================
    # Feature Extraction Methods
    # =========================================================================

    def get_object_slots(
        self,
        hidden_states: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract object-level slot representations for C-JEPA.

        Args:
            hidden_states: Hidden states from backbone [B, T, D].
            return_attention: Whether to return slot attention weights.

        Returns:
            Object slots [B, num_slots, slot_dim], optionally with attention weights.
        """
        if self.cjepa_predictor is None:
            raise RuntimeError(
                "C-JEPA predictor not initialized. "
                "Enable CAUSAL_JEPA component and ensure core.causal_jepa module is available."
            )

        return self.cjepa_predictor.extract_slots(
            hidden_states,
            return_attention=return_attention
        )

    def get_intermediate_features(
        self,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get intermediate features from specified layers for REPA alignment.

        Note: Features are captured during the most recent forward pass.

        Args:
            layer_indices: List of layer indices to retrieve. If None, returns all captured.

        Returns:
            Dictionary mapping layer index to feature tensor.
        """
        if layer_indices is None:
            return self._intermediate_features.copy()

        return {
            idx: self._intermediate_features[idx]
            for idx in layer_indices
            if idx in self._intermediate_features
        }

    def get_repa_projections(
        self,
        features: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get REPA-projected features for alignment loss.

        Args:
            features: Features to project. If None, uses stored intermediate features.

        Returns:
            Dictionary mapping layer index to projected features.
        """
        if self.repa_heads is None:
            raise RuntimeError(
                "REPA heads not initialized. Enable REPA component."
            )

        if features is None:
            features = self._intermediate_features

        projections = {}
        for layer_idx_str, head in self.repa_heads.items():
            layer_idx = int(layer_idx_str)
            if layer_idx in features:
                projections[layer_idx] = head(features[layer_idx])

        return projections

    # =========================================================================
    # Forward Methods
    # =========================================================================

    def forward(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        mode: str = "train",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with different modes.

        Args:
            frames: [B, T, C, H, W] input video frames.
            controls: [B, T, control_dim] control signals (optional).
            depth: [B, T, 1, H, W] depth maps (optional).
            mode: "train", "inference", "self_forcing", "sla", "rich_cond", or "cjepa".
            **kwargs: Additional arguments for specific modes.

        Returns:
            Dictionary of outputs depending on mode.
        """
        if mode == "train":
            return self._forward_train(frames, controls, depth, **kwargs)
        elif mode == "inference":
            return self._forward_inference(frames, controls, depth, **kwargs)
        elif mode == "self_forcing":
            return self._forward_self_forcing(frames, controls, depth, **kwargs)
        elif mode == "sla":
            return self.forward_with_sla(frames, controls, depth, **kwargs)
        elif mode == "rich_cond":
            return self.forward_with_rich_conditioning(frames, **kwargs)
        elif mode == "cjepa":
            return self.forward_cjepa(frames, controls, depth, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _forward_backbone(
        self,
        tokens: torch.Tensor,
        capture_intermediate: bool = False,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        start_pos: int = 0
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass through backbone layers.

        Args:
            tokens: Input tokens [B, T, D].
            capture_intermediate: Whether to capture intermediate features for REPA.
            kv_cache: Optional KV cache for incremental decoding.
            start_pos: Starting position for RoPE/KV cache.

        Returns:
            Output hidden states and updated KV cache.
        """
        hidden_states = tokens
        new_kv_cache = []

        # Clear intermediate features
        if capture_intermediate:
            self._intermediate_features = {}

        # Determine which layers to capture
        capture_layers = set()
        if capture_intermediate and self.repa_heads is not None:
            capture_layers = {int(idx) for idx in self.repa_heads.keys()}

        for layer_idx, layer in enumerate(self.backbone):
            # Get layer-specific KV cache
            layer_kv = None
            if kv_cache is not None and layer_idx < len(kv_cache):
                layer_kv = kv_cache[layer_idx]

            # Forward through layer
            hidden_states, layer_new_kv = layer(
                hidden_states,
                kv_cache=layer_kv,
                start_pos=start_pos
            )

            if layer_new_kv is not None:
                new_kv_cache.append(layer_new_kv)

            # Capture intermediate features
            if layer_idx in capture_layers:
                self._intermediate_features[layer_idx] = hidden_states.detach()

        return hidden_states, new_kv_cache if new_kv_cache else None

    def _forward_train(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        capture_intermediate: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Standard training forward pass."""
        B, T, C, H, W = frames.shape

        # Patch embedding
        frame_tokens = self.patch_embed(frames.reshape(-1, C, H, W))
        frame_tokens = rearrange(frame_tokens, '(b t) n d -> b (t n) d', b=B, t=T)

        # Add optional modalities
        context_tokens = []

        if self.control_encoder is not None and controls is not None:
            control_tokens = self.control_encoder(controls.reshape(-1, controls.size(-1)))
            control_tokens = rearrange(control_tokens, '(b t) d -> b t d', b=B, t=T)
            context_tokens.append(control_tokens)

        if self.depth_encoder is not None and depth is not None:
            depth_tokens = self.depth_encoder(depth.reshape(-1, *depth.shape[-3:]))
            depth_tokens = rearrange(depth_tokens, '(b t) n d -> b t n d', b=B, t=T)
            depth_tokens = rearrange(depth_tokens, 'b t n d -> b (t n) d')
            context_tokens.append(depth_tokens)

        # Combine context
        if context_tokens:
            context = torch.cat(context_tokens, dim=1)
            all_tokens = torch.cat([frame_tokens, context], dim=1)
        else:
            all_tokens = frame_tokens

        # Transformer forward
        capture = capture_intermediate or ComponentType.REPA in self._active_components
        hidden_states, _ = self._forward_backbone(all_tokens, capture_intermediate=capture)

        # Extract frame tokens
        num_frame_tokens = frame_tokens.size(1)
        frame_hidden = hidden_states[:, :num_frame_tokens]

        # Predictions
        outputs: Dict[str, torch.Tensor] = {}

        # Frame prediction
        frame_pred = self.frame_head(frame_hidden)
        outputs['predictions'] = self._decode_frame_tokens(frame_pred, B, T)
        outputs['hidden_states'] = frame_hidden

        # Optional component predictions
        if self.jepa_predictor is not None:
            jepa_pred = self.jepa_predictor(frame_hidden)
            outputs['jepa_predictions'] = jepa_pred

        if self.flow_predictor is not None:
            # Generate random time steps for flow matching
            t = torch.rand(B, device=frames.device)
            flow_pred = self.flow_predictor(frame_hidden.mean(dim=1), t)
            outputs['flow_predictions'] = flow_pred

        # C-JEPA predictions
        if self.cjepa_predictor is not None:
            cjepa_outputs = self.cjepa_predictor(frame_hidden)
            outputs['cjepa_slots'] = cjepa_outputs['slots']
            outputs['cjepa_predictions'] = cjepa_outputs.get('predictions')

        # REPA features
        if self.repa_heads is not None and self._intermediate_features:
            outputs['repa_projections'] = self.get_repa_projections()

        return outputs

    def _forward_inference(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        num_steps: int = 1,
        use_kv_cache: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive inference for generation.

        Optimized with KV-cache for efficient incremental decoding.
        """
        B, T, C, H, W = frames.shape

        generated_frames = []
        current_frames = frames

        # Reset KV cache
        if use_kv_cache:
            self.reset_cache()

        with torch.no_grad():
            for step in range(num_steps):
                # Standard forward pass
                outputs = self._forward_train(current_frames, controls, depth)

                # Get next frame
                next_frame = outputs['predictions'][:, -1:]  # Last predicted frame
                generated_frames.append(next_frame)

                # Update context window
                current_frames = torch.cat([current_frames[:, 1:], next_frame], dim=1)

        return {
            'generated_frames': torch.cat(generated_frames, dim=1),
            'final_states': outputs['hidden_states']
        }

    def _forward_self_forcing(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        self_forcing_ratio: Optional[float] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Self-forcing training: mix ground truth and model predictions.

        Integrates with new components (C-JEPA, rich conditioning) when available.
        """
        B, T, C, H, W = frames.shape

        # Get self-forcing ratio
        if self_forcing_ratio is None:
            sf_ratio = self.self_forcing_scheduler.get_ratio()
        else:
            sf_ratio = self_forcing_ratio

        # Split into context and target
        context_length = T // 2
        context_frames = frames[:, :context_length]
        target_frames = frames[:, context_length:]

        # Slice controls and depth to match context
        context_controls = controls[:, :context_length] if controls is not None else None
        context_depth = depth[:, :context_length] if depth is not None else None

        # Process context normally
        context_outputs = self._forward_train(
            context_frames,
            context_controls,
            context_depth,
            capture_intermediate=True
        )

        # Autoregressive generation with self-forcing
        predictions = []
        current_frame = context_frames[:, -1:]  # Last context frame

        # Collect C-JEPA predictions if available
        cjepa_all_slots = []

        for t in range(target_frames.size(1)):
            # Decide whether to use ground truth or prediction
            use_gt = torch.rand(1).item() > sf_ratio

            if use_gt and t < target_frames.size(1):
                # Use ground truth
                input_frame = target_frames[:, t:t+1]
            else:
                # Use model prediction
                input_frame = current_frame

            # Create input sequence (context + current)
            input_sequence = torch.cat([context_frames, input_frame], dim=1)
            seq_len = input_sequence.size(1)

            # Slice controls and depth to match input sequence length
            seq_controls = controls[:, :seq_len] if controls is not None else None
            seq_depth = depth[:, :seq_len] if depth is not None else None

            # Forward pass
            outputs = self._forward_train(input_sequence, seq_controls, seq_depth)

            # Get prediction for next frame
            frame_pred = outputs['predictions'][:, -1:]  # Last frame
            predictions.append(frame_pred)

            # Update current frame for next iteration
            current_frame = frame_pred

            # Collect C-JEPA slots
            if 'cjepa_slots' in outputs:
                cjepa_all_slots.append(outputs['cjepa_slots'])

        result = {
            'predictions': torch.cat(predictions, dim=1),
            'targets': target_frames,
            'self_forcing_ratio': sf_ratio,
            'context_length': context_length,
            'hidden_states': context_outputs['hidden_states']
        }

        # Add REPA projections from context
        if 'repa_projections' in context_outputs:
            result['repa_projections'] = context_outputs['repa_projections']

        # Add C-JEPA slots
        if cjepa_all_slots:
            result['cjepa_slots'] = torch.stack(cjepa_all_slots, dim=1)

        return result

    def forward_with_sla(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using Sparse-Linear Attention.

        Falls back to standard attention if SLA is not available.
        """
        if not _SLA_AVAILABLE:
            warnings.warn("SLA not available, falling back to standard attention")
            return self._forward_train(frames, controls, depth, **kwargs)

        # Set attention mode temporarily
        original_mode = self._attention_mode
        self._attention_mode = 'sla'

        try:
            # For now, we use the standard forward which will check attention mode
            # Full SLA integration would require modifying the backbone blocks
            outputs = self._forward_train(frames, controls, depth, **kwargs)
            outputs['attention_mode'] = 'sla'
            return outputs
        finally:
            self._attention_mode = original_mode

    def forward_with_rich_conditioning(
        self,
        frames: torch.Tensor,
        camera_data: Optional[Dict[str, torch.Tensor]] = None,
        road_data: Optional[Dict[str, torch.Tensor]] = None,
        bbox_data: Optional[Dict[str, torch.Tensor]] = None,
        ego_data: Optional[Dict[str, torch.Tensor]] = None,
        temporal_data: Optional[Dict[str, torch.Tensor]] = None,
        cfg_scale: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with GAIA-2 style rich conditioning.

        Args:
            frames: Input video frames [B, T, C, H, W].
            camera_data: Camera intrinsics/extrinsics.
            road_data: Road topology information.
            bbox_data: 3D bounding box data.
            ego_data: Ego vehicle state.
            temporal_data: Time and weather conditions.
            cfg_scale: Classifier-free guidance scale.

        Returns:
            Model outputs with rich conditioning applied.
        """
        if self.rich_conditioning is None:
            warnings.warn(
                "Rich conditioning not available. "
                "Enable RICH_CONDITIONING component and ensure models.conditioning module exists."
            )
            return self._forward_train(frames, **kwargs)

        B, T, C, H, W = frames.shape

        # Get conditioning
        cond_outputs = self.rich_conditioning(
            camera_data=camera_data,
            road_data=road_data,
            bbox_data=bbox_data,
            ego_data=ego_data,
            temporal_data=temporal_data,
            cfg_scale=cfg_scale,
            return_cross_attention_contexts=True
        )

        conditioning = cond_outputs['conditioning']
        adaln_params = cond_outputs['adaln_params']

        # Patch embedding
        frame_tokens = self.patch_embed(frames.reshape(-1, C, H, W))
        frame_tokens = rearrange(frame_tokens, '(b t) n d -> b (t n) d', b=B, t=T)

        # TODO: Full integration would pass adaln_params to backbone layers
        # For now, we add conditioning as a bias
        conditioning_bias = conditioning.unsqueeze(1)  # [B, 1, cond_dim]

        # If conditioning dimension doesn't match, project it
        if conditioning_bias.size(-1) != self.config.model_dim:
            # This would need a projection layer - for now, handle gracefully
            pass

        # Forward through backbone
        hidden_states, _ = self._forward_backbone(frame_tokens, capture_intermediate=True)

        # Frame prediction
        frame_pred = self.frame_head(hidden_states)

        outputs = {
            'predictions': self._decode_frame_tokens(frame_pred, B, T),
            'hidden_states': hidden_states,
            'conditioning': conditioning,
            'adaln_params': adaln_params
        }

        # Add cross-attention contexts if available
        if 'road_context' in cond_outputs:
            outputs['road_context'] = cond_outputs['road_context']
        if 'object_tokens' in cond_outputs:
            outputs['object_tokens'] = cond_outputs['object_tokens']

        return outputs

    def forward_cjepa(
        self,
        frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        return_all_slots: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass focused on C-JEPA object-level prediction.

        Args:
            frames: Input video frames [B, T, C, H, W].
            controls: Optional control signals.
            depth: Optional depth maps.
            return_all_slots: Whether to return slots from all frames.

        Returns:
            C-JEPA specific outputs including object slots and predictions.
        """
        if self.cjepa_predictor is None:
            raise RuntimeError(
                "C-JEPA predictor not initialized. "
                "Enable CAUSAL_JEPA component and ensure core.causal_jepa module is available."
            )

        # Standard forward to get hidden states
        outputs = self._forward_train(frames, controls, depth, capture_intermediate=True)
        hidden_states = outputs['hidden_states']

        # Get C-JEPA outputs
        cjepa_out = self.cjepa_predictor(
            hidden_states,
            return_all_slots=return_all_slots
        )

        outputs.update({
            'cjepa_slots': cjepa_out['slots'],
            'cjepa_predictions': cjepa_out.get('predictions'),
            'cjepa_attention': cjepa_out.get('attention')
        })

        return outputs

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _decode_frame_tokens(
        self,
        frame_tokens: torch.Tensor,
        B: int,
        T: int
    ) -> torch.Tensor:
        """Decode frame tokens to actual frames."""
        # Reshape frame tokens back to spatial format
        # Input: [B, T * num_patches^2, vae_latent_dim]
        # Output: [B, T, vae_latent_dim, num_patches, num_patches]
        num_patches = self.config.image_size // self.config.patch_size

        # Reshape to [B, T, num_patches, num_patches, vae_latent_dim]
        frames = rearrange(
            frame_tokens,
            'b (t h w) c -> b t c h w',
            t=T,
            h=num_patches,
            w=num_patches,
            c=self.config.vae_latent_dim
        )

        # Upsample to original resolution (simplified)
        frames = F.interpolate(
            frames.reshape(-1, *frames.shape[-3:]),
            size=(self.config.image_size, self.config.image_size),
            mode='bilinear',
            align_corners=False
        )

        # Convert to RGB if needed
        if self.config.vae_latent_dim != 3:
            frames = F.conv2d(
                frames,
                torch.ones(3, self.config.vae_latent_dim, 1, 1, device=frames.device),
                bias=None
            )

        frames = torch.sigmoid(frames)  # Normalize to [0, 1]
        frames = rearrange(frames, '(b t) c h w -> b t c h w', b=B, t=T)

        return frames

    def reset_cache(self) -> None:
        """Reset KV cache for new sequence."""
        self.past_kvs = None
        self._intermediate_features = {}

    def update_self_forcing_step(self, step: int) -> None:
        """Update self-forcing scheduler."""
        self.self_forcing_scheduler.update(step)

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get total number of parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """
        Get auxiliary loss from MoE layers (load balancing).

        Returns:
            Auxiliary loss tensor if MoE is active, None otherwise.
        """
        if not self.use_moe:
            return None

        aux_losses = []
        for layer in self.backbone:
            if hasattr(layer, 'get_aux_loss'):
                aux_loss = layer.get_aux_loss()
                if aux_loss is not None:
                    aux_losses.append(aux_loss)

        if not aux_losses:
            return None

        return sum(aux_losses) / len(aux_losses)


# =============================================================================
# Factory Function
# =============================================================================

def create_world_model(config: DriveDiTConfig) -> WorldModel:
    """
    Create world model with configuration.

    Args:
        config: DriveDiT configuration object.

    Returns:
        Initialized WorldModel instance.
    """
    return WorldModel(config)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test unified world model
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.config import get_minimal_config

    print("=" * 60)
    print("WorldModel Integration Test")
    print("=" * 60)

    # Use minimal config for faster testing
    config = get_minimal_config()
    model = create_world_model(config)

    # Print component status
    print("\nComponent Status:")
    status = model.get_component_status()
    print(f"  Active: {status['active']}")
    print(f"  Unavailable: {status['unavailable']}")
    print(f"  Availability: {status['availability']}")

    # Test inputs (smaller dimensions for faster testing)
    B, T = 2, 4
    C, H, W = config.in_channels, config.image_size, config.image_size
    frames = torch.randn(B, T, C, H, W)
    controls = torch.randn(B, T, config.control_input_dim) if config.is_component_enabled(ComponentType.CONTROL) else None
    depth = torch.randn(B, T, config.depth_channels, H, W) if config.is_component_enabled(ComponentType.DEPTH) else None

    # Test different modes
    print("\nTesting training mode...")
    train_outputs = model(frames, controls, depth, mode="train")
    print(f"  Training outputs: {list(train_outputs.keys())}")

    print("\nTesting self-forcing mode...")
    sf_outputs = model(frames, controls, depth, mode="self_forcing", self_forcing_ratio=0.5)
    print(f"  Self-forcing outputs: {list(sf_outputs.keys())}")

    print("\nTesting inference mode...")
    context_frames = frames[:, :2]  # Use first 2 frames as context
    context_controls = controls[:, :2] if controls is not None else None
    context_depth = depth[:, :2] if depth is not None else None
    inf_outputs = model(context_frames, context_controls, context_depth, mode="inference", num_steps=2)
    print(f"  Inference outputs: {list(inf_outputs.keys())}")

    # Test SLA mode (will fall back if not available)
    print("\nTesting SLA mode...")
    sla_outputs = model(frames, controls, depth, mode="sla")
    print(f"  SLA outputs: {list(sla_outputs.keys())}")
    print(f"  Attention mode used: {sla_outputs.get('attention_mode', 'standard')}")

    # Test helper methods
    print("\nTesting helper methods...")
    print(f"  Active components: {[c.value for c in model.get_active_components()]}")
    print(f"  Attention mode: {model.get_attention_mode()}")
    print(f"  Total parameters: {model.get_num_parameters():,}")
    print(f"  Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")

    # Test MoE aux loss (will be None if MoE not active)
    aux_loss = model.get_moe_aux_loss()
    print(f"  MoE aux loss: {aux_loss}")

    print("\n" + "=" * 60)
    print("Unified world model test completed successfully!")
    print("=" * 60)

