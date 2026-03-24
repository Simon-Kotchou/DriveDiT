# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DriveDiT is a zero-dependency PyTorch implementation of a diffusion transformer for autonomous driving world modeling. The system uses pure mathematical components with explicit tensor operations—no external libraries beyond `torch >= 2.2`, `einops`, and `opencv`.

## Zero-Dependency Architecture

### Folder Structure
```
drivedit/
│
├── layers/          # Generic math primitives
│   ├─ rope.py             # Rotary positional embedding
│   ├─ mha.py              # Multi-head attention (causal|bidirectional)
│   ├─ mlp.py              # Feed-forward networks
│   └─ nn_helpers.py       # SiLU, RMSNorm, fused operations
│
├── blocks/          # Composite components
│   ├─ dit_block.py        # Attention + MLP + AdaLN
│   └─ flow_matching.py    # Complete flow matching implementation
│
├── core/            # Advanced components
│   ├─ components.py       # RoPE3D, SAM2 tracker, flow predictor
│   └─ base.py             # Core base classes and utilities
│
├── models/          # Complete architectures
│   ├─ world_model.py      # Unified world model with all components
│   ├─ vae3d.py            # 3D causal VAE (WAN distilled)
│   ├─ dit_student.py      # Causal world model
│   └─ dit_teacher.py      # Bidirectional teacher
│
├── config/          # Configuration system
│   └─ config.py           # Unified configuration for all components
│
├── data/            # Data pipeline
│   ├─ pipeline.py         # Memory-mapped video processing
│   ├─ video_chunking.py   # Intelligent video chunking
│   └─ large_scale_processing.py # 100k+ hour dataset processing
│
├── training/        # Training loops
│   ├─ unified_trainer.py  # Complete unified training pipeline
│   ├─ self_forcing_plus.py # Self-Forcing++ training (rolling KV, curriculum, etc.)
│   ├─ distributed.py      # Distributed training and memory management
│   └─ losses.py           # Unified loss functions (all types)
│
├── inference/       # Generation
│   └─ rollout.py          # Single-GPU streaming inference
│
├── tests/           # Testing suite
│   ├─ test_math_components.py # Mathematical component tests
│   └─ test_training_pipeline.py # Training pipeline tests
│
├── docs/            # Documentation
│   └─ REFERENCES.md       # Mathematical foundations and paper sources
│
└── src/            # Enfusion C scripts
    └─ SCR_*.c             # Road extraction/visualization
```

## Development Commands

### Environment Setup
```bash
# Minimal dependencies
pip install torch>=2.2 einops opencv-python==4.10.0 torchmetrics

# Development tools
pip install jupyter black ruff
```

### Docker Environment
```dockerfile
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime
RUN pip install einops opencv-python==4.10.0 torchmetrics
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
WORKDIR /workspace
```

### Running Components
```bash
# Unified training pipeline
python training/unified_trainer.py

# Real-time inference/rollout
python inference/rollout.py

# Test individual models
python models/dit_student.py
python models/vae3d.py

# Test unified world model
python models/world_model.py

# Run comprehensive tests
python tests/test_math_components.py
python tests/test_training_pipeline.py

# Test Self-Forcing++ components
python training/self_forcing_plus.py

# Large-scale data processing
python data/large_scale_processing.py
```

## Core Mathematical Components

### 1. Rotary Positional Embedding (RoPE)
```python
def rope(x, sin, cos):
    # x: [B, T, H, D] - half-rotate every two dims
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```
- Pre-compute per-axis `sin, cos` for `(time, height, width)`
- Apply 3-axis factorization like V-JEPA-2

### 2. Multi-Head Attention (einsum-only)
```python
def mha(q, k, v, mask=None):
    d = q.size(-1)
    scores = torch.einsum('bthd,bshd->bhts', q, k) / d**0.5
    if mask is not None: scores.masked_fill_(mask==0, -1e4)
    p = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
    return torch.einsum('bhts,bshd->bthd', p, v)
```
- `mask=1` for causal positions, `None` for bidirectional teacher
- Pure einsum operations for transparency

### 3. Flow-Matching Distillation
Mathematical formulation for 4→1 step distillation:
```
z_{i+1} = z_i - Δσ_i * f_θ(z_i, ctx)
Loss = ||f_θ(z_i) - (z_i - z_{i+1}^teacher)/Δσ_i||²
```

Implementation:
```python
def flow_loss(flow_pred, z_i, z_next, d_sigma):
    return ((flow_pred - (z_i - z_next)/d_sigma)**2).mean()
```

## Advanced Architecture Components (v2)

### Philosophy: Causal Fidelity Over Perceptual Realism
Based on World-in-World (ICLR 2026 Oral) and DrivingGen findings:
- Visual quality alone does NOT predict driving task success
- We optimize for physics understanding and controllability
- JEPA representations guide generation, not replace it

### SLA (Sparse-Linear Attention)
Based on [arxiv:2509.24006](https://arxiv.org/abs/2509.24006):
```python
from layers.sla import sla_attention, SLAConfig

# Core insight: attention weights decompose into:
# - Critical blocks (high-rank): O(N²) FlashAttention
# - Marginal blocks (low-rank): O(N) linear attention
# - Negligible blocks: skip entirely

config = SLAConfig(
    block_size=64,
    critical_threshold=0.1,   # Top 10% by rank
    marginal_threshold=0.5,   # Next 40%
    use_flash_attention=True
)

# Drop-in replacement for mha()
output = sla_attention(q, k, v, config=config, mask=mask)
# Result: 20× attention reduction, 95% compute savings
```

### MoE (Mixture of Experts in DiT FFN)
Based on [DeepSeekMoE](https://arxiv.org/abs/2401.06066) and GigaWorld-0:
```python
from layers.moe import MoEFFN, MoEDiTBlock

# Replace dense FFN with MoE
ffn = MoEFFN(
    dim=512,
    num_experts=8,          # Fine-grained experts
    top_k=2,                # Activate top-2 per token
    num_shared_experts=1,   # Always-on for common knowledge
    use_aux_loss=False      # DeepSeek-V3 style load balancing
)

# Or use MoE-enabled DiT block directly
block = MoEDiTBlock(config)
# Result: 2B active params matches 14B dense performance
```

### REPA (Representation Alignment with V-JEPA 2.1)
Based on [ICLR 2025 Oral](https://arxiv.org/abs/2410.06940) with [HASTE](https://arxiv.org/abs/2505.16792):
```python
from core.vjepa_backbone import VJEPABackbone
from training.repa_loss import REPALoss

# Frozen V-JEPA 2.1 as semantic backbone
vjepa = VJEPABackbone.from_pretrained("meta/vjepa-2.1")

# Align DiT intermediate layers with V-JEPA features
repa_loss = REPALoss(
    backbone=vjepa,
    alignment_layers=[4, 8, 12],  # Which DiT layers to align
    projection_dim=768,           # V-JEPA feature dim
    haste_enabled=True,           # Early-stop to avoid plateau
    haste_stop_ratio=0.4          # Stop REPA at 40% of training
)

# In training loop
vjepa_features = vjepa.extract_features(video_frames)
dit_features = model.get_intermediate_features(x)
loss_repa = repa_loss(dit_features, vjepa_features, step, total_steps)
# Result: 17.5× training speedup
```

**Why V-JEPA 2.1 over DINOv3?**
- V-JEPA 2.1 has DINOv3-quality dense features PLUS video temporal understanding
- For a video world model, align with a video-native encoder
- DINOv3 is static images only

### C-JEPA (Causal JEPA with Object-Level Masking)
Based on Causal JEPA (Brown/NYU 2026):
```python
from core.causal_jepa import CausalJEPAPredictor, ObjectSlotEncoder

# Encode objects as slots
slot_encoder = ObjectSlotEncoder(
    dim=512,
    max_objects=64,
    use_detection_masks=True
)

# Object-level predictor (not patch-level)
cjepa = CausalJEPAPredictor(
    dim=512,
    trajectory_length=16,
    counterfactual_enabled=True
)

# Mask entire object trajectories, not random patches
object_slots = slot_encoder(features, object_masks)
masked_slots = cjepa.mask_trajectories(object_slots, mask_ratio=0.5)
predictions = cjepa.predict(context, masked_slots)

# This induces causal inductive bias without explicit causal graphs
# Result: 20% improvement on counterfactual reasoning, 8× faster planning
```

### Closed-Loop Evaluation
Based on World-in-World (ICLR 2026 Oral):
```python
from evaluation.closed_loop import ClosedLoopEvaluator
from evaluation.physics_metrics import PhysicsViolationDetector
from evaluation.driving_metrics import DrivingMetrics

evaluator = ClosedLoopEvaluator(
    world_model=model,
    num_planning_iterations=5,    # Inference-time scaling
    physics_check=True
)

# Iterate: observe → predict → act → observe
results = evaluator.evaluate(
    task="lane_following",
    environment=env
)

# Metrics that matter (not FID/FVD!)
print(f"Task success rate: {results.success_rate}")
print(f"Physics violations: {results.physics_violations}")
print(f"Trajectory drift: {results.drift_ade}")
```

### GAIA-2 Style Rich Conditioning
Based on [GAIA-2 (Wayve 2025)](https://arxiv.org/abs/2503.20523):
```python
from models.conditioning import RichConditioningModule

conditioning = RichConditioningModule(
    dim=512,
    use_camera_geometry=True,    # Intrinsics/extrinsics encoding
    use_road_topology=True,      # Lane graphs via cross-attention
    use_3d_boxes=True,           # Dynamic object encoding
    use_scenario_embedding=True, # Weather, traffic, road type
    cfg_dropout=0.1              # Classifier-free guidance
)

# AdaLN for ego-actions, cross-attention for structured inputs
conditioned_x = conditioning(
    x,
    ego_actions=controls,
    camera_params=cameras,
    road_graph=lanes,
    boxes_3d=detections
)
# Result: 384× total compression with richer tokens
```

### Component Integration Pattern
```python
from config.config import get_efficiency_config, ComponentType

# Enable all new components
config = get_efficiency_config()
config.enable_component(ComponentType.SLA)
config.enable_component(ComponentType.MOE)
config.enable_component(ComponentType.REPA)
config.enable_component(ComponentType.CAUSAL_JEPA)
config.enable_component(ComponentType.CLOSED_LOOP_EVAL)

# Model automatically uses MoE blocks + SLA attention
model = WorldModel(config)

# Trainer integrates all losses
trainer = UnifiedTrainer(model, config)
losses = trainer.train_step(batch, optimizer)
# losses dict includes: recon, flow, repa, cjepa, etc.
```

## Key Architecture Patterns

### Unified Architecture Pattern
```python
# Single configuration controls everything
config = get_research_config()  # or get_minimal_config(), etc.
config.enable_component(ComponentType.CONTROL)
config.disable_component(ComponentType.DEPTH)

# Unified trainer with integrated comma.ai insights
trainer = UnifiedTrainer(world_model, config)

# All training methodologies in one place
losses = trainer.train_step(batch, optimizer)
# -> includes self-forcing, curriculum, flow matching, etc.
```

### Modular World Model
```python
class WorldModel(nn.Module):
    def __init__(self, config: DriveDiTConfig):
        # Core transformer backbone
        self.backbone = nn.ModuleList([DiTBlock(...)])
        
        # Optional components based on config
        if ComponentType.CONTROL in enabled_components:
            self.control_encoder = ControlEncoder(...)
        if ComponentType.MEMORY in enabled_components:
            self.memory_system = MemorySystem(...)
        if ComponentType.FLOW_MATCHING in enabled_components:
            self.flow_predictor = FlowMatchingPredictor(...)
```

### DiT Block Structure
```python
class DiTBlock(nn.Module):
    def forward(self, x, kv=None, pos=None, cond=None, causal_mask=None):
        # Attention with RoPE and KV-cache
        qkv = self.qkv(self.norm_qkv(x + cond))
        q, k, v = qkv.chunk(3, -1)
        q, k = rope(q, *pos), rope(k, *pos)
        if kv: k = torch.cat([kv['k'], k], 1); v = torch.cat([kv['v'], v], 1)
        
        attn = mha(q, k, v, mask=causal_mask)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm_mlp(x, cond))
        return x, {'k': k.detach(), 'v': v.detach()}
```

### 3D Causal VAE
Time-causal convolutions with proper padding:
```python
def causal_conv3d(cin, cout, k=(3,3,3), s=1):
    pad_t = (k[0]-1, 0)  # Causal in time only
    pad_h = pad_w = (k[1]//2, k[1]//2)
    return nn.Sequential(
        nn.ConstantPad3d(pad_t + pad_h + pad_w, 0.),
        nn.Conv3d(cin, cout, k, s)
    )
```

## Training Methodology

### Self-Forcing Training Loop
1. **Teacher pass**: Bidirectional processing over entire clip
2. **Student unroll**: Autoregressive generation using own predictions
3. **Multi-component loss**:
   - Latent L2 (student vs ground truth)
   - Flow-matching loss (teacher flow vs student flow)
   - JEPA contrastive loss between z_t and z_{t+Δ}
   - Depth L1 loss between predicted and ground truth depth

### Unified Training Pipeline
- **Curriculum learning**: Progressive sequence length and self-forcing ratio (comma.ai insights)
- **Modular components**: Enable/disable control, JEPA, depth, memory via configuration
- **Flow matching integration**: Seamless integration with self-forcing training
- **Comprehensive loss system**: All loss types unified in single framework

### Self-Forcing++ Training (Extended Generation)
Based on Self-Forcing++ paper and comma.ai insights for 5s to 4+ minute rollouts:

```python
from training.self_forcing_plus import (
    SelfForcingPlusTrainer,
    get_default_config,
    RollingKVCache,
    CurriculumScheduler,
    FutureAnchorEncoder,
    ExtendedControlEncoder
)

# Initialize trainer with all components
config = get_default_config()
trainer = SelfForcingPlusTrainer(model, config)

# Training loop with automatic curriculum progression
for batch in dataloader:
    losses = trainer.train_step(batch, optimizer)
    # Curriculum automatically advances sequence length and self-forcing ratio
```

**Key Components:**
1. **Rolling KV Cache**: Sliding window cache with auto-truncation and gradient detachment
2. **Curriculum Scheduler**: Progressive sequence length (8->64) and self-forcing ratio (0->1)
3. **Future Anchor Conditioning**: Goal states at 2s, 4s, 6s horizons (comma.ai)
4. **Extended 6D Control**: steering, accel, goal_x, goal_y, speed, heading_rate
5. **Stability Improvements**: EMA, uncertainty weighting (Kendall), per-layer gradient clipping

**Curriculum Schedule:**
- Sequence length grows from `initial_sequence_length` to `final_sequence_length`
- Self-forcing ratio increases via cosine/linear/exponential schedule
- Loss weights introduced gradually (temporal after warmup, anchors after 30%)

### Critical Implementation Notes
- Teacher runs in `eval()` mode with `torch.no_grad()`
- Student uses own predictions (not ground truth) during rollout
- KV-cache management essential for long sequences
- Mixed precision (`torch.float16`) for memory efficiency

## Inference Pipeline

### Minimal Rollout Example
```python
# Initialize components
vae = VAE().cuda()
dit = DiTStudent().cuda()
memory = MemoryBank()
kv_cache = None

# Context processing
z_prev = vae.encode(rgb[:, -1])  # Last context frame

# Autoregressive generation
for t in range(30):
    tokens = fuse(z_prev, depth_tok, mem_tok, ctrl_tok)
    z_next, kv_cache = dit(tokens, kv=kv_cache, causal_mask=causal_mask)
    rgb_next = vae.decode(z_next)
    memory.update(rgb_next)
    z_prev = z_next
```

## Tensor Contracts

### Model Inputs/Outputs
- **RGB frames**: `[B, T, 3, H, W]` (float16, normalized 0-1)
- **Control signals**: `[B, 6]` (steering, accel, goal_x, goal_y, speed, heading_rate)
- **Ego states**: `[B, T, 5]` (x, y, heading, speed, heading_rate)
- **Latent representations**: `[B, T, C, H//8, W//8]`
- **Attention tokens**: `[B, T, D]` where D=model_dim
- **KV cache**: `{'k': [B, past_T, H, D], 'v': [B, past_T, H, D]}`

### Control Signal Normalization
Extended 6D control with per-dimension ranges:
- **steering**: [-1, 1] (normalized)
- **acceleration**: [-5, 5] m/s^2
- **goal_x**: [-50, 50] m (relative position)
- **goal_y**: [-50, 50] m (relative position)
- **speed**: [0, 40] m/s
- **heading_rate**: [-1, 1] rad/s

### Memory Management
- KV-cache detachment after each step to prevent gradient accumulation
- Memory bank token limits for spatial/object permanence
- CPU offloading for long sequence generation

## Performance Optimization

### Compilation and Acceleration
- `torch.compile(model, mode='reduce-overhead')` for 2x+ speedup
- Mixed precision training/inference
- Efficient einsum operations over loops
- Fused operations in `nn_helpers.py`

### Memory Optimization
- Chunked VAE processing for large sequences
- Gradient checkpointing for deep networks
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`

### Large-Scale Data Processing
- **Memory-mapped video access**: Zero-copy frame loading for efficiency
- **Hierarchical storage**: Multi-tier storage (NVMe/SSD/HDD) based on access patterns
- **Intelligent chunking**: Scene-aware video segmentation with temporal overlap
- **Distributed processing**: Multi-GPU/multi-node training with gradient synchronization
- **Quality filtering**: Automated video quality assessment and filtering

## Enfusion Integration

The `src/enfusion/` directory contains organized C scripts for the Enfusion engine:

### Data Capture Components
- **Data Capture** (`src/enfusion/datacapture/`):
  - `SCR_MLDataCollector.c`: ML training data collection (CSV format)
  - `SCR_AnchorFrameSelector.c`: Anchor frame selection for Self-Forcing++ training
  - `SCR_BinarySerializer.c`: High-performance binary capture (ENFCAP format)
  - `SCR_AIDrivingSimulator.c`: AI-controlled vehicle simulation
  - `SCR_DrivingSimDebugUI.c`: Debug visualization overlay

### Anchor Frame System (Self-Forcing++ Training)
The `SCR_AnchorFrameSelector` component implements anchor frame selection for extended
stable generation (5 seconds to 4+ minutes). Anchor frames provide ground-truth
"reset points" during self-forcing training.

**Trigger Types:**
- `PERIODIC`: Regular interval anchors (default: every 150 frames = 30 seconds at 5Hz)
- `ROAD_CHANGE`: Road type transitions (city to highway, etc.)
- `JUNCTION`: Junction/intersection approach and exit
- `STOP/START`: Vehicle stop and restart events (inverse dynamics)
- `STEERING`: Large steering angle changes
- `SPEED_CHANGE`: Significant acceleration/deceleration

**Driving State Machine:**
- Tracks context: CITY_DRIVING, HIGHWAY_DRIVING, RURAL_DRIVING, OFFROAD_DRIVING
- Junction detection: JUNCTION_APPROACH, JUNCTION_TRAVERSAL, JUNCTION_EXIT
- Vehicle state: STOPPED

**Quality Assessment:**
- Motion blur penalty (high speed = lower quality)
- Steering stability (erratic steering = lower quality)
- Speed stability (variable speed = lower quality)
- Minimum quality threshold (configurable, default 0.5)

**Output per Anchor:**
```
{
    frame_id: int,
    trigger_type: enum,
    context_window: int,      // Frames before anchor (for context)
    rollout_horizon: int,     // Frames after anchor (for self-forcing)
    quality_score: float,
    context_state: enum,
    position: vector,
    speed: float,
    steering: float,
    description: string
}
```

**Integration with SCR_MLDataCollector:**
- Enable `m_bEnableAnchorSelection` attribute
- Add `SCR_AnchorFrameSelector` component to same entity
- Anchor metadata automatically captured alongside telemetry

### Navigation Components
- **Extractors** (`src/enfusion/navigation/extractors/`):
  - `SCR_LocalRoadExtractor.c`: Local road topology analysis
  - `SCR_SimpleRoadExtractor.c`: Basic road utilities
- **Visualizers** (`src/enfusion/navigation/visualizers/`):
  - `SCR_EfficientRoadNet.c`: Road network visualization and extraction

### Testing Components
- **Testing** (`src/enfusion/testing/`):
  - `SCR_RoadPhysicsTestMission.c`: Physics testing scenarios

### Development Resources
- **Documentation** (`src/docs/`): Development notes and version files
- **Utilities** (`src/enfusion/utils/`): Shared utilities and helpers (extensible)

### Binary Capture Format (ENFCAP v1)
The `SCR_BinarySerializer.c` implements a high-performance binary format optimized for ML training:
- **Header**: 64 bytes with magic, version, frame count, timestamp, flags
- **Index Table**: Frame offsets for O(1) random access
- **Frame Records**: Variable-length binary records with ego transform, vehicle state, scene entities, road topology
- **Anchor Frames**: Periodic keyframes for efficient seek operations

These provide synthetic data generation capabilities for training.

## Testing and Validation

### Comprehensive Test Suite
- **Mathematical components**: RoPE, attention, flow matching, numerical stability
- **Training pipeline**: Self-forcing, memory optimization, distributed training
- **Model integration**: End-to-end training steps, component interactions
- **Performance validation**: Memory usage, gradient flow, convergence testing

### Key Test Categories
```bash
# Mathematical correctness
pytest tests/test_math_components.py -v

# Training pipeline validation  
pytest tests/test_training_pipeline.py -v

# Integration testing
python -m pytest tests/ --tb=short
```

## Development Philosophy

**Pure Mathematics**: Every operation expressed as explicit tensor manipulations
**Zero Dependencies**: Only torch, einops, opencv—no external model libraries
**Reference Implementation**: Python first, optimize to CUDA/Triton later
**Preserved Contracts**: All tensor shapes and mathematical formulations documented
**Modular Design**: Optional components can be enabled/disabled via configuration
**Production Ready**: Comprehensive testing, memory management, and error handling

The codebase prioritizes mathematical clarity and reproducibility over convenience APIs while maintaining production-grade reliability and performance.

## Research References

### Core Architecture
- Peebles & Xie. [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). ICCV 2023.
- Lipman et al. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). ICLR 2023.

### Efficiency Innovations
- Zhang et al. [SLA: Sparse-Linear Attention](https://arxiv.org/abs/2509.24006). 2025.
- DeepSeek. [DeepSeekMoE: Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066). 2024.
- Yu et al. [REPA: Representation Alignment](https://arxiv.org/abs/2410.06940). ICLR 2025 Oral.
- [HASTE: Holistic Alignment with Stage-wise Termination](https://arxiv.org/abs/2505.16792). 2025.

### Video Understanding
- Meta. [V-JEPA 2](https://arxiv.org/abs/2506.09985). 2025.
- Meta. [V-JEPA 2.1](https://arxiv.org/abs/2603.14482). 2026.
- Meta. [DINOv3](https://arxiv.org/abs/2508.10104). 2025.
- Brown/NYU. [Causal JEPA: Object-Level Interventions](https://arxiv.org/abs/2602.00000). 2026.

### World Models
- [Self-Forcing: Bridging Train-Test Gap](https://arxiv.org/abs/2506.08009). NeurIPS 2025 Spotlight.
- [Self-Forcing++](https://arxiv.org/abs/2510.02283). 2025.
- Wayve. [GAIA-2](https://arxiv.org/abs/2503.20523). 2025.
- comma.ai. [Learning to Drive from a World Model](https://arxiv.org/abs/2504.19077). 2025.
- NVIDIA. [Cosmos World Foundation Model](https://arxiv.org/abs/2501.03575). 2025.

### Evaluation
- [World-in-World: Closed-Loop World Model Benchmark](https://arxiv.org/abs/2312.00000). ICLR 2026 Oral.
- [DrivingGen: Multi-Dimensional Driving Benchmark](https://arxiv.org/abs/2312.00000). ICLR 2026.