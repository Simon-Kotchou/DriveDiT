# DriveDiT API Reference

Complete API documentation for all DriveDiT modules and functions.

## Table of Contents

1. [Core Models](#core-models)
2. [Training](#training)
3. [Inference](#inference)
4. [Data Pipeline](#data-pipeline)
5. [Evaluation](#evaluation)
6. [Utilities](#utilities)
7. [Configuration](#configuration)

---

## Core Models

### VAE3D

**Class**: `drivedit.models.VAE3D`

3D Causal Variational Autoencoder for video encoding and decoding.

#### Constructor

```python
VAE3D(
    in_channels: int = 3,
    latent_dim: int = 64,
    hidden_dims: List[int] = [64, 128, 256, 512],
    num_res_blocks: int = 2,
    use_attention: bool = True,
    beta: float = 4.0
)
```

**Parameters:**
- `in_channels`: Number of input channels (RGB = 3)
- `latent_dim`: Latent space dimensionality
- `hidden_dims`: Hidden layer dimensions for encoder/decoder
- `num_res_blocks`: Number of residual blocks per layer
- `use_attention`: Whether to use attention in bottleneck
- `beta`: β-VAE regularization parameter

#### Methods

##### encode()
```python
encode(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```
Encode input frames to latent space.

**Args:**
- `x`: Input frames `[B, C, T, H, W]`

**Returns:**
- `mean`: Latent mean `[B, latent_dim, T, H//8, W//8]`
- `logvar`: Latent log variance `[B, latent_dim, T, H//8, W//8]`

##### decode()
```python
decode(z: torch.Tensor) -> torch.Tensor
```
Decode latent representations to frames.

**Args:**
- `z`: Latent representations `[B, latent_dim, T, H//8, W//8]`

**Returns:**
- Reconstructed frames `[B, C, T, H, W]`

##### reparameterize()
```python
reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor
```
Reparameterization trick for VAE sampling.

##### loss_function()
```python
loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor
```
Compute VAE loss (reconstruction + KL divergence).

---

### DiTStudent

**Class**: `drivedit.models.DiTStudent`

Causal Diffusion Transformer for autoregressive world modeling.

#### Constructor

```python
DiTStudent(
    latent_dim: int = 64,
    d_model: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
    d_ff: Optional[int] = None,
    max_seq_len: int = 256,
    dropout: float = 0.1,
    use_rope: bool = True,
    rope_base: float = 10000.0
)
```

#### Methods

##### forward()
```python
forward(
    x: torch.Tensor,
    kv_cache: Optional[Dict] = None,
    use_cache: bool = False
) -> Tuple[torch.Tensor, Optional[Dict]]
```
Forward pass through the model.

**Args:**
- `x`: Input tokens `[B, T, D]`
- `kv_cache`: Key-value cache for efficient generation
- `use_cache`: Whether to return updated cache

**Returns:**
- `output`: Generated tokens `[B, T, D]`
- `kv_cache`: Updated key-value cache (if use_cache=True)

##### generate()
```python
generate(
    context: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor
```
Autoregressive generation.

---

### DiTTeacher

**Class**: `drivedit.models.DiTTeacher`

Bidirectional Diffusion Transformer for teacher training.

#### Constructor

```python
DiTTeacher(
    latent_dim: int = 64,
    d_model: int = 768,
    n_layers: int = 16,
    n_heads: int = 12,
    d_ff: Optional[int] = None,
    num_diffusion_steps: int = 1000,
    beta_schedule: str = 'linear'
)
```

#### Methods

##### add_noise()
```python
add_noise(
    x: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor
) -> torch.Tensor
```
Add noise according to diffusion schedule.

##### sample()
```python
sample(
    shape: Tuple[int, ...],
    num_steps: int = 50,
    guidance_scale: float = 1.0
) -> torch.Tensor
```
Sample from the learned distribution.

---

## Training

### SelfForcingTrainer

**Class**: `drivedit.training.SelfForcingTrainer`

Trainer for self-forcing (autoregressive) training.

#### Constructor

```python
SelfForcingTrainer(
    student_model: nn.Module,
    vae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    mixed_precision: bool = True,
    use_wandb: bool = False
)
```

#### Methods

##### train_epoch()
```python
train_epoch(dataloader: DataLoader) -> Dict[str, float]
```
Train for one epoch.

**Returns:**
- Dictionary of averaged losses

##### validate_epoch()
```python
validate_epoch(dataloader: DataLoader) -> Dict[str, float]
```
Validate for one epoch.

##### save_checkpoint()
```python
save_checkpoint(
    filepath: str,
    epoch: int = 0,
    best_metric: float = float('inf'),
    **kwargs
) -> None
```
Save training checkpoint.

---

### DistillationTrainer

**Class**: `drivedit.training.DistillationTrainer`

Trainer for teacher-student distillation.

#### Constructor

```python
DistillationTrainer(
    teacher_model: nn.Module,
    student_model: nn.Module,
    vae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda',
    distillation_alpha: float = 0.7,
    temperature: float = 4.0
)
```

#### Methods

##### train_epoch()
```python
train_epoch(dataloader: DataLoader) -> Dict[str, float]
```
Train one distillation epoch.

---

### Loss Functions

#### ReconstructionLoss

**Class**: `drivedit.training.losses.ReconstructionLoss`

Multi-component reconstruction loss.

```python
ReconstructionLoss(
    pixel_weight: float = 1.0,
    perceptual_weight: float = 0.1,
    ssim_weight: float = 0.1
)
```

#### FlowMatchingLoss

**Class**: `drivedit.training.losses.FlowMatchingLoss`

Flow matching loss for distillation.

```python
FlowMatchingLoss(sigma_min: float = 0.002, sigma_max: float = 80.0)
```

#### CompositeLoss

**Class**: `drivedit.training.losses.CompositeLoss`

Combines multiple loss components.

```python
CompositeLoss(
    loss_weights: Dict[str, float],
    uncertainty_weighting: bool = False
)
```

---

## Inference

### StreamingRollout

**Class**: `drivedit.inference.StreamingRollout`

Real-time streaming inference for video generation.

#### Constructor

```python
StreamingRollout(
    world_model: nn.Module,
    vae_model: nn.Module,
    config: InferenceConfig,
    device: str = 'cuda'
)
```

#### Methods

##### generate_sequence()
```python
generate_sequence(
    context_frames: torch.Tensor,
    control_sequence: Optional[torch.Tensor] = None,
    max_new_frames: int = 16,
    temperature: float = 0.8,
    return_intermediates: bool = False
) -> Dict[str, Any]
```
Generate video sequence from context.

**Args:**
- `context_frames`: Context frames `[B, T_ctx, C, H, W]`
- `control_sequence`: Control signals `[B, T_total, control_dim]`
- `max_new_frames`: Number of frames to generate
- `temperature`: Sampling temperature
- `return_intermediates`: Whether to return intermediate results

**Returns:**
- Dictionary containing:
  - `generated_frames`: Generated frames `[B, T_gen, C, H, W]`
  - `full_sequence`: Context + generated `[B, T_total, C, H, W]`
  - `metadata`: Generation metadata
  - `performance`: Performance metrics

##### reset_state()
```python
reset_state() -> None
```
Reset internal state for new sequence.

---

### InferenceConfig

**Class**: `drivedit.inference.InferenceConfig`

Configuration for inference settings.

```python
InferenceConfig(
    max_sequence_length: int = 256,
    context_window: int = 16,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    use_kv_cache: bool = True,
    mixed_precision: bool = True,
    memory_offload_freq: int = 10
)
```

---

### MemoryBank

**Class**: `drivedit.models.MemoryBank`

Episodic memory for spatial and object permanence.

#### Constructor

```python
MemoryBank(
    d_model: int = 512,
    max_spatial_memory: int = 1000,
    max_object_memory: int = 500,
    similarity_threshold: float = 0.8
)
```

#### Methods

##### update()
```python
update(
    spatial_tokens: torch.Tensor,
    importance_scores: torch.Tensor,
    frame_idx: Optional[int] = None
) -> None
```
Update memory with new observations.

##### retrieve()
```python
retrieve(
    query: torch.Tensor,
    top_k: int = 32
) -> torch.Tensor
```
Retrieve relevant memories for query.

##### clear()
```python
clear() -> None
```
Clear all memories.

---

## Data Pipeline

### VideoDataset

**Class**: `drivedit.data.VideoDataset`

Dataset for loading video sequences with controls and depth.

#### Constructor

```python
VideoDataset(
    data_root: Union[str, Path],
    sequence_length: int = 16,
    frame_skip: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    split: str = 'train',
    cache_frames: bool = True,
    load_controls: bool = True,
    load_depth: bool = False
)
```

#### Methods

##### __getitem__()
```python
__getitem__(idx: int) -> Dict[str, torch.Tensor]
```
Get sequence at index.

**Returns:**
- Dictionary containing:
  - `frames`: Video frames `[T, C, H, W]`
  - `controls`: Control signals `[T, control_dim]` (if enabled)
  - `depth`: Depth maps `[T, 1, H, W]` (if enabled)
  - `sequence_id`: Sequence identifier
  - `start_idx`: Starting frame index

---

### VideoLoader

**Class**: `drivedit.data.VideoLoader`

DataLoader wrapper with video-specific functionality.

#### Constructor

```python
VideoLoader(
    dataset: VideoDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2
)
```

#### Methods

##### get_cache_stats()
```python
get_cache_stats() -> Dict[str, Any]
```
Get frame cache statistics.

##### clear_cache()
```python
clear_cache() -> None
```
Clear frame cache.

---

### Transforms

#### VideoTransforms

**Class**: `drivedit.data.VideoTransforms`

Video-level transformations.

##### Static Methods

```python
temporal_crop(video: torch.Tensor, length: int, start: Optional[int] = None) -> torch.Tensor
temporal_subsample(video: torch.Tensor, factor: int) -> torch.Tensor
temporal_flip(video: torch.Tensor) -> torch.Tensor
apply_frame_transform(video: torch.Tensor, transform_fn: Callable, **kwargs) -> torch.Tensor
```

#### FrameTransforms

**Class**: `drivedit.data.FrameTransforms`

Frame-level transformations.

##### Static Methods

```python
normalize(frame: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor
resize(frame: torch.Tensor, size: Tuple[int, int], mode: str = 'bilinear') -> torch.Tensor
crop(frame: torch.Tensor, top: int, left: int, height: int, width: int) -> torch.Tensor
horizontal_flip(frame: torch.Tensor) -> torch.Tensor
brightness(frame: torch.Tensor, factor: float) -> torch.Tensor
contrast(frame: torch.Tensor, factor: float) -> torch.Tensor
gaussian_noise(frame: torch.Tensor, std: float = 0.01) -> torch.Tensor
```

#### AugmentationPipeline

**Class**: `drivedit.data.AugmentationPipeline`

Composable augmentation pipeline.

```python
AugmentationPipeline(transforms: List[Dict[str, Any]], p: float = 0.5)
```

##### Class Methods

```python
create_training_pipeline(image_size: Tuple[int, int]) -> 'AugmentationPipeline'
create_validation_pipeline(image_size: Tuple[int, int]) -> 'AugmentationPipeline'
```

---

### Preprocessing

#### VideoPreprocessor

**Class**: `drivedit.data.VideoPreprocessor`

Video preprocessing utilities.

```python
VideoPreprocessor(
    target_size: Tuple[int, int] = (256, 256),
    normalize: bool = True,
    mean: List[float] = None,
    std: List[float] = None,
    dtype: torch.dtype = torch.float32
)
```

#### ControlPreprocessor

**Class**: `drivedit.data.ControlPreprocessor`

Control signal preprocessing.

```python
ControlPreprocessor(
    control_dim: int = 4,
    normalize_controls: bool = True,
    control_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    smooth_controls: bool = True,
    smooth_window: int = 3
)
```

---

## Evaluation

### Metrics

#### VideoMetrics

**Class**: `drivedit.evaluation.VideoMetrics`

Video quality metrics.

##### Static Methods

```python
mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor
psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor
ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor
lpips_proxy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor
```

#### TemporalConsistencyMetrics

**Class**: `drivedit.evaluation.TemporalConsistencyMetrics`

Temporal consistency metrics.

##### Static Methods

```python
temporal_mse(frames: torch.Tensor) -> torch.Tensor
optical_flow_consistency(frames: torch.Tensor) -> torch.Tensor
frame_warping_error(frames: torch.Tensor, flows: Optional[torch.Tensor] = None) -> torch.Tensor
```

#### ControlMetrics

**Class**: `drivedit.evaluation.ControlMetrics`

Control signal metrics.

##### Static Methods

```python
control_mse(pred_controls: torch.Tensor, target_controls: torch.Tensor) -> torch.Tensor
steering_accuracy(pred_controls: torch.Tensor, target_controls: torch.Tensor, threshold: float = 0.1) -> torch.Tensor
control_smoothness(controls: torch.Tensor) -> torch.Tensor
```

#### compute_all_metrics()

**Function**: `drivedit.evaluation.compute_all_metrics`

Compute comprehensive evaluation metrics.

```python
compute_all_metrics(
    pred_frames: torch.Tensor,
    target_frames: torch.Tensor,
    pred_controls: Optional[torch.Tensor] = None,
    target_controls: Optional[torch.Tensor] = None,
    pred_depth: Optional[torch.Tensor] = None,
    target_depth: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]
```

---

### Evaluators

#### SequenceEvaluator

**Class**: `drivedit.evaluation.SequenceEvaluator`

High-level sequence evaluation.

```python
SequenceEvaluator(
    model,
    vae_model=None,
    device: str = 'cuda',
    compute_expensive_metrics: bool = True
)
```

##### Methods

```python
evaluate_sequence(
    context_frames: torch.Tensor,
    target_frames: torch.Tensor,
    controls: Optional[torch.Tensor] = None,
    return_generated: bool = False
) -> Dict[str, Any]

evaluate_batch(
    batch: Dict[str, torch.Tensor],
    context_ratio: float = 0.5
) -> Dict[str, Any]
```

#### PerformanceBenchmark

**Class**: `drivedit.evaluation.PerformanceBenchmark`

Performance benchmarking suite.

```python
PerformanceBenchmark(model, device: str = 'cuda')
```

##### Methods

```python
run_performance_suite() -> Dict[str, Any]
```

Returns comprehensive performance metrics including:
- `throughput`: Frames per second
- `latency`: Inference latency statistics
- `memory`: Memory usage metrics
- `scalability`: Performance vs sequence length

---

### Benchmarks

#### WorldModelBenchmark

**Class**: `drivedit.evaluation.WorldModelBenchmark`

Comprehensive world model evaluation.

```python
WorldModelBenchmark(
    model,
    vae_model=None,
    device: str = 'cuda',
    save_path: Optional[str] = None
)
```

##### Methods

```python
run_full_benchmark(
    test_data: Dict[str, torch.Tensor],
    sequence_lengths: List[int] = [8, 16, 32],
    batch_sizes: List[int] = [1, 4, 8],
    num_iterations: int = 10
) -> Dict[str, Any]
```

---

## Utilities

### Tensor Utils

#### Functions

```python
safe_cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor
safe_stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor
pad_sequence(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0) -> torch.Tensor
chunk_tensor(tensor: torch.Tensor, chunk_size: int, dim: int = 0, overlap: int = 0) -> List[torch.Tensor]
move_to_device(obj, device: torch.device) -> Any
tensor_memory_usage(tensor: torch.Tensor) -> float
normalize_tensor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor
```

### Model Utils

#### Functions

```python
count_parameters(model: nn.Module, trainable_only: bool = True) -> int
freeze_parameters(model: nn.Module, freeze: bool = True) -> None
get_device(model: nn.Module) -> torch.device
set_seed(seed: int) -> None
initialize_weights(model: nn.Module, init_type: str = 'xavier', gain: float = 1.0) -> None
clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float
get_gradient_norm(model: nn.Module) -> float
save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, filepath: str, **kwargs) -> None
load_checkpoint(filepath: str, model: nn.Module, optimizer=None, device=None) -> Dict[str, Any]
exponential_moving_average(model: nn.Module, ema_model: nn.Module, decay: float = 0.999) -> None
get_model_size_mb(model: nn.Module) -> float
```

### Math Utils

#### Functions

```python
gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor
stable_softmax(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor
cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
pairwise_distance_matrix(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor
finite_difference_gradient(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
laplacian(tensor: torch.Tensor) -> torch.Tensor
gumbel_noise(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor
gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor
```

---

## Configuration

### BaseConfig

**Class**: `drivedit.config.BaseConfig`

Base configuration class with JSON serialization.

#### Methods

```python
to_dict() -> Dict[str, Any]
to_json() -> str
save_json(filepath: str) -> None

@classmethod
from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig'

@classmethod
from_json(cls, json_str_or_path: str) -> 'BaseConfig'
```

### ModelConfig

**Class**: `drivedit.config.ModelConfig`

Complete model configuration.

```python
@dataclass
class ModelConfig(BaseConfig):
    vae: VAEConfig
    student: StudentConfig
    teacher: TeacherConfig
    training: TrainingConfig
```

### VAEConfig

**Class**: `drivedit.config.VAEConfig`

VAE model configuration.

```python
@dataclass
class VAEConfig(BaseConfig):
    in_channels: int = 3
    latent_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_res_blocks: int = 2
    use_attention: bool = True
    beta: float = 4.0
```

### StudentConfig

**Class**: `drivedit.config.StudentConfig`

Student model configuration.

```python
@dataclass
class StudentConfig(BaseConfig):
    latent_dim: int = 64
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: Optional[int] = None
    max_seq_len: int = 256
    dropout: float = 0.1
    use_rope: bool = True
    rope_base: float = 10000.0
```

### get_small_config()

**Function**: `drivedit.config.get_small_config`

Get configuration for small model (testing/development).

```python
get_small_config() -> ModelConfig
```

### get_large_config()

**Function**: `drivedit.config.get_large_config`

Get configuration for large model (production).

```python
get_large_config() -> ModelConfig
```

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameter values or tensor shapes
- `RuntimeError`: CUDA out of memory or computation errors
- `FileNotFoundError`: Missing model files or data
- `KeyError`: Missing required configuration keys

### Example Error Handling

```python
try:
    result = rollout.generate_sequence(context_frames, max_new_frames=16)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Handle OOM
        torch.cuda.empty_cache()
        result = rollout.generate_sequence(context_frames, max_new_frames=8)
    else:
        raise e
except ValueError as e:
    print(f"Invalid input: {e}")
    # Handle invalid input gracefully
```

---

## Version Information

Current API version: 1.0.0

### Compatibility

- **PyTorch**: >= 2.2.0
- **Python**: >= 3.8
- **CUDA**: >= 11.8 (optional, for GPU acceleration)

### Breaking Changes

None for version 1.0.0 (initial release).

---

This API reference provides comprehensive documentation for all public interfaces in DriveDiT. For implementation details and examples, see the [Examples](EXAMPLES.md) documentation.