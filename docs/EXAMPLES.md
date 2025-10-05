# DriveDiT Examples

This document provides practical examples for using DriveDiT components.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Data Loading](#data-loading)
3. [Training Examples](#training-examples)
4. [Inference Examples](#inference-examples)
5. [Evaluation Examples](#evaluation-examples)
6. [Advanced Usage](#advanced-usage)

## Basic Usage

### Quick Start: Loading and Using Models

```python
import torch
from drivedit import DiTStudent, VAE3D, StreamingRollout

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create student model
student = DiTStudent(
    latent_dim=64,
    d_model=512,
    n_layers=12,
    n_heads=8,
    max_seq_len=256
).to(device)

# Create VAE
vae = VAE3D(
    in_channels=3,
    latent_dim=64,
    hidden_dims=[64, 128, 256, 512],
    num_res_blocks=2
).to(device)

# Set to evaluation mode
student.eval()
vae.eval()

print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
```

### Configuration Management

```python
from drivedit.config import ModelConfig, VAEConfig, StudentConfig, TeacherConfig

# Create configuration
config = ModelConfig(
    vae=VAEConfig(
        in_channels=3,
        latent_dim=64,
        hidden_dims=[64, 128, 256, 512],
        num_res_blocks=2,
        use_attention=True
    ),
    student=StudentConfig(
        latent_dim=64,
        d_model=512,
        n_layers=12,
        n_heads=8,
        max_seq_len=256,
        dropout=0.1
    ),
    teacher=TeacherConfig(
        latent_dim=64,
        d_model=768,
        n_layers=16,
        n_heads=12,
        num_diffusion_steps=1000
    )
)

# Save configuration
config.save_json('model_config.json')

# Load configuration
loaded_config = ModelConfig.from_json('model_config.json')

# Create models from config
student = DiTStudent.from_config(loaded_config.student)
vae = VAE3D.from_config(loaded_config.vae)
```

## Data Loading

### Basic Video Dataset

```python
from drivedit.data import VideoDataset, VideoLoader
from pathlib import Path

# Create dataset
dataset = VideoDataset(
    data_root='./data/driving_videos',
    sequence_length=16,
    frame_skip=1,
    image_size=(256, 256),
    split='train',
    cache_frames=True,
    load_controls=True,
    load_depth=False
)

# Create data loader
loader = VideoLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate through data
for batch in loader:
    frames = batch['frames']  # [B, T, C, H, W]
    controls = batch['controls']  # [B, T, 4]
    
    print(f"Batch frames shape: {frames.shape}")
    print(f"Batch controls shape: {controls.shape}")
    break
```

### Data Preprocessing Pipeline

```python
from drivedit.data import VideoPreprocessor, AugmentationPipeline, DataPreprocessingPipeline

# Create preprocessors
video_preprocessor = VideoPreprocessor(
    target_size=(256, 256),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Create augmentation pipeline
augmentation = AugmentationPipeline([
    {
        'name': 'horizontal_flip',
        'p': 0.5
    },
    {
        'name': 'brightness',
        'p': 0.3,
        'params': {'factor_range': (0.9, 1.1)}
    },
    {
        'name': 'temporal_jitter',
        'p': 0.2,
        'params': {'max_jitter': 1}
    }
], p=0.8)

# Combined pipeline
pipeline = DataPreprocessingPipeline(
    video_preprocessor=video_preprocessor
)

# Apply to batch
processed_batch = pipeline(batch)
augmented_frames = augmentation(processed_batch['frames'])
```

### Custom Dataset Creation

```python
import torch.utils.data as data
from drivedit.data import VideoCollator

class CustomDrivingDataset(data.Dataset):
    def __init__(self, data_dir, sequence_length=16):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.sequences = self._load_sequences()
    
    def _load_sequences(self):
        # Custom loading logic
        sequences = []
        for video_dir in self.data_dir.iterdir():
            if video_dir.is_dir():
                frames = list(video_dir.glob('*.jpg'))
                if len(frames) >= self.sequence_length:
                    sequences.append({
                        'frames': sorted(frames),
                        'sequence_id': video_dir.name
                    })
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load frames (simplified)
        frames = []
        for frame_path in sequence['frames'][:self.sequence_length]:
            # Load and process frame
            frame = torch.randn(3, 256, 256)  # Placeholder
            frames.append(frame)
        
        return {
            'frames': torch.stack(frames),
            'sequence_id': sequence['sequence_id']
        }

# Use custom dataset
custom_dataset = CustomDrivingDataset('./custom_data')
collator = VideoCollator(pad_to_max=True)
custom_loader = data.DataLoader(
    custom_dataset,
    batch_size=4,
    collate_fn=collator,
    shuffle=True
)
```

## Training Examples

### Self-Forcing Training

```python
from drivedit.training import SelfForcingTrainer
from drivedit.training.losses import CompositeLoss
import torch.optim as optim

# Create models
student = DiTStudent(latent_dim=64, d_model=512, n_layers=12, n_heads=8)
vae = VAE3D(in_channels=3, latent_dim=64)

# Create optimizer
optimizer = optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.01)

# Create trainer
trainer = SelfForcingTrainer(
    student_model=student,
    vae_model=vae,
    optimizer=optimizer,
    device='cuda',
    mixed_precision=True,
    use_wandb=False
)

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    
    # Train epoch
    train_losses = trainer.train_epoch(train_loader)
    
    # Validation
    val_losses = trainer.validate_epoch(val_loader)
    
    print(f"Train Loss: {train_losses['total_loss']:.4f}")
    print(f"Val Loss: {val_losses['total_loss']:.4f}")
    
    # Save checkpoint
    if epoch % 5 == 0:
        trainer.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
```

### Teacher-Student Distillation

```python
from drivedit.training import DistillationTrainer
from drivedit.models import DiTTeacher

# Create teacher model
teacher = DiTTeacher(
    latent_dim=64,
    d_model=768,
    n_layers=16,
    n_heads=12,
    num_diffusion_steps=1000
)

# Create distillation trainer
distill_trainer = DistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    vae_model=vae,
    optimizer=optimizer,
    device='cuda',
    use_wandb=False
)

# Distillation training
for epoch in range(20):
    print(f"Distillation Epoch {epoch+1}/20")
    
    losses = distill_trainer.train_epoch(train_loader)
    
    print(f"Distillation Loss: {losses['distillation_loss']:.4f}")
    print(f"Flow Matching Loss: {losses['flow_matching_loss']:.4f}")
    
    if epoch % 10 == 0:
        distill_trainer.save_checkpoint(f'distill_checkpoint_{epoch}.pt')
```

### Custom Loss Functions

```python
from drivedit.training.losses import CompositeLoss
import torch.nn as nn

class CustomWorldModelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred_frames, target_frames, pred_controls=None, target_controls=None):
        losses = {}
        
        # Reconstruction loss
        losses['reconstruction'] = self.mse_loss(pred_frames, target_frames)
        
        # Perceptual loss (simplified)
        losses['perceptual'] = self.l1_loss(pred_frames, target_frames)
        
        # Control loss
        if pred_controls is not None and target_controls is not None:
            losses['control'] = self.mse_loss(pred_controls, target_controls)
        
        return losses

# Use custom loss
custom_loss = CustomWorldModelLoss()
composite_loss = CompositeLoss({
    'reconstruction': 1.0,
    'perceptual': 0.1,
    'control': 0.5
})

# In training loop
def train_step(batch):
    frames = batch['frames']
    controls = batch.get('controls')
    
    # Forward pass
    pred_frames, pred_controls = model(frames, controls)
    
    # Compute losses
    individual_losses = custom_loss(pred_frames, frames, pred_controls, controls)
    total_loss, weights = composite_loss(individual_losses)
    
    return total_loss, individual_losses
```

## Inference Examples

### Basic Sequence Generation

```python
from drivedit.inference import StreamingRollout, InferenceConfig

# Create inference configuration
config = InferenceConfig(
    max_sequence_length=64,
    context_window=8,
    temperature=0.8,
    use_kv_cache=True,
    mixed_precision=True
)

# Create rollout
rollout = StreamingRollout(student, vae, config, device='cuda')

# Generate sequence
context_frames = torch.randn(1, 4, 3, 256, 256).cuda()
control_sequence = torch.randn(1, 12, 4).cuda()

with torch.no_grad():
    result = rollout.generate_sequence(
        context_frames=context_frames,
        control_sequence=control_sequence,
        max_new_frames=8,
        return_intermediates=True
    )

print(f"Generated frames shape: {result['generated_frames'].shape}")
print(f"Full sequence shape: {result['full_sequence'].shape}")
print(f"Generation FPS: {result['performance']['avg_fps']:.2f}")
```

### Real-time Streaming Inference

```python
from drivedit.inference import StreamingRollout
import time

# Setup streaming
rollout = StreamingRollout(student, vae, config, device='cuda')
context_buffer = []
max_context = 4

# Simulate real-time stream
def process_frame_stream():
    for frame_idx in range(100):  # Simulate 100 frames
        # Get new frame (placeholder)
        new_frame = torch.randn(1, 1, 3, 256, 256).cuda()
        
        # Update context buffer
        context_buffer.append(new_frame)
        if len(context_buffer) > max_context:
            context_buffer.pop(0)
        
        # Generate next frame when we have enough context
        if len(context_buffer) == max_context:
            context_frames = torch.cat(context_buffer, dim=1)
            
            start_time = time.time()
            
            with torch.no_grad():
                result = rollout.generate_sequence(
                    context_frames=context_frames,
                    max_new_frames=1,
                    temperature=0.7
                )
            
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time
            
            print(f"Frame {frame_idx}: {fps:.1f} FPS")
            
            # Use generated frame as next context
            generated_frame = result['generated_frames'][:, -1:]
            context_buffer.append(generated_frame)
            context_buffer.pop(0)

# Run streaming
process_frame_stream()
```

### Controllable Generation

```python
# Create control sequences for different behaviors
def create_control_sequence(behavior_type, length=16):
    controls = torch.zeros(1, length, 4)  # [batch, time, control_dim]
    
    if behavior_type == 'straight':
        controls[:, :, 0] = 0.0  # No steering
        controls[:, :, 1] = 0.5  # Constant speed
    
    elif behavior_type == 'left_turn':
        controls[:, :, 0] = -0.3  # Left steering
        controls[:, :, 1] = 0.3   # Slower speed
    
    elif behavior_type == 'right_turn':
        controls[:, :, 0] = 0.3   # Right steering
        controls[:, :, 1] = 0.3   # Slower speed
    
    elif behavior_type == 'stop':
        controls[:, :, 0] = 0.0   # No steering
        controls[:, :, 1] = -0.5  # Deceleration
    
    return controls.cuda()

# Generate different behaviors
behaviors = ['straight', 'left_turn', 'right_turn', 'stop']
context_frames = torch.randn(1, 4, 3, 256, 256).cuda()

for behavior in behaviors:
    controls = create_control_sequence(behavior)
    
    with torch.no_grad():
        result = rollout.generate_sequence(
            context_frames=context_frames,
            control_sequence=controls,
            max_new_frames=8
        )
    
    print(f"{behavior.title()} behavior generated: {result['generated_frames'].shape}")
```

### Memory Bank Usage

```python
from drivedit.models import MemoryBank

# Create memory bank
memory = MemoryBank(
    d_model=512,
    max_spatial_memory=1000,
    max_object_memory=500
)

# Simulate memory updates during driving
for step in range(100):
    # New spatial observations
    spatial_tokens = torch.randn(1, 64, 512).cuda()
    importance_scores = torch.rand(1, 64).cuda()
    
    # Update memory
    memory.update(spatial_tokens, importance_scores)
    
    # Query relevant memories
    query = torch.randn(1, 8, 512).cuda()
    retrieved_memories = memory.retrieve(query, top_k=32)
    
    if step % 20 == 0:
        print(f"Step {step}: Memory size = {len(memory.spatial_memory)}")
        print(f"Retrieved memories shape: {retrieved_memories.shape}")

# Get memory statistics
stats = memory.get_statistics()
print(f"Memory statistics: {stats}")
```

## Evaluation Examples

### Comprehensive Model Evaluation

```python
from drivedit.evaluation import SequenceEvaluator, WorldModelBenchmark

# Create evaluator
evaluator = SequenceEvaluator(
    model=student,
    vae_model=vae,
    device='cuda',
    compute_expensive_metrics=True
)

# Prepare test data
test_frames = torch.randn(4, 16, 3, 256, 256).cuda()
context_frames = test_frames[:, :8]
target_frames = test_frames[:, 8:]

# Evaluate single sequence
results = evaluator.evaluate_sequence(
    context_frames=context_frames[:1],
    target_frames=target_frames[:1],
    return_generated=True
)

print("Evaluation Results:")
for metric, value in results.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")

# Evaluate full batch
batch_results = evaluator.evaluate_batch({
    'frames': test_frames,
    'controls': torch.randn(4, 16, 4).cuda()
})

print(f"\nBatch Results (averaged):")
for metric, value in batch_results.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
```

### Performance Benchmarking

```python
from drivedit.evaluation import PerformanceBenchmark, MemoryBenchmark

# Performance benchmark
perf_benchmark = PerformanceBenchmark(student, device='cuda')
perf_results = perf_benchmark.run_performance_suite()

print("Performance Results:")
print(f"Throughput: {perf_results['throughput']['frames_per_second']:.1f} FPS")
print(f"Latency: {perf_results['latency']['mean_latency_ms']:.1f} ms")
print(f"Memory: {perf_results['memory']['peak_memory_mb']:.1f} MB")

# Memory benchmark
memory_benchmark = MemoryBenchmark(student, device='cuda')

# Find optimal batch size
optimal_batch = memory_benchmark.find_max_batch_size(sequence_length=16)
print(f"Max batch size: {optimal_batch['max_batch_size']}")

# Find optimal sequence length
optimal_seq = memory_benchmark.find_max_sequence_length(batch_size=1)
print(f"Max sequence length: {optimal_seq['max_sequence_length']}")
```

### Model Comparison

```python
from drivedit.evaluation import ComparisonEvaluator

# Create multiple models for comparison
models = {
    'small_model': DiTStudent(latent_dim=32, d_model=256, n_layers=6, n_heads=4),
    'medium_model': DiTStudent(latent_dim=64, d_model=512, n_layers=12, n_heads=8),
    'large_model': DiTStudent(latent_dim=128, d_model=768, n_layers=18, n_heads=12)
}

# Move to device
for model in models.values():
    model.to('cuda').eval()

# Compare models
comparator = ComparisonEvaluator(models, device='cuda')

test_data = {
    'frames': torch.randn(2, 16, 3, 256, 256).cuda(),
    'controls': torch.randn(2, 16, 4).cuda()
}

comparison_results = comparator.compare_models(
    test_data,
    metrics_to_compare=['mse', 'psnr', 'generation_time_s']
)

print("Model Comparison:")
for model_name, results in comparison_results['individual_results'].items():
    print(f"\n{model_name}:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

print("\nBest Models:")
for metric, summary in comparison_results['comparison_summary'].items():
    print(f"  {metric}: {summary['best_model']} ({summary['best_value']:.4f})")
```

## Advanced Usage

### Custom Training Loop

```python
import torch.amp as amp

# Setup
student.train()
vae.eval()  # Keep VAE frozen
optimizer = optim.AdamW(student.parameters(), lr=1e-4)
scaler = amp.GradScaler()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

# Advanced training loop
for epoch in range(100):
    epoch_losses = []
    
    for batch_idx, batch in enumerate(train_loader):
        frames = batch['frames'].cuda()
        controls = batch.get('controls', None)
        
        # Mixed precision forward pass
        with amp.autocast():
            # Encode frames
            with torch.no_grad():
                latents = vae.encode(frames)[0]
            
            # Student forward pass
            B, T, C, H, W = latents.shape
            tokens = latents.view(B, T, C * H * W)
            pred_tokens, _ = student(tokens)
            pred_latents = pred_tokens.view(B, T, C, H, W)
            
            # Compute loss
            loss = F.mse_loss(pred_latents, latents)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        epoch_losses.append(loss.item())
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
```

### Distributed Training

```python
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = DiTStudent(latent_dim=64, d_model=512, n_layers=12, n_heads=8)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, sampler=sampler
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)
        
        for batch in dataloader:
            # Training step
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            optimizer.step()
    
    cleanup()

# Run distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

### Model Compilation and Optimization

```python
# Compile model for faster inference
compiled_student = torch.compile(student, mode='reduce-overhead')
compiled_vae = torch.compile(vae, mode='reduce-overhead')

# Create optimized rollout
optimized_rollout = StreamingRollout(
    compiled_student, 
    compiled_vae, 
    config, 
    device='cuda'
)

# Benchmark compiled vs non-compiled
def benchmark_models():
    context_frames = torch.randn(1, 4, 3, 256, 256).cuda()
    
    # Original models
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = rollout.generate_sequence(context_frames, max_new_frames=1)
    original_time = time.time() - start_time
    
    # Compiled models
    start_time = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = optimized_rollout.generate_sequence(context_frames, max_new_frames=1)
    compiled_time = time.time() - start_time
    
    speedup = original_time / compiled_time
    print(f"Speedup: {speedup:.2f}x")

benchmark_models()
```

### Export and Deployment

```python
# Export to TorchScript
def export_to_torchscript():
    student.eval()
    
    # Trace the model
    dummy_input = torch.randn(1, 8, 4096).cuda()  # [batch, seq, features]
    traced_model = torch.jit.trace(student, dummy_input)
    
    # Save traced model
    traced_model.save('student_traced.pt')
    
    # Load and test
    loaded_model = torch.jit.load('student_traced.pt')
    
    with torch.no_grad():
        original_output = student(dummy_input)[0]
        traced_output = loaded_model(dummy_input)[0]
        
        print(f"Output difference: {(original_output - traced_output).abs().max().item()}")

export_to_torchscript()

# Export to ONNX
def export_to_onnx():
    dummy_input = torch.randn(1, 8, 4096).cuda()
    
    torch.onnx.export(
        student,
        dummy_input,
        'student_model.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    print("Model exported to ONNX format")

export_to_onnx()
```

This completes the comprehensive examples documentation for DriveDiT. Each section provides practical, runnable code that demonstrates different aspects of the system.