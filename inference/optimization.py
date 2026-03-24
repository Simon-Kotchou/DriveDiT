"""
Comprehensive inference optimization utilities for DriveDiT.

This module provides production-ready optimization tools including:
1. ModelCompiler - torch.compile wrapper with fallback and warmup
2. InferenceOptimizer - Mixed precision, KV-cache optimization, memory-efficient attention
3. StreamingInference - Real-time streaming video generation
4. Benchmarking utilities - FPS, latency, memory profiling

Performance targets from research:
- LongLive: 20.7 FPS real-time interactive
- HiStream: 0.34s per frame at 1080p (107x faster than baseline)
- Self-Forcing: 16 FPS streaming generation

References:
- LongLive: Stable Long-Form Video Generation (20.7 FPS)
- HiStream: High-Throughput Streaming Inference
- Self-Forcing: Autoregressive Video Diffusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import (
    Dict, List, Optional, Tuple, Union, Any, Callable,
    Iterator, TypeVar, Generic
)
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
from enum import Enum
import time
import warnings
import gc
import math
import functools
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training.self_forcing_plus import RollingKVCache
    HAS_ROLLING_KV_CACHE = True
except ImportError:
    HAS_ROLLING_KV_CACHE = False
    RollingKVCache = None


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar('T', bound=nn.Module)


class CompileMode(Enum):
    """Available torch.compile modes."""
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"
    DEFAULT = "default"
    NONE = None


class PrecisionMode(Enum):
    """Precision modes for inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    AUTO = "auto"


@dataclass
class OptimizationConfig:
    """Configuration for inference optimization."""

    # Compilation settings
    compile_mode: CompileMode = CompileMode.REDUCE_OVERHEAD
    compile_dynamic: bool = False
    compile_fullgraph: bool = False
    warmup_iterations: int = 3

    # Precision settings
    precision_mode: PrecisionMode = PrecisionMode.FP16
    use_autocast: bool = True

    # KV-cache settings
    kv_cache_max_length: int = 512
    kv_cache_truncate_to: int = 256
    kv_cache_detach_interval: int = 8

    # Memory settings
    max_memory_gb: float = 16.0
    memory_cleanup_threshold: float = 0.85
    chunk_size: int = 4
    use_gradient_checkpointing: bool = False

    # Streaming settings
    frame_buffer_size: int = 8
    target_fps: float = 20.0
    latency_budget_ms: float = 50.0

    # Auto-tuning
    auto_tune_batch_size: bool = True
    max_batch_size: int = 32
    min_batch_size: int = 1


@dataclass
class BenchmarkResult:
    """Results from inference benchmarking."""

    # Timing metrics
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput
    fps: float = 0.0
    throughput_samples_per_sec: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_reserved_mb: float = 0.0

    # Additional info
    batch_size: int = 1
    sequence_length: int = 1
    warmup_iterations: int = 0
    benchmark_iterations: int = 0
    device: str = "cuda"

    # Per-operation breakdown
    operation_times: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'latency': {
                'avg_ms': self.avg_latency_ms,
                'min_ms': self.min_latency_ms,
                'max_ms': self.max_latency_ms,
                'p50_ms': self.p50_latency_ms,
                'p95_ms': self.p95_latency_ms,
                'p99_ms': self.p99_latency_ms
            },
            'throughput': {
                'fps': self.fps,
                'samples_per_sec': self.throughput_samples_per_sec
            },
            'memory': {
                'peak_mb': self.peak_memory_mb,
                'avg_mb': self.avg_memory_mb,
                'reserved_mb': self.memory_reserved_mb
            },
            'config': {
                'batch_size': self.batch_size,
                'sequence_length': self.sequence_length,
                'device': self.device,
                'warmup': self.warmup_iterations,
                'iterations': self.benchmark_iterations
            },
            'operations': self.operation_times
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Benchmark Results:\n"
            f"  Latency: {self.avg_latency_ms:.2f}ms avg "
            f"(p50: {self.p50_latency_ms:.2f}ms, p95: {self.p95_latency_ms:.2f}ms)\n"
            f"  FPS: {self.fps:.1f} ({self.throughput_samples_per_sec:.1f} samples/sec)\n"
            f"  Memory: {self.peak_memory_mb:.1f}MB peak, {self.avg_memory_mb:.1f}MB avg\n"
            f"  Config: batch={self.batch_size}, seq_len={self.sequence_length}, "
            f"device={self.device}"
        )


# =============================================================================
# Model Compiler
# =============================================================================

class ModelCompiler:
    """
    Wrapper for torch.compile with mode selection, partial compilation,
    warmup mechanism, and fallback handling.

    Features:
    - Multiple compilation modes (reduce-overhead, max-autotune)
    - Partial compilation for specific submodules
    - Automatic warmup for optimal performance
    - Graceful fallback on compilation failure
    - Dynamic shape support configuration

    Example:
        compiler = ModelCompiler(mode=CompileMode.REDUCE_OVERHEAD)
        compiled_model = compiler.compile(model)

        # With warmup
        compiler.warmup(compiled_model, example_input, iterations=3)
    """

    def __init__(
        self,
        mode: CompileMode = CompileMode.REDUCE_OVERHEAD,
        dynamic: bool = False,
        fullgraph: bool = False,
        backend: str = "inductor",
        disable_cudagraphs: bool = False
    ):
        """
        Initialize the model compiler.

        Args:
            mode: Compilation mode (reduce-overhead, max-autotune, default)
            dynamic: Enable dynamic shape support
            fullgraph: Require full graph compilation (no graph breaks)
            backend: Compilation backend (inductor, eager, etc.)
            disable_cudagraphs: Disable CUDA graphs for debugging
        """
        self.mode = mode
        self.dynamic = dynamic
        self.fullgraph = fullgraph
        self.backend = backend
        self.disable_cudagraphs = disable_cudagraphs

        # Track compilation state
        self._compiled_modules: Dict[int, nn.Module] = {}
        self._compilation_errors: Dict[int, Exception] = {}
        self._warmup_complete: Dict[int, bool] = {}

    def compile(
        self,
        model: T,
        fallback_on_error: bool = True,
        verbose: bool = True
    ) -> T:
        """
        Compile a model with torch.compile.

        Args:
            model: Model to compile
            fallback_on_error: Return original model if compilation fails
            verbose: Print compilation status messages

        Returns:
            Compiled model or original model on failure
        """
        model_id = id(model)

        # Check if already compiled
        if model_id in self._compiled_modules:
            return self._compiled_modules[model_id]

        # Skip compilation if mode is None
        if self.mode == CompileMode.NONE:
            if verbose:
                print("Compilation disabled (mode=NONE)")
            return model

        # Check torch version
        if not hasattr(torch, 'compile'):
            if verbose:
                print("torch.compile not available (requires PyTorch 2.0+)")
            return model

        try:
            compile_kwargs = {
                'mode': self.mode.value if self.mode.value else None,
                'dynamic': self.dynamic,
                'fullgraph': self.fullgraph,
                'backend': self.backend
            }

            # Handle disable_cudagraphs option
            if self.disable_cudagraphs and self.mode == CompileMode.REDUCE_OVERHEAD:
                compile_kwargs['options'] = {"triton.cudagraphs": False}

            # Compile the model
            compiled_model = torch.compile(model, **compile_kwargs)

            self._compiled_modules[model_id] = compiled_model
            self._warmup_complete[model_id] = False

            if verbose:
                print(f"Model compiled with mode='{self.mode.value}', "
                      f"dynamic={self.dynamic}, backend='{self.backend}'")

            return compiled_model

        except Exception as e:
            self._compilation_errors[model_id] = e

            if verbose:
                print(f"Compilation failed: {e}")

            if fallback_on_error:
                if verbose:
                    print("Falling back to uncompiled model")
                return model
            else:
                raise

    def compile_submodule(
        self,
        model: nn.Module,
        submodule_name: str,
        fallback_on_error: bool = True,
        verbose: bool = True
    ) -> nn.Module:
        """
        Compile a specific submodule within a larger model.

        Args:
            model: Parent model
            submodule_name: Name of submodule to compile (e.g., 'encoder.attention')
            fallback_on_error: Return original on failure
            verbose: Print status messages

        Returns:
            Model with compiled submodule
        """
        # Get the submodule
        parts = submodule_name.split('.')
        submodule = model
        for part in parts:
            submodule = getattr(submodule, part)

        # Compile it
        compiled_submodule = self.compile(
            submodule,
            fallback_on_error=fallback_on_error,
            verbose=verbose
        )

        # Replace in parent model
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], compiled_submodule)

        return model

    def compile_multiple_submodules(
        self,
        model: nn.Module,
        submodule_names: List[str],
        verbose: bool = True
    ) -> nn.Module:
        """
        Compile multiple submodules within a model.

        Args:
            model: Parent model
            submodule_names: List of submodule names to compile
            verbose: Print status messages

        Returns:
            Model with compiled submodules
        """
        for name in submodule_names:
            try:
                model = self.compile_submodule(
                    model, name,
                    fallback_on_error=True,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"Failed to compile submodule '{name}': {e}")

        return model

    def warmup(
        self,
        model: nn.Module,
        example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
        iterations: int = 3,
        verbose: bool = True
    ) -> None:
        """
        Warm up compiled model with example inputs.

        The first forward pass through a compiled model is typically slow
        due to compilation overhead. Running warmup iterations ensures
        subsequent inference is fast.

        Args:
            model: Compiled model to warm up
            example_input: Example input tensor or tuple/dict of tensors
            iterations: Number of warmup iterations
            verbose: Print timing information
        """
        model_id = id(model)

        if model_id in self._warmup_complete and self._warmup_complete[model_id]:
            if verbose:
                print("Model already warmed up")
            return

        model.eval()
        times = []

        with torch.no_grad():
            for i in range(iterations):
                # Sync before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()

                # Forward pass
                if isinstance(example_input, dict):
                    _ = model(**example_input)
                elif isinstance(example_input, tuple):
                    _ = model(*example_input)
                else:
                    _ = model(example_input)

                # Sync after
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

                if verbose:
                    print(f"Warmup iteration {i+1}/{iterations}: {elapsed:.2f}ms")

        self._warmup_complete[model_id] = True

        if verbose:
            speedup = times[0] / times[-1] if times[-1] > 0 else 1.0
            print(f"Warmup complete. First: {times[0]:.2f}ms, "
                  f"Last: {times[-1]:.2f}ms, Speedup: {speedup:.1f}x")

    def is_compiled(self, model: nn.Module) -> bool:
        """Check if a model has been compiled."""
        return id(model) in self._compiled_modules

    def is_warmed_up(self, model: nn.Module) -> bool:
        """Check if a model has been warmed up."""
        model_id = id(model)
        return self._warmup_complete.get(model_id, False)

    def get_compilation_error(self, model: nn.Module) -> Optional[Exception]:
        """Get compilation error for a model, if any."""
        return self._compilation_errors.get(id(model))

    def reset(self) -> None:
        """Reset all compilation state."""
        self._compiled_modules.clear()
        self._compilation_errors.clear()
        self._warmup_complete.clear()


# =============================================================================
# Inference Optimizer
# =============================================================================

class InferenceOptimizer:
    """
    High-level optimizer for inference pipelines.

    Features:
    - Mixed precision inference (fp16/bf16 autocast)
    - KV-cache optimization for long sequences
    - Memory-efficient attention (chunked processing)
    - Batch size auto-tuning based on available memory

    Example:
        optimizer = InferenceOptimizer(config)

        # Optimize a model
        optimized_model = optimizer.optimize_model(model)

        # Run optimized inference
        with optimizer.inference_context():
            output = optimized_model(input)
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the inference optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()

        # Initialize model compiler
        self.compiler = ModelCompiler(
            mode=self.config.compile_mode,
            dynamic=self.config.compile_dynamic,
            fullgraph=self.config.compile_fullgraph
        )

        # KV-cache management
        self._kv_cache: Optional[RollingKVCache] = None
        self._kv_cache_enabled = HAS_ROLLING_KV_CACHE

        # Memory tracking
        self._memory_tracker = MemoryTracker()

        # Auto-tuned batch size
        self._optimal_batch_size: Optional[int] = None

        # Precision settings
        self._dtype = self._get_dtype()

    def _get_dtype(self) -> torch.dtype:
        """Get the appropriate dtype based on precision mode."""
        mode = self.config.precision_mode

        if mode == PrecisionMode.FP32:
            return torch.float32
        elif mode == PrecisionMode.FP16:
            return torch.float16
        elif mode == PrecisionMode.BF16:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                warnings.warn("BF16 not supported, falling back to FP16")
                return torch.float16
        elif mode == PrecisionMode.AUTO:
            # Choose based on hardware
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                else:
                    return torch.float16
            return torch.float32
        else:
            return torch.float32

    def optimize_model(
        self,
        model: nn.Module,
        example_input: Optional[torch.Tensor] = None,
        compile_model: bool = True,
        warmup: bool = True
    ) -> nn.Module:
        """
        Apply all optimizations to a model.

        Args:
            model: Model to optimize
            example_input: Example input for warmup
            compile_model: Whether to apply torch.compile
            warmup: Whether to run warmup iterations

        Returns:
            Optimized model
        """
        model.eval()

        # Apply gradient checkpointing if configured
        if self.config.use_gradient_checkpointing:
            self._enable_gradient_checkpointing(model)

        # Compile model
        if compile_model and self.config.compile_mode != CompileMode.NONE:
            model = self.compiler.compile(model, verbose=True)

        # Convert to target precision
        if self.config.precision_mode != PrecisionMode.FP32:
            model = model.to(self._dtype)

        # Warmup
        if warmup and example_input is not None:
            self.compiler.warmup(
                model,
                example_input.to(self._dtype),
                iterations=self.config.warmup_iterations
            )

        return model

    def _enable_gradient_checkpointing(self, model: nn.Module) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)

    @contextmanager
    def inference_context(
        self,
        use_autocast: Optional[bool] = None,
        device: str = 'cuda'
    ):
        """
        Context manager for optimized inference.

        Args:
            use_autocast: Override autocast setting
            device: Device for autocast

        Yields:
            Context for running optimized inference
        """
        use_autocast = use_autocast if use_autocast is not None else self.config.use_autocast

        with torch.inference_mode():
            if use_autocast and torch.cuda.is_available():
                with autocast(dtype=self._dtype):
                    yield
            else:
                yield

    def initialize_kv_cache(
        self,
        num_layers: int,
        device: Optional[torch.device] = None
    ) -> Optional['RollingKVCache']:
        """
        Initialize KV-cache for efficient long-sequence inference.

        Args:
            num_layers: Number of transformer layers
            device: Device for cache storage

        Returns:
            Initialized KV-cache or None if not available
        """
        if not self._kv_cache_enabled:
            warnings.warn("RollingKVCache not available. "
                         "Install training.self_forcing_plus module.")
            return None

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._kv_cache = RollingKVCache(
            max_length=self.config.kv_cache_max_length,
            truncate_to=self.config.kv_cache_truncate_to,
            detach_interval=self.config.kv_cache_detach_interval,
            num_layers=num_layers,
            device=device
        )

        return self._kv_cache

    def reset_kv_cache(self) -> None:
        """Reset the KV-cache for a new sequence."""
        if self._kv_cache is not None:
            self._kv_cache.reset()

    def get_kv_cache(self) -> Optional['RollingKVCache']:
        """Get the current KV-cache."""
        return self._kv_cache

    def chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: Optional[int] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Memory-efficient chunked attention computation.

        Processes attention in chunks to reduce memory usage for long sequences.

        Args:
            query: Query tensor [B, H, T_q, D]
            key: Key tensor [B, H, T_k, D]
            value: Value tensor [B, H, T_k, D]
            chunk_size: Size of chunks for processing (default from config)
            mask: Optional attention mask

        Returns:
            Attention output [B, H, T_q, D]
        """
        chunk_size = chunk_size or self.config.chunk_size
        B, H, T_q, D = query.shape
        T_k = key.shape[2]

        # If sequence is short, use standard attention
        if T_q <= chunk_size:
            scale = D ** -0.5
            scores = torch.einsum('bhqd,bhkd->bhqk', query, key) * scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            return torch.einsum('bhqk,bhkd->bhqd', attn, value)

        # Chunked processing
        outputs = []
        scale = D ** -0.5

        for start in range(0, T_q, chunk_size):
            end = min(start + chunk_size, T_q)
            q_chunk = query[:, :, start:end]

            # Compute attention for this chunk
            scores = torch.einsum('bhqd,bhkd->bhqk', q_chunk, key) * scale

            if mask is not None:
                mask_chunk = mask[:, :, start:end]
                scores = scores.masked_fill(mask_chunk == 0, float('-inf'))

            attn = F.softmax(scores, dim=-1)
            out_chunk = torch.einsum('bhqk,bhkd->bhqd', attn, value)
            outputs.append(out_chunk)

        return torch.cat(outputs, dim=2)

    def auto_tune_batch_size(
        self,
        model: nn.Module,
        example_input_fn: Callable[[int], torch.Tensor],
        target_memory_fraction: float = 0.8,
        verbose: bool = True
    ) -> int:
        """
        Automatically find optimal batch size based on available memory.

        Args:
            model: Model to test
            example_input_fn: Function that creates input given batch size
            target_memory_fraction: Target memory usage fraction
            verbose: Print tuning progress

        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return self.config.min_batch_size

        # Binary search for optimal batch size
        min_bs = self.config.min_batch_size
        max_bs = self.config.max_batch_size
        optimal_bs = min_bs

        model.eval()

        while min_bs <= max_bs:
            current_bs = (min_bs + max_bs) // 2

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            try:
                # Test this batch size
                example_input = example_input_fn(current_bs)

                with torch.inference_mode():
                    with autocast(dtype=self._dtype):
                        _ = model(example_input)

                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

                if memory_used < target_memory_fraction:
                    optimal_bs = current_bs
                    min_bs = current_bs + 1
                    if verbose:
                        print(f"Batch size {current_bs}: OK "
                              f"(memory: {memory_used*100:.1f}%)")
                else:
                    max_bs = current_bs - 1
                    if verbose:
                        print(f"Batch size {current_bs}: Too large "
                              f"(memory: {memory_used*100:.1f}%)")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    max_bs = current_bs - 1
                    torch.cuda.empty_cache()
                    if verbose:
                        print(f"Batch size {current_bs}: OOM")
                else:
                    raise

        self._optimal_batch_size = optimal_bs

        if verbose:
            print(f"Optimal batch size: {optimal_bs}")

        return optimal_bs

    def get_optimal_batch_size(self) -> Optional[int]:
        """Get the auto-tuned optimal batch size."""
        return self._optimal_batch_size

    def cleanup_memory(self, force: bool = False) -> None:
        """
        Clean up GPU memory.

        Args:
            force: Force cleanup regardless of threshold
        """
        if not torch.cuda.is_available():
            return

        memory_fraction = (
            torch.cuda.memory_allocated() /
            torch.cuda.max_memory_reserved()
            if torch.cuda.max_memory_reserved() > 0 else 0
        )

        if force or memory_fraction > self.config.memory_cleanup_threshold:
            torch.cuda.empty_cache()
            gc.collect()


# =============================================================================
# Streaming Inference
# =============================================================================

@dataclass
class FrameMetrics:
    """Metrics for a single generated frame."""
    frame_idx: int
    generation_time_ms: float
    total_latency_ms: float
    memory_mb: float
    kv_cache_length: int = 0


class StreamingInference:
    """
    Real-time streaming video generation with frame-by-frame output.

    Features:
    - Frame-by-frame generation with configurable buffer
    - Latency tracking and optimization
    - Integration with RollingKVCache
    - Automatic memory management

    Performance targets:
    - LongLive: 20.7 FPS real-time interactive
    - Self-Forcing: 16 FPS streaming generation

    Example:
        streaming = StreamingInference(model, config)

        # Generate frames as an iterator
        for frame, metrics in streaming.generate_stream(context, controls):
            display(frame)
            print(f"Latency: {metrics.total_latency_ms}ms")
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OptimizationConfig] = None,
        device: str = 'cuda'
    ):
        """
        Initialize streaming inference.

        Args:
            model: World model for generation
            config: Optimization configuration
            device: Device for inference
        """
        self.model = model
        self.config = config or OptimizationConfig()
        self.device = torch.device(device)

        # Initialize optimizer
        self.optimizer = InferenceOptimizer(self.config)

        # Frame buffer
        self.frame_buffer: deque = deque(maxlen=self.config.frame_buffer_size)

        # Metrics tracking
        self.frame_metrics: List[FrameMetrics] = []
        self._latency_history: deque = deque(maxlen=100)

        # KV-cache (will be initialized on first use)
        self._kv_cache: Optional[RollingKVCache] = None

        # Timing
        self._frame_deadline_ms = 1000.0 / self.config.target_fps

        # State
        self._is_streaming = False
        self._current_context: Optional[torch.Tensor] = None

    def initialize(
        self,
        num_layers: int = 12,
        compile_model: bool = True,
        example_input: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initialize streaming infrastructure.

        Args:
            num_layers: Number of transformer layers for KV-cache
            compile_model: Whether to compile the model
            example_input: Example input for warmup
        """
        # Optimize model
        if compile_model:
            self.model = self.optimizer.optimize_model(
                self.model,
                example_input=example_input,
                compile_model=True,
                warmup=example_input is not None
            )

        # Initialize KV-cache
        self._kv_cache = self.optimizer.initialize_kv_cache(
            num_layers=num_layers,
            device=self.device
        )

        # Move model to device
        self.model = self.model.to(self.device)

    def reset(self) -> None:
        """Reset streaming state for a new sequence."""
        self.frame_buffer.clear()
        self.frame_metrics.clear()
        self._latency_history.clear()

        if self._kv_cache is not None:
            self._kv_cache.reset()

        self._current_context = None
        self._is_streaming = False

        # Clear memory
        self.optimizer.cleanup_memory(force=True)

    @torch.inference_mode()
    def generate_frame(
        self,
        context: torch.Tensor,
        control: Optional[torch.Tensor] = None,
        frame_idx: int = 0
    ) -> Tuple[torch.Tensor, FrameMetrics]:
        """
        Generate a single frame.

        Args:
            context: Context frames [B, T, C, H, W] or last frame [B, C, H, W]
            control: Control signal [B, control_dim]
            frame_idx: Current frame index

        Returns:
            Tuple of (generated_frame, metrics)
        """
        start_time = time.perf_counter()

        # Ensure context has time dimension
        if context.dim() == 4:
            context = context.unsqueeze(1)

        context = context.to(self.device)
        if control is not None:
            control = control.to(self.device)

        # Generate with autocast
        with self.optimizer.inference_context():
            # Forward pass through model
            if hasattr(self.model, 'forward'):
                output = self.model(
                    frames=context,
                    controls=control.unsqueeze(1) if control is not None else None,
                    mode='inference',
                    num_steps=1
                )

                if isinstance(output, dict):
                    frame = output.get('generated_frames', output.get('predictions'))
                    if frame is not None:
                        frame = frame[:, -1]  # Get last frame
                else:
                    frame = output
            else:
                frame = context[:, -1]  # Fallback

        # Sync for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        generation_time = (time.perf_counter() - start_time) * 1000

        # Get memory usage
        memory_mb = 0.0
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated(self.device) / 1e6

        # Get KV-cache length
        kv_length = 0
        if self._kv_cache is not None:
            kv_length = self._kv_cache.get_length()

        # Create metrics
        metrics = FrameMetrics(
            frame_idx=frame_idx,
            generation_time_ms=generation_time,
            total_latency_ms=generation_time,  # Will be updated in streaming
            memory_mb=memory_mb,
            kv_cache_length=kv_length
        )

        self.frame_metrics.append(metrics)
        self._latency_history.append(generation_time)

        return frame, metrics

    def generate_stream(
        self,
        context_frames: torch.Tensor,
        control_sequence: Optional[torch.Tensor] = None,
        max_frames: int = 100,
        yield_buffer: bool = False
    ) -> Iterator[Tuple[torch.Tensor, FrameMetrics]]:
        """
        Generate frames as a streaming iterator.

        Args:
            context_frames: Initial context [B, T_ctx, C, H, W]
            control_sequence: Control signals [B, T_total, control_dim]
            max_frames: Maximum frames to generate
            yield_buffer: Whether to yield buffered frames

        Yields:
            Tuples of (frame, metrics)
        """
        self.reset()
        self._is_streaming = True

        B, T_ctx, C, H, W = context_frames.shape

        # Initialize with context
        self._current_context = context_frames.to(self.device)

        # Add context to buffer
        for t in range(T_ctx):
            self.frame_buffer.append(context_frames[:, t])

        # Generate frames
        frame_idx = 0
        stream_start = time.perf_counter()

        try:
            for t in range(max_frames):
                # Get control for this step
                control = None
                if control_sequence is not None:
                    ctrl_idx = min(T_ctx + t, control_sequence.size(1) - 1)
                    control = control_sequence[:, ctrl_idx]

                # Generate next frame
                frame, metrics = self.generate_frame(
                    context=self._current_context,
                    control=control,
                    frame_idx=frame_idx
                )

                # Update total latency
                metrics.total_latency_ms = (time.perf_counter() - stream_start) * 1000

                # Update context (sliding window)
                self._current_context = torch.cat([
                    self._current_context[:, 1:],
                    frame.unsqueeze(1)
                ], dim=1)

                # Add to buffer
                self.frame_buffer.append(frame)

                # Yield frame
                yield frame, metrics

                frame_idx += 1

                # Memory management
                if frame_idx % 10 == 0:
                    self.optimizer.cleanup_memory()

        finally:
            self._is_streaming = False

    def generate_batch(
        self,
        context_frames: torch.Tensor,
        control_sequence: Optional[torch.Tensor] = None,
        num_frames: int = 16
    ) -> Tuple[torch.Tensor, List[FrameMetrics]]:
        """
        Generate a batch of frames (non-streaming).

        Args:
            context_frames: Initial context [B, T_ctx, C, H, W]
            control_sequence: Control signals [B, T_total, control_dim]
            num_frames: Number of frames to generate

        Returns:
            Tuple of (generated_frames [B, T, C, H, W], metrics)
        """
        frames = []
        metrics = []

        for frame, metric in self.generate_stream(
            context_frames, control_sequence, max_frames=num_frames
        ):
            frames.append(frame)
            metrics.append(metric)

        return torch.stack(frames, dim=1), metrics

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics."""
        if not self._latency_history:
            return {}

        latencies = list(self._latency_history)
        latencies_sorted = sorted(latencies)

        return {
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p50_latency_ms': latencies_sorted[len(latencies) // 2],
            'p95_latency_ms': latencies_sorted[int(len(latencies) * 0.95)],
            'current_fps': 1000.0 / (sum(latencies) / len(latencies)),
            'target_fps': self.config.target_fps,
            'frame_deadline_ms': self._frame_deadline_ms,
            'frames_generated': len(self.frame_metrics),
            'kv_cache_length': self._kv_cache.get_length() if self._kv_cache else 0
        }

    def is_meeting_target_fps(self) -> bool:
        """Check if current performance meets target FPS."""
        if not self._latency_history:
            return True

        avg_latency = sum(self._latency_history) / len(self._latency_history)
        return avg_latency <= self._frame_deadline_ms


# =============================================================================
# Memory Tracker
# =============================================================================

class MemoryTracker:
    """Track GPU memory usage over time."""

    def __init__(self, history_size: int = 1000):
        """
        Initialize memory tracker.

        Args:
            history_size: Number of measurements to keep
        """
        self.history: deque = deque(maxlen=history_size)
        self._peak_memory: float = 0.0

    def record(self, label: str = "") -> Dict[str, float]:
        """
        Record current memory state.

        Args:
            label: Optional label for this measurement

        Returns:
            Dictionary with memory metrics
        """
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        max_allocated = torch.cuda.max_memory_allocated() / 1e6

        self._peak_memory = max(self._peak_memory, allocated)

        measurement = {
            'timestamp': time.time(),
            'label': label,
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated
        }

        self.history.append(measurement)
        return measurement

    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return self._peak_memory

    def get_average_memory(self) -> float:
        """Get average memory usage in MB."""
        if not self.history:
            return 0.0
        return sum(m['allocated_mb'] for m in self.history) / len(self.history)

    def reset(self) -> None:
        """Reset tracking state."""
        self.history.clear()
        self._peak_memory = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_report(self) -> Dict[str, Any]:
        """Get memory usage report."""
        if not self.history:
            return {'error': 'No measurements recorded'}

        return {
            'peak_mb': self._peak_memory,
            'average_mb': self.get_average_memory(),
            'current_mb': self.history[-1]['allocated_mb'] if self.history else 0,
            'num_measurements': len(self.history)
        }


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    num_warmup: int = 10,
    num_iterations: int = 100,
    use_autocast: bool = True,
    dtype: torch.dtype = torch.float16,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Comprehensive inference benchmarking.

    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        use_autocast: Use mixed precision
        dtype: Data type for autocast
        verbose: Print progress

    Returns:
        BenchmarkResult with detailed metrics
    """
    model.eval()
    device_obj = torch.device(device)
    model = model.to(device_obj)

    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device_obj)
    if use_autocast:
        dummy_input = dummy_input.to(dtype)

    memory_tracker = MemoryTracker()
    latencies = []

    # Warmup
    if verbose:
        print(f"Running {num_warmup} warmup iterations...")

    with torch.inference_mode():
        for i in range(num_warmup):
            if use_autocast and device == 'cuda':
                with autocast(dtype=dtype):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

    # Clear memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if verbose:
        print(f"Running {num_iterations} benchmark iterations...")

    with torch.inference_mode():
        for i in range(num_iterations):
            # Sync before
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()

            if use_autocast and device == 'cuda':
                with autocast(dtype=dtype):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)

            # Sync after
            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

            # Record memory
            memory_tracker.record(f"iter_{i}")

    # Compute statistics
    latencies_sorted = sorted(latencies)
    n = len(latencies)

    result = BenchmarkResult(
        avg_latency_ms=sum(latencies) / n,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        p50_latency_ms=latencies_sorted[n // 2],
        p95_latency_ms=latencies_sorted[int(n * 0.95)],
        p99_latency_ms=latencies_sorted[int(n * 0.99)],
        fps=1000.0 / (sum(latencies) / n),
        throughput_samples_per_sec=input_shape[0] * 1000.0 / (sum(latencies) / n),
        peak_memory_mb=memory_tracker.get_peak_memory(),
        avg_memory_mb=memory_tracker.get_average_memory(),
        memory_reserved_mb=torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0,
        batch_size=input_shape[0],
        sequence_length=input_shape[1] if len(input_shape) > 1 else 1,
        warmup_iterations=num_warmup,
        benchmark_iterations=num_iterations,
        device=device
    )

    if verbose:
        print(result.summary())

    return result


def profile_bottlenecks(
    model: nn.Module,
    example_input: torch.Tensor,
    num_iterations: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Identify performance bottlenecks in a model.

    Uses PyTorch profiler to identify slow operations.

    Args:
        model: Model to profile
        example_input: Example input tensor
        num_iterations: Number of profiling iterations
        verbose: Print detailed results

    Returns:
        Dictionary with profiling results
    """
    model.eval()
    device = example_input.device

    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        with torch.inference_mode():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    with autocast(dtype=torch.float16):
                        _ = model(example_input)
                    torch.cuda.synchronize()
                else:
                    _ = model(example_input)

    # Get top operations by time
    key_averages = prof.key_averages()

    # Sort by CUDA time if available, else CPU time
    if torch.cuda.is_available():
        sorted_ops = sorted(
            key_averages,
            key=lambda x: x.cuda_time_total,
            reverse=True
        )
    else:
        sorted_ops = sorted(
            key_averages,
            key=lambda x: x.cpu_time_total,
            reverse=True
        )

    # Extract top bottlenecks
    bottlenecks = []
    total_time = sum(op.cuda_time_total if torch.cuda.is_available()
                     else op.cpu_time_total for op in sorted_ops)

    for op in sorted_ops[:20]:
        op_time = op.cuda_time_total if torch.cuda.is_available() else op.cpu_time_total
        bottlenecks.append({
            'name': op.key,
            'time_us': op_time,
            'percentage': (op_time / total_time * 100) if total_time > 0 else 0,
            'calls': op.count,
            'memory_mb': op.self_cuda_memory_usage / 1e6 if torch.cuda.is_available() else 0
        })

    results = {
        'bottlenecks': bottlenecks,
        'total_time_us': total_time,
        'num_operations': len(key_averages),
        'profile_iterations': num_iterations
    }

    if verbose:
        print("\nTop Performance Bottlenecks:")
        print("-" * 70)
        for i, b in enumerate(bottlenecks[:10], 1):
            print(f"{i:2d}. {b['name'][:40]:40s} "
                  f"{b['time_us']/1000:8.2f}ms ({b['percentage']:5.1f}%) "
                  f"[{b['calls']} calls]")

    return results


def optimize_for_deployment(
    model: nn.Module,
    example_input: torch.Tensor,
    optimization_level: str = 'balanced',
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply all optimizations for production deployment.

    Args:
        model: Model to optimize
        example_input: Example input for tracing/warmup
        optimization_level: 'minimal', 'balanced', or 'aggressive'
        verbose: Print optimization progress

    Returns:
        Tuple of (optimized_model, optimization_report)
    """
    report = {
        'original_latency_ms': None,
        'optimized_latency_ms': None,
        'speedup': None,
        'optimizations_applied': []
    }

    # Define optimization levels
    levels = {
        'minimal': OptimizationConfig(
            compile_mode=CompileMode.NONE,
            precision_mode=PrecisionMode.FP32,
            use_autocast=False
        ),
        'balanced': OptimizationConfig(
            compile_mode=CompileMode.REDUCE_OVERHEAD,
            precision_mode=PrecisionMode.FP16,
            use_autocast=True,
            warmup_iterations=3
        ),
        'aggressive': OptimizationConfig(
            compile_mode=CompileMode.MAX_AUTOTUNE,
            precision_mode=PrecisionMode.FP16,
            use_autocast=True,
            warmup_iterations=5,
            compile_fullgraph=True
        )
    }

    config = levels.get(optimization_level, levels['balanced'])

    # Benchmark original
    if verbose:
        print("Benchmarking original model...")

    original_result = benchmark_inference(
        model,
        example_input.shape,
        num_warmup=5,
        num_iterations=20,
        use_autocast=False,
        verbose=False
    )
    report['original_latency_ms'] = original_result.avg_latency_ms

    # Create optimizer
    optimizer = InferenceOptimizer(config)

    # Apply optimizations
    if verbose:
        print(f"Applying {optimization_level} optimizations...")

    # Optimize model
    optimized_model = optimizer.optimize_model(
        model,
        example_input=example_input,
        compile_model=config.compile_mode != CompileMode.NONE,
        warmup=True
    )

    report['optimizations_applied'].append(f'precision_{config.precision_mode.value}')

    if config.compile_mode != CompileMode.NONE:
        report['optimizations_applied'].append(f'compile_{config.compile_mode.value}')

    if config.use_autocast:
        report['optimizations_applied'].append('autocast')

    # Benchmark optimized
    if verbose:
        print("Benchmarking optimized model...")

    optimized_result = benchmark_inference(
        optimized_model,
        example_input.shape,
        num_warmup=5,
        num_iterations=20,
        use_autocast=config.use_autocast,
        dtype=torch.float16 if config.precision_mode == PrecisionMode.FP16 else torch.bfloat16,
        verbose=False
    )

    report['optimized_latency_ms'] = optimized_result.avg_latency_ms
    report['speedup'] = report['original_latency_ms'] / report['optimized_latency_ms']

    if verbose:
        print(f"\nOptimization Results:")
        print(f"  Original: {report['original_latency_ms']:.2f}ms")
        print(f"  Optimized: {report['optimized_latency_ms']:.2f}ms")
        print(f"  Speedup: {report['speedup']:.2f}x")
        print(f"  Optimizations: {', '.join(report['optimizations_applied'])}")

    return optimized_model, report


def compare_optimization_modes(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, BenchmarkResult]:
    """
    Compare different optimization modes.

    Args:
        model: Model to benchmark
        input_shape: Input shape
        device: Device to run on
        verbose: Print comparison table

    Returns:
        Dictionary mapping mode name to BenchmarkResult
    """
    results = {}

    modes = [
        ('baseline', CompileMode.NONE, False),
        ('fp16', CompileMode.NONE, True),
        ('compiled', CompileMode.REDUCE_OVERHEAD, False),
        ('compiled_fp16', CompileMode.REDUCE_OVERHEAD, True),
        ('max_autotune', CompileMode.MAX_AUTOTUNE, True),
    ]

    for name, compile_mode, use_fp16 in modes:
        if verbose:
            print(f"\nBenchmarking: {name}")

        try:
            # Create fresh model copy
            test_model = model

            # Apply compilation
            if compile_mode != CompileMode.NONE:
                compiler = ModelCompiler(mode=compile_mode)
                test_model = compiler.compile(test_model, verbose=False)

            # Benchmark
            result = benchmark_inference(
                test_model,
                input_shape,
                device=device,
                num_warmup=5,
                num_iterations=50,
                use_autocast=use_fp16,
                verbose=False
            )

            results[name] = result

        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
            results[name] = None

    # Print comparison table
    if verbose:
        print("\n" + "=" * 80)
        print("Optimization Mode Comparison")
        print("=" * 80)
        print(f"{'Mode':<20} {'Latency (ms)':<15} {'FPS':<10} {'Memory (MB)':<15} {'Speedup':<10}")
        print("-" * 80)

        baseline_latency = results.get('baseline')
        if baseline_latency:
            baseline_latency = baseline_latency.avg_latency_ms

        for name, result in results.items():
            if result:
                speedup = (baseline_latency / result.avg_latency_ms
                          if baseline_latency else 1.0)
                print(f"{name:<20} {result.avg_latency_ms:<15.2f} "
                      f"{result.fps:<10.1f} {result.peak_memory_mb:<15.1f} "
                      f"{speedup:<10.2f}x")
            else:
                print(f"{name:<20} {'Failed':<15}")

    return results


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("DriveDiT Inference Optimization Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test 1: ModelCompiler
    print("\n1. ModelCompiler Test")
    print("-" * 40)

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleModel().to(device)
    compiler = ModelCompiler(mode=CompileMode.REDUCE_OVERHEAD)

    # Test compilation
    compiled_model = compiler.compile(model, verbose=True)
    print(f"Is compiled: {compiler.is_compiled(compiled_model)}")

    # Test warmup
    example_input = torch.randn(8, 256, device=device)
    compiler.warmup(compiled_model, example_input, iterations=3, verbose=True)
    print(f"Is warmed up: {compiler.is_warmed_up(compiled_model)}")

    # Test 2: InferenceOptimizer
    print("\n2. InferenceOptimizer Test")
    print("-" * 40)

    config = OptimizationConfig(
        compile_mode=CompileMode.REDUCE_OVERHEAD,
        precision_mode=PrecisionMode.FP16,
        warmup_iterations=2
    )

    optimizer = InferenceOptimizer(config)

    # Create new model for optimization
    model2 = SimpleModel().to(device)
    optimized_model = optimizer.optimize_model(
        model2,
        example_input=example_input,
        compile_model=True,
        warmup=True
    )

    # Test inference context
    with optimizer.inference_context():
        output = optimized_model(example_input.half())
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")

    # Test chunked attention
    print("\nChunked attention test:")
    B, H, T, D = 2, 8, 64, 64
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    attn_out = optimizer.chunked_attention(q, k, v, chunk_size=16)
    print(f"Chunked attention output shape: {attn_out.shape}")

    # Test 3: StreamingInference
    print("\n3. StreamingInference Test")
    print("-" * 40)

    # Create a mock world model
    class MockWorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3 * 64 * 64, 512)
            self.decoder = nn.Linear(512, 3 * 64 * 64)

        def forward(self, frames, controls=None, mode='train', **kwargs):
            B, T, C, H, W = frames.shape
            flat = frames.view(B * T, -1)
            encoded = self.encoder(flat)
            decoded = self.decoder(encoded)
            output = decoded.view(B, T, C, H, W)
            return {'generated_frames': output, 'predictions': output}

    mock_model = MockWorldModel().to(device)

    streaming_config = OptimizationConfig(
        compile_mode=CompileMode.NONE,  # Skip compilation for quick test
        precision_mode=PrecisionMode.FP16,
        frame_buffer_size=4,
        target_fps=30.0
    )

    streaming = StreamingInference(mock_model, streaming_config, device=str(device))
    streaming.initialize(num_layers=12, compile_model=False)

    # Test frame generation
    context = torch.randn(1, 4, 3, 64, 64, device=device)
    controls = torch.randn(1, 20, 6, device=device)

    print("Generating streaming frames...")
    frame_count = 0
    for frame, metrics in streaming.generate_stream(context, controls, max_frames=10):
        frame_count += 1
        if frame_count <= 3 or frame_count == 10:
            print(f"  Frame {metrics.frame_idx}: {metrics.generation_time_ms:.2f}ms, "
                  f"memory: {metrics.memory_mb:.1f}MB")

    stats = streaming.get_streaming_stats()
    print(f"\nStreaming stats:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  Current FPS: {stats['current_fps']:.1f}")
    print(f"  Meeting target: {streaming.is_meeting_target_fps()}")

    # Test 4: Benchmarking
    print("\n4. Benchmarking Test")
    print("-" * 40)

    model3 = SimpleModel().to(device)
    result = benchmark_inference(
        model3,
        input_shape=(8, 256),
        device=str(device),
        num_warmup=5,
        num_iterations=50,
        use_autocast=True,
        verbose=True
    )

    # Test 5: Bottleneck profiling
    print("\n5. Bottleneck Profiling Test")
    print("-" * 40)

    profile_input = torch.randn(8, 256, device=device)
    bottlenecks = profile_bottlenecks(
        model3,
        profile_input,
        num_iterations=5,
        verbose=True
    )

    # Test 6: Deployment optimization
    print("\n6. Deployment Optimization Test")
    print("-" * 40)

    model4 = SimpleModel().to(device)
    opt_input = torch.randn(8, 256, device=device)

    optimized, report = optimize_for_deployment(
        model4,
        opt_input,
        optimization_level='balanced',
        verbose=True
    )

    # Test 7: Mode comparison
    print("\n7. Optimization Mode Comparison Test")
    print("-" * 40)

    model5 = SimpleModel().to(device)
    comparison = compare_optimization_modes(
        model5,
        input_shape=(8, 256),
        device=str(device),
        verbose=True
    )

    print("\n" + "=" * 60)
    print("All optimization tests completed!")

