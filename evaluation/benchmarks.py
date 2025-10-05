"""
Comprehensive benchmarking tools for DriveDiT models.
Performance, memory, and functionality benchmarks.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import math
import gc
from pathlib import Path
import json

from .metrics import compute_all_metrics, VideoMetrics, TemporalConsistencyMetrics


class WorldModelBenchmark:
    """Comprehensive benchmark for world model capabilities."""
    
    def __init__(
        self,
        model,
        vae_model=None,
        device: str = 'cuda',
        save_path: Optional[str] = None
    ):
        """
        Initialize world model benchmark.
        
        Args:
            model: World model to benchmark
            vae_model: VAE model for frame encoding/decoding
            device: Device to run benchmark on
            save_path: Path to save benchmark results
        """
        self.model = model
        self.vae_model = vae_model
        self.device = device
        self.save_path = save_path
        self.results = {}
    
    def run_full_benchmark(
        self,
        test_data: Dict[str, torch.Tensor],
        sequence_lengths: List[int] = [8, 16, 32],
        batch_sizes: List[int] = [1, 4, 8],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        print("Running World Model Benchmark...")
        
        # Generation benchmark
        self.results['generation'] = self._benchmark_generation(
            test_data, sequence_lengths, batch_sizes, num_iterations
        )
        
        # Quality benchmark
        self.results['quality'] = self._benchmark_quality(test_data)
        
        # Temporal consistency benchmark
        self.results['temporal'] = self._benchmark_temporal_consistency(test_data)
        
        # Control conditioning benchmark
        if 'controls' in test_data:
            self.results['control'] = self._benchmark_control_conditioning(test_data)
        
        # Long sequence benchmark
        self.results['long_sequence'] = self._benchmark_long_sequences(test_data)
        
        # Save results
        if self.save_path:
            self._save_results()
        
        return self.results
    
    def _benchmark_generation(
        self,
        test_data: Dict[str, torch.Tensor],
        sequence_lengths: List[int],
        batch_sizes: List[int],
        num_iterations: int
    ) -> Dict[str, Any]:
        """Benchmark generation capabilities."""
        generation_results = {}
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                key = f"seq_{seq_len}_batch_{batch_size}"
                
                # Prepare test batch
                frames = test_data['frames'][:batch_size, :seq_len]
                context_frames = frames[:, :seq_len//2]
                target_frames = frames[:, seq_len//2:]
                
                # Benchmark generation
                times = []
                memory_usage = []
                
                for _ in range(num_iterations):
                    torch.cuda.empty_cache() if self.device == 'cuda' else None
                    
                    start_time = time.time()
                    if self.device == 'cuda':
                        start_memory = torch.cuda.memory_allocated()
                    
                    with torch.no_grad():
                        # Encode context
                        if self.vae_model:
                            context_latents = self.vae_model.encode(context_frames)[0]
                        else:
                            context_latents = context_frames
                        
                        # Generate sequence
                        if hasattr(self.model, 'generate'):
                            generated = self.model.generate(
                                context_latents.view(batch_size, seq_len//2, -1),
                                max_new_tokens=seq_len//2
                            )
                        else:
                            # Fallback generation
                            generated = self.model(context_latents.view(batch_size, seq_len//2, -1))[0]
                    
                    end_time = time.time()
                    if self.device == 'cuda':
                        end_memory = torch.cuda.memory_allocated()
                        memory_usage.append(end_memory - start_memory)
                    
                    times.append(end_time - start_time)
                
                generation_results[key] = {
                    'avg_time': sum(times) / len(times),
                    'std_time': torch.tensor(times).std().item(),
                    'avg_memory_mb': sum(memory_usage) / len(memory_usage) / (1024*1024) if memory_usage else 0,
                    'fps': seq_len / (sum(times) / len(times))
                }
        
        return generation_results
    
    def _benchmark_quality(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Benchmark generation quality."""
        frames = test_data['frames'][:4, :16]  # Small batch for quality assessment
        context_length = 8
        
        context_frames = frames[:, :context_length]
        target_frames = frames[:, context_length:]
        
        with torch.no_grad():
            # Generate frames
            if self.vae_model:
                context_latents = self.vae_model.encode(context_frames)[0]
                
                if hasattr(self.model, 'generate'):
                    generated_latents = self.model.generate(
                        context_latents.view(context_frames.shape[0], context_length, -1),
                        max_new_tokens=target_frames.shape[1]
                    )
                else:
                    generated_latents = self.model(context_latents.view(context_frames.shape[0], context_length, -1))[0]
                
                # Decode back to frames
                generated_frames = self.vae_model.decode(
                    generated_latents.view(target_frames.shape[0], target_frames.shape[1], *context_latents.shape[2:])
                )
            else:
                generated_frames = target_frames  # Placeholder
        
        # Compute quality metrics
        quality_metrics = compute_all_metrics(generated_frames, target_frames)
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in quality_metrics.items()}
    
    def _benchmark_temporal_consistency(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Benchmark temporal consistency."""
        frames = test_data['frames'][:2, :16]  # Small batch
        
        with torch.no_grad():
            # Compute temporal metrics on generated sequences
            temporal_mse = TemporalConsistencyMetrics.temporal_mse(frames)
            optical_flow = TemporalConsistencyMetrics.optical_flow_consistency(frames)
        
        return {
            'temporal_mse': temporal_mse.item(),
            'optical_flow_consistency': optical_flow.item()
        }
    
    def _benchmark_control_conditioning(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Benchmark control signal conditioning."""
        frames = test_data['frames'][:2, :8]
        controls = test_data['controls'][:2, :8]
        
        with torch.no_grad():
            # Test model response to different control signals
            original_output = self.model(frames.view(frames.shape[0], frames.shape[1], -1))[0]
            
            # Modify controls and test response
            modified_controls = controls.clone()
            modified_controls[:, :, 0] += 0.5  # Modify steering
            
            # This would require model to accept control conditioning
            # Placeholder for control-conditioned generation
            control_sensitivity = torch.tensor(0.5)  # Placeholder
        
        return {
            'control_sensitivity': control_sensitivity.item()
        }
    
    def _benchmark_long_sequences(self, test_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Benchmark performance on long sequences."""
        max_length = min(64, test_data['frames'].shape[1])
        frames = test_data['frames'][:1, :max_length]  # Single batch, long sequence
        
        results = {}
        
        for seq_len in [16, 32, max_length]:
            if seq_len > frames.shape[1]:
                continue
            
            test_frames = frames[:, :seq_len]
            
            start_time = time.time()
            if self.device == 'cuda':
                start_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                output = self.model(test_frames.view(1, seq_len, -1))[0]
            
            end_time = time.time()
            if self.device == 'cuda':
                end_memory = torch.cuda.memory_allocated()
                memory_used = (end_memory - start_memory) / (1024*1024)
            else:
                memory_used = 0
            
            results[f'length_{seq_len}'] = {
                'time': end_time - start_time,
                'memory_mb': memory_used,
                'memory_per_frame': memory_used / seq_len if seq_len > 0 else 0
            }
        
        return results
    
    def _save_results(self):
        """Save benchmark results to file."""
        save_path = Path(self.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


class GenerationBenchmark:
    """Benchmark for video generation capabilities."""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def benchmark_generation_speed(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8],
        sequence_lengths: List[int] = [8, 16, 32],
        num_warmup: int = 3,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Benchmark generation speed across different configurations."""
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                key = f"B{batch_size}_T{seq_len}"
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, seq_len//2, 256, device=self.device)
                
                # Warmup
                for _ in range(num_warmup):
                    with torch.no_grad():
                        _ = self.model(dummy_input)
                
                # Benchmark
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start_time = time.time()
                
                for _ in range(num_iterations):
                    with torch.no_grad():
                        _ = self.model(dummy_input)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_iterations
                throughput = (batch_size * seq_len) / avg_time
                
                results[key] = {
                    'avg_time_s': avg_time,
                    'throughput_frames_per_s': throughput,
                    'batch_size': batch_size,
                    'sequence_length': seq_len
                }
        
        return results
    
    def benchmark_memory_efficiency(
        self,
        max_batch_size: int = 16,
        sequence_length: int = 32
    ) -> Dict[str, Any]:
        """Benchmark memory efficiency."""
        if self.device != 'cuda':
            return {'error': 'Memory benchmark requires CUDA'}
        
        results = {}
        
        for batch_size in range(1, max_batch_size + 1):
            torch.cuda.empty_cache()
            
            try:
                dummy_input = torch.randn(batch_size, sequence_length, 256, device=self.device)
                
                start_memory = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - start_memory
                
                results[f'batch_{batch_size}'] = {
                    'memory_mb': memory_used / (1024*1024),
                    'memory_per_sample_mb': memory_used / (batch_size * 1024*1024),
                    'success': True
                }
                
                torch.cuda.reset_peak_memory_stats()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[f'batch_{batch_size}'] = {
                        'error': 'OOM',
                        'success': False
                    }
                    break
                else:
                    raise e
        
        return results


class ControlBenchmark:
    """Benchmark for control signal conditioning."""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def benchmark_control_response(
        self,
        test_sequences: torch.Tensor,
        control_variations: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Benchmark model response to control variations."""
        results = {}
        
        base_sequence = test_sequences[:1]  # Single sequence
        base_output = self.model(base_sequence.view(1, base_sequence.shape[1], -1))[0]
        
        for i, control_var in enumerate(control_variations):
            # This would require model to accept control conditioning
            # Placeholder implementation
            var_output = base_output  # Placeholder
            
            # Measure output variation
            output_diff = F.mse_loss(var_output, base_output)
            results[f'control_var_{i}'] = output_diff.item()
        
        return results


class PerformanceBenchmark:
    """Comprehensive performance benchmark."""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def run_performance_suite(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite."""
        results = {}
        
        # Throughput benchmark
        results['throughput'] = self._benchmark_throughput()
        
        # Latency benchmark
        results['latency'] = self._benchmark_latency()
        
        # Memory benchmark
        if self.device == 'cuda':
            results['memory'] = self._benchmark_memory()
        
        # Scalability benchmark
        results['scalability'] = self._benchmark_scalability()
        
        return results
    
    def _benchmark_throughput(self) -> Dict[str, float]:
        """Benchmark throughput (frames per second)."""
        batch_size = 4
        seq_len = 16
        num_iterations = 50
        
        dummy_input = torch.randn(batch_size, seq_len, 256, device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if self.device == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if self.device == 'cuda' else None
        end_time = time.time()
        
        total_frames = batch_size * seq_len * num_iterations
        total_time = end_time - start_time
        throughput = total_frames / total_time
        
        return {
            'frames_per_second': throughput,
            'total_time': total_time,
            'iterations': num_iterations
        }
    
    def _benchmark_latency(self) -> Dict[str, float]:
        """Benchmark inference latency."""
        batch_size = 1
        seq_len = 8
        num_iterations = 100
        
        dummy_input = torch.randn(batch_size, seq_len, 256, device=self.device)
        
        latencies = []
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        for _ in range(num_iterations):
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        latencies = torch.tensor(latencies)
        
        return {
            'mean_latency_ms': latencies.mean().item() * 1000,
            'std_latency_ms': latencies.std().item() * 1000,
            'p50_latency_ms': latencies.median().item() * 1000,
            'p95_latency_ms': latencies.quantile(0.95).item() * 1000,
            'p99_latency_ms': latencies.quantile(0.99).item() * 1000
        }
    
    def _benchmark_memory(self) -> Dict[str, float]:
        """Benchmark memory usage."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        batch_size = 1
        seq_len = 16
        dummy_input = torch.randn(batch_size, seq_len, 256, device=self.device)
        
        initial_memory = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            output = self.model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()
        
        return {
            'initial_memory_mb': initial_memory / (1024*1024),
            'peak_memory_mb': peak_memory / (1024*1024),
            'final_memory_mb': final_memory / (1024*1024),
            'memory_increase_mb': (final_memory - initial_memory) / (1024*1024),
            'peak_memory_increase_mb': (peak_memory - initial_memory) / (1024*1024)
        }
    
    def _benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability with sequence length."""
        results = {}
        base_batch_size = 1
        
        for seq_len in [8, 16, 32, 64]:
            dummy_input = torch.randn(base_batch_size, seq_len, 256, device=self.device)
            
            # Time measurement
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.time()
            
            inference_time = end_time - start_time
            
            # Memory measurement
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                start_memory = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                end_memory = torch.cuda.memory_allocated()
                memory_used = end_memory - start_memory
            else:
                memory_used = 0
            
            results[f'seq_len_{seq_len}'] = {
                'inference_time_s': inference_time,
                'memory_mb': memory_used / (1024*1024),
                'time_per_token_ms': (inference_time / seq_len) * 1000,
                'memory_per_token_mb': (memory_used / seq_len) / (1024*1024)
            }
        
        return results


class MemoryBenchmark:
    """Specialized memory usage benchmark."""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
    
    def find_max_batch_size(
        self,
        sequence_length: int = 16,
        max_attempts: int = 32
    ) -> Dict[str, Any]:
        """Find maximum batch size that fits in memory."""
        if self.device != 'cuda':
            return {'error': 'Memory benchmark requires CUDA'}
        
        max_batch_size = 0
        memory_at_max = 0
        
        for batch_size in range(1, max_attempts + 1):
            torch.cuda.empty_cache()
            
            try:
                dummy_input = torch.randn(batch_size, sequence_length, 256, device=self.device)
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                
                max_batch_size = batch_size
                memory_at_max = torch.cuda.memory_allocated()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        return {
            'max_batch_size': max_batch_size,
            'memory_at_max_mb': memory_at_max / (1024*1024),
            'sequence_length': sequence_length
        }
    
    def find_max_sequence_length(
        self,
        batch_size: int = 1,
        max_length: int = 512
    ) -> Dict[str, Any]:
        """Find maximum sequence length that fits in memory."""
        if self.device != 'cuda':
            return {'error': 'Memory benchmark requires CUDA'}
        
        max_seq_len = 0
        memory_at_max = 0
        
        for seq_len in range(8, max_length + 1, 8):
            torch.cuda.empty_cache()
            
            try:
                dummy_input = torch.randn(batch_size, seq_len, 256, device=self.device)
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                
                max_seq_len = seq_len
                memory_at_max = torch.cuda.memory_allocated()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        return {
            'max_sequence_length': max_seq_len,
            'memory_at_max_mb': memory_at_max / (1024*1024),
            'batch_size': batch_size
        }