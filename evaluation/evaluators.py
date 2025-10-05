"""
High-level evaluators for comprehensive model assessment.
Orchestrates metrics computation and benchmarking.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import numpy as np
from pathlib import Path
import json

from .metrics import compute_all_metrics, VideoMetrics, LatentMetrics, ControlMetrics
from .benchmarks import PerformanceBenchmark, MemoryBenchmark


class SequenceEvaluator:
    """Evaluator for video sequence generation and prediction."""
    
    def __init__(
        self,
        model,
        vae_model=None,
        device: str = 'cuda',
        compute_expensive_metrics: bool = True
    ):
        """
        Initialize sequence evaluator.
        
        Args:
            model: Model to evaluate
            vae_model: VAE model for encoding/decoding
            device: Device to run evaluation on
            compute_expensive_metrics: Whether to compute expensive metrics like SSIM
        """
        self.model = model
        self.vae_model = vae_model
        self.device = device
        self.compute_expensive_metrics = compute_expensive_metrics
        
        self.model.eval()
        if self.vae_model:
            self.vae_model.eval()
    
    def evaluate_sequence(
        self,
        context_frames: torch.Tensor,
        target_frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        return_generated: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single sequence.
        
        Args:
            context_frames: Context frames [B, T_ctx, C, H, W]
            target_frames: Target frames [B, T_target, C, H, W]
            controls: Control signals [B, T_total, control_dim]
            return_generated: Whether to return generated frames
            
        Returns:
            Evaluation results dictionary
        """
        with torch.no_grad():
            # Generate sequence
            start_time = time.time()
            generated_frames = self._generate_sequence(context_frames, target_frames.shape[1], controls)
            generation_time = time.time() - start_time
            
            # Compute metrics
            metrics = self._compute_sequence_metrics(generated_frames, target_frames, controls)
            
            # Add timing information
            metrics['generation_time_s'] = generation_time
            metrics['fps'] = target_frames.shape[1] / generation_time
            
            # Add generated frames if requested
            if return_generated:
                metrics['generated_frames'] = generated_frames
                metrics['context_frames'] = context_frames
                metrics['target_frames'] = target_frames
            
            return metrics
    
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        context_ratio: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on a batch of sequences.
        
        Args:
            batch: Batch containing 'frames' and optionally 'controls'
            context_ratio: Ratio of frames to use as context
            
        Returns:
            Averaged evaluation results
        """
        frames = batch['frames']  # [B, T, C, H, W]
        controls = batch.get('controls')  # [B, T, control_dim]
        
        B, T = frames.shape[:2]
        context_length = int(T * context_ratio)
        
        context_frames = frames[:, :context_length]
        target_frames = frames[:, context_length:]
        
        if controls is not None:
            batch_controls = controls[:, :T]
        else:
            batch_controls = None
        
        # Evaluate each sequence in batch
        all_metrics = []
        
        for i in range(B):
            seq_context = context_frames[i:i+1]
            seq_target = target_frames[i:i+1]
            seq_controls = batch_controls[i:i+1] if batch_controls is not None else None
            
            seq_metrics = self.evaluate_sequence(seq_context, seq_target, seq_controls)
            all_metrics.append(seq_metrics)
        
        # Average metrics across batch
        averaged_metrics = self._average_metrics(all_metrics)
        averaged_metrics['batch_size'] = B
        averaged_metrics['sequence_length'] = T
        averaged_metrics['context_length'] = context_length
        
        return averaged_metrics
    
    def _generate_sequence(
        self,
        context_frames: torch.Tensor,
        target_length: int,
        controls: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate sequence from context frames."""
        if self.vae_model:
            # Encode context frames
            context_mean, context_logvar = self.vae_model.encode(context_frames)
            context_latents = self.vae_model.reparameterize(context_mean, context_logvar)
            
            # Prepare for model input
            B, T, C, H, W = context_latents.shape
            context_tokens = context_latents.view(B, T, C * H * W)
            
            # Generate latent tokens
            if hasattr(self.model, 'generate'):
                generated_tokens = self.model.generate(
                    context_tokens,
                    max_new_tokens=target_length,
                    temperature=0.8
                )
                generated_latents = generated_tokens[:, T:].view(B, target_length, C, H, W)
            else:
                # Fallback: use model forward pass
                full_tokens = self.model(context_tokens)[0]
                if full_tokens.shape[1] > T:
                    generated_latents = full_tokens[:, T:T+target_length].view(B, target_length, C, H, W)
                else:
                    # Repeat last token if needed
                    last_token = full_tokens[:, -1:].view(B, 1, C, H, W)
                    generated_latents = last_token.repeat(1, target_length, 1, 1, 1)
            
            # Decode to frames
            generated_frames = self.vae_model.decode(generated_latents)
            
        else:
            # Direct frame generation (placeholder)
            B, T, C, H, W = context_frames.shape
            generated_frames = torch.randn(B, target_length, C, H, W, device=context_frames.device)
        
        return generated_frames
    
    def _compute_sequence_metrics(
        self,
        generated_frames: torch.Tensor,
        target_frames: torch.Tensor,
        controls: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute metrics for generated sequence."""
        metrics = {}
        
        # Basic video metrics
        metrics['mse'] = VideoMetrics.mse(generated_frames, target_frames).item()
        metrics['mae'] = VideoMetrics.mae(generated_frames, target_frames).item()
        metrics['psnr'] = VideoMetrics.psnr(generated_frames, target_frames).item()
        
        if self.compute_expensive_metrics:
            metrics['ssim'] = VideoMetrics.ssim(generated_frames, target_frames).item()
            metrics['lpips_proxy'] = VideoMetrics.lpips_proxy(generated_frames, target_frames).item()
        
        # Temporal consistency
        from .metrics import TemporalConsistencyMetrics
        metrics['temporal_mse'] = TemporalConsistencyMetrics.temporal_mse(generated_frames).item()
        metrics['optical_flow_consistency'] = TemporalConsistencyMetrics.optical_flow_consistency(generated_frames).item()
        
        # Control metrics if available
        if controls is not None:
            # This would require model to output control predictions
            # Placeholder implementation
            metrics['control_accuracy'] = 0.8  # Placeholder
        
        return metrics
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across multiple evaluations."""
        if not metrics_list:
            return {}
        
        averaged = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
                if values:
                    averaged[key] = sum(values) / len(values)
                    averaged[f'{key}_std'] = torch.tensor(values).std().item() if len(values) > 1 else 0.0
        
        return averaged


class BatchEvaluator:
    """Evaluator for batch processing and efficiency."""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_batch_efficiency(
        self,
        batch_sizes: List[int] = [1, 2, 4, 8, 16],
        sequence_length: int = 16,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """Evaluate batch processing efficiency."""
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Create dummy batch
                dummy_input = torch.randn(batch_size, sequence_length, 256, device=self.device)
                
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model(dummy_input)
                
                # Benchmark
                times = []
                memory_usage = []
                
                for _ in range(num_iterations):
                    torch.cuda.empty_cache() if self.device == 'cuda' else None
                    
                    if self.device == 'cuda':
                        start_memory = torch.cuda.memory_allocated()
                    
                    start_time = time.time()
                    
                    with torch.no_grad():
                        output = self.model(dummy_input)
                    
                    torch.cuda.synchronize() if self.device == 'cuda' else None
                    end_time = time.time()
                    
                    if self.device == 'cuda':
                        end_memory = torch.cuda.memory_allocated()
                        memory_usage.append(end_memory - start_memory)
                    
                    times.append(end_time - start_time)
                
                # Calculate metrics
                avg_time = sum(times) / len(times)
                throughput = (batch_size * sequence_length) / avg_time
                efficiency = throughput / batch_size  # Per-sample efficiency
                
                results[f'batch_{batch_size}'] = {
                    'avg_time_s': avg_time,
                    'throughput_tokens_per_s': throughput,
                    'efficiency_tokens_per_s_per_sample': efficiency,
                    'avg_memory_mb': sum(memory_usage) / len(memory_usage) / (1024*1024) if memory_usage else 0,
                    'memory_per_sample_mb': (sum(memory_usage) / len(memory_usage) / (1024*1024)) / batch_size if memory_usage else 0
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[f'batch_{batch_size}'] = {'error': 'OOM'}
                    break
                else:
                    raise e
        
        return results
    
    def find_optimal_batch_size(
        self,
        sequence_length: int = 16,
        max_batch_size: int = 32,
        target_memory_util: float = 0.8
    ) -> Dict[str, Any]:
        """Find optimal batch size for given constraints."""
        if self.device != 'cuda':
            return {'error': 'Batch size optimization requires CUDA'}
        
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        target_memory = total_memory * target_memory_util
        
        best_batch_size = 1
        best_throughput = 0
        memory_usage = {}
        
        for batch_size in range(1, max_batch_size + 1):
            torch.cuda.empty_cache()
            
            try:
                dummy_input = torch.randn(batch_size, sequence_length, 256, device=self.device)
                
                start_memory = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    output = self.model(dummy_input)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - start_memory
                
                if memory_used <= target_memory:
                    # Measure throughput
                    start_time = time.time()
                    for _ in range(5):
                        with torch.no_grad():
                            _ = self.model(dummy_input)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    throughput = (batch_size * sequence_length * 5) / (end_time - start_time)
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_batch_size = batch_size
                    
                    memory_usage[batch_size] = memory_used
                else:
                    break
                
                torch.cuda.reset_peak_memory_stats()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        return {
            'optimal_batch_size': best_batch_size,
            'max_throughput': best_throughput,
            'memory_usage_mb': {k: v / (1024*1024) for k, v in memory_usage.items()},
            'target_memory_mb': target_memory / (1024*1024)
        }


class RealTimeEvaluator:
    """Evaluator for real-time performance characteristics."""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_real_time_performance(
        self,
        target_fps: float = 30.0,
        sequence_length: int = 8,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Evaluate real-time performance capabilities."""
        target_frame_time = 1.0 / target_fps
        
        dummy_input = torch.randn(1, sequence_length, 256, device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Measure frame times
        frame_times = []
        
        for _ in range(num_samples):
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(dummy_input)
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.time()
            
            frame_time = (end_time - start_time) / sequence_length
            frame_times.append(frame_time)
        
        frame_times = torch.tensor(frame_times)
        
        # Calculate statistics
        mean_frame_time = frame_times.mean().item()
        std_frame_time = frame_times.std().item()
        p95_frame_time = frame_times.quantile(0.95).item()
        p99_frame_time = frame_times.quantile(0.99).item()
        
        # Real-time metrics
        real_time_capable = mean_frame_time < target_frame_time
        real_time_ratio = target_frame_time / mean_frame_time
        dropped_frames_p95 = max(0, (p95_frame_time - target_frame_time) / target_frame_time)
        
        return {
            'target_fps': target_fps,
            'target_frame_time_ms': target_frame_time * 1000,
            'mean_frame_time_ms': mean_frame_time * 1000,
            'std_frame_time_ms': std_frame_time * 1000,
            'p95_frame_time_ms': p95_frame_time * 1000,
            'p99_frame_time_ms': p99_frame_time * 1000,
            'achievable_fps': 1.0 / mean_frame_time,
            'real_time_capable': real_time_capable,
            'real_time_ratio': real_time_ratio,
            'dropped_frames_p95_ratio': dropped_frames_p95
        }
    
    def evaluate_streaming_performance(
        self,
        total_duration_s: float = 10.0,
        chunk_size: int = 8
    ) -> Dict[str, Any]:
        """Evaluate streaming/online performance."""
        target_chunks = int(total_duration_s * 30 / chunk_size)  # Assuming 30 FPS
        
        dummy_chunk = torch.randn(1, chunk_size, 256, device=self.device)
        
        chunk_times = []
        memory_usage = []
        
        start_total = time.time()
        
        for i in range(target_chunks):
            if self.device == 'cuda':
                start_memory = torch.cuda.memory_allocated()
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                output = self.model(dummy_chunk)
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.time()
            
            chunk_time = end_time - start_time
            chunk_times.append(chunk_time)
            
            if self.device == 'cuda':
                end_memory = torch.cuda.memory_allocated()
                memory_usage.append(end_memory - start_memory)
        
        total_time = time.time() - start_total
        
        chunk_times = torch.tensor(chunk_times)
        
        return {
            'total_duration_s': total_duration_s,
            'total_chunks': target_chunks,
            'actual_duration_s': total_time,
            'mean_chunk_time_ms': chunk_times.mean().item() * 1000,
            'max_chunk_time_ms': chunk_times.max().item() * 1000,
            'chunk_time_variance': chunk_times.var().item(),
            'real_time_factor': total_duration_s / total_time,
            'avg_memory_per_chunk_mb': sum(memory_usage) / len(memory_usage) / (1024*1024) if memory_usage else 0
        }


class ComparisonEvaluator:
    """Evaluator for comparing multiple models."""
    
    def __init__(self, models: Dict[str, torch.nn.Module], device: str = 'cuda'):
        self.models = {name: model.eval() for name, model in models.items()}
        self.device = device
    
    def compare_models(
        self,
        test_data: Dict[str, torch.Tensor],
        metrics_to_compare: List[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple models on same test data."""
        if metrics_to_compare is None:
            metrics_to_compare = ['mse', 'psnr', 'ssim', 'generation_time_s']
        
        results = {}
        
        for model_name, model in self.models.items():
            evaluator = SequenceEvaluator(model, device=self.device)
            
            # Prepare test data
            frames = test_data['frames'][:4]  # Small batch for comparison
            context_length = frames.shape[1] // 2
            context_frames = frames[:, :context_length]
            target_frames = frames[:, context_length:]
            
            # Evaluate model
            model_results = evaluator.evaluate_batch({
                'frames': frames,
                'controls': test_data.get('controls', None)
            })
            
            # Extract requested metrics
            results[model_name] = {k: v for k, v in model_results.items() if k in metrics_to_compare}
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(results, metrics_to_compare)
        
        return {
            'individual_results': results,
            'comparison_summary': comparison_summary
        }
    
    def _create_comparison_summary(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Create summary of model comparison."""
        summary = {}
        
        for metric in metrics:
            metric_values = {}
            for model_name, model_results in results.items():
                if metric in model_results:
                    metric_values[model_name] = model_results[metric]
            
            if metric_values:
                best_model = min(metric_values.items(), key=lambda x: x[1])
                worst_model = max(metric_values.items(), key=lambda x: x[1])
                
                summary[metric] = {
                    'best_model': best_model[0],
                    'best_value': best_model[1],
                    'worst_model': worst_model[0],
                    'worst_value': worst_model[1],
                    'all_values': metric_values
                }
        
        return summary


class AblationEvaluator:
    """Evaluator for ablation studies."""
    
    def __init__(self, base_model, device: str = 'cuda'):
        self.base_model = base_model
        self.device = device
    
    def evaluate_component_ablation(
        self,
        test_data: Dict[str, torch.Tensor],
        ablation_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate ablation study by modifying model components."""
        results = {}
        
        # Evaluate base model
        base_evaluator = SequenceEvaluator(self.base_model, device=self.device)
        base_results = base_evaluator.evaluate_batch(test_data)
        results['base_model'] = base_results
        
        # Evaluate ablated versions
        for i, config in enumerate(ablation_configs):
            config_name = config.get('name', f'ablation_{i}')
            
            # This would require model modification based on config
            # Placeholder implementation
            ablated_model = self._create_ablated_model(config)
            
            if ablated_model:
                ablated_evaluator = SequenceEvaluator(ablated_model, device=self.device)
                ablated_results = ablated_evaluator.evaluate_batch(test_data)
                results[config_name] = ablated_results
        
        # Create ablation analysis
        analysis = self._analyze_ablation_results(results)
        
        return {
            'results': results,
            'analysis': analysis
        }
    
    def _create_ablated_model(self, config: Dict[str, Any]):
        """Create ablated version of model (placeholder)."""
        # This would implement actual model modifications
        # For now, return None to indicate not implemented
        return None
    
    def _analyze_ablation_results(self, results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze ablation study results."""
        if 'base_model' not in results:
            return {}
        
        base_metrics = results['base_model']
        analysis = {}
        
        for model_name, model_results in results.items():
            if model_name == 'base_model':
                continue
            
            model_analysis = {}
            for metric, value in model_results.items():
                if metric in base_metrics and isinstance(value, (int, float)):
                    base_value = base_metrics[metric]
                    if base_value != 0:
                        relative_change = (value - base_value) / base_value
                        model_analysis[metric] = {
                            'absolute_change': value - base_value,
                            'relative_change': relative_change,
                            'improved': relative_change < 0 if 'mse' in metric or 'mae' in metric else relative_change > 0
                        }
            
            analysis[model_name] = model_analysis
        
        return analysis