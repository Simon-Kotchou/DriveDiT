"""
Model optimization utilities for inference acceleration.
Includes quantization, compilation, and export functionality.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import onnx
import onnxruntime as ort
from pathlib import Path
import json
import time


class ModelOptimizer:
    """Utilities for optimizing models for inference."""
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Apply general optimizations for inference.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        model.eval()
        
        # Fuse operations where possible
        try:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
        except Exception as e:
            print(f"JIT optimization failed: {e}")
        
        return model
    
    @staticmethod
    def apply_torch_compile(
        model: nn.Module,
        mode: str = 'reduce-overhead',
        dynamic: bool = False
    ) -> nn.Module:
        """
        Apply torch.compile for acceleration.
        
        Args:
            model: Model to compile
            mode: Compilation mode
            dynamic: Whether to use dynamic compilation
            
        Returns:
            Compiled model
        """
        try:
            compiled_model = torch.compile(
                model,
                mode=mode,
                dynamic=dynamic
            )
            print(f"Model compiled with mode: {mode}")
            return compiled_model
        except Exception as e:
            print(f"Compilation failed: {e}")
            return model
    
    @staticmethod
    def benchmark_model(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cuda',
        num_warmup: int = 10,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            device: Device to run on
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics
        """
        model.eval()
        device_obj = torch.device(device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device_obj)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Benchmark
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.inference_mode():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = 1.0 / avg_time
        
        return {
            'avg_latency_ms': avg_time * 1000,
            'throughput_fps': throughput,
            'total_time_s': total_time,
            'iterations': num_iterations
        }


class QuantizedInference:
    """Quantized inference for reduced memory and faster computation."""
    
    def __init__(self, model: nn.Module, quantization_type: str = 'dynamic'):
        """
        Initialize quantized inference.
        
        Args:
            model: Model to quantize
            quantization_type: Type of quantization ('dynamic', 'static')
        """
        self.original_model = model
        self.quantization_type = quantization_type
        self.quantized_model = self._quantize_model(model, quantization_type)
    
    def _quantize_model(self, model: nn.Module, quantization_type: str) -> nn.Module:
        """Apply quantization to model."""
        model.eval()
        
        if quantization_type == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d},
                dtype=torch.qint8
            )
        elif quantization_type == 'static':
            # Static quantization (requires calibration)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # TODO: Add calibration step with representative data
            # calibrate_model(model, calibration_data)
            
            quantized_model = torch.quantization.convert(model, inplace=False)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return quantized_model
    
    def __call__(self, *args, **kwargs):
        """Forward pass through quantized model."""
        return self.quantized_model(*args, **kwargs)
    
    def compare_performance(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Compare performance between original and quantized models."""
        # Benchmark original model
        original_metrics = ModelOptimizer.benchmark_model(
            self.original_model,
            input_tensor.shape,
            device='cpu'  # Quantization typically on CPU
        )
        
        # Benchmark quantized model
        quantized_metrics = ModelOptimizer.benchmark_model(
            self.quantized_model,
            input_tensor.shape,
            device='cpu'
        )
        
        # Calculate improvements
        latency_improvement = (
            original_metrics['avg_latency_ms'] / quantized_metrics['avg_latency_ms']
        )
        
        return {
            'original': original_metrics,
            'quantized': quantized_metrics,
            'latency_speedup': latency_improvement,
            'quantization_type': self.quantization_type
        }


class CompiledInference:
    """TorchScript compiled inference for deployment."""
    
    def __init__(self, model: nn.Module, example_input: torch.Tensor):
        """
        Initialize compiled inference.
        
        Args:
            model: Model to compile
            example_input: Example input for tracing
        """
        self.original_model = model
        self.example_input = example_input
        self.scripted_model = self._compile_model(model, example_input)
    
    def _compile_model(self, model: nn.Module, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        """Compile model to TorchScript."""
        model.eval()
        
        try:
            # Try tracing first
            with torch.inference_mode():
                scripted_model = torch.jit.trace(model, example_input)
            print("Model traced successfully")
        except Exception as e:
            print(f"Tracing failed: {e}, trying scripting...")
            try:
                # Fallback to scripting
                scripted_model = torch.jit.script(model)
                print("Model scripted successfully")
            except Exception as e2:
                print(f"Scripting also failed: {e2}")
                raise e2
        
        # Optimize for inference
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        
        return scripted_model
    
    def save(self, path: str):
        """Save compiled model."""
        self.scripted_model.save(path)
        print(f"Compiled model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'CompiledInference':
        """Load compiled model."""
        scripted_model = torch.jit.load(path, map_location=device)
        
        # Create instance
        instance = cls.__new__(cls)
        instance.scripted_model = scripted_model
        instance.original_model = None
        instance.example_input = None
        
        return instance
    
    def __call__(self, *args, **kwargs):
        """Forward pass through compiled model."""
        return self.scripted_model(*args, **kwargs)


class TorchScriptExporter:
    """Export models to TorchScript format."""
    
    @staticmethod
    def export_model(
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        method: str = 'trace'
    ) -> str:
        """
        Export model to TorchScript.
        
        Args:
            model: Model to export
            example_input: Example input for tracing
            output_path: Output file path
            method: Export method ('trace' or 'script')
            
        Returns:
            Path to exported model
        """
        model.eval()
        
        if method == 'trace':
            with torch.inference_mode():
                traced_model = torch.jit.trace(model, example_input)
            exported_model = traced_model
        elif method == 'script':
            exported_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown export method: {method}")
        
        # Optimize for inference
        exported_model = torch.jit.optimize_for_inference(exported_model)
        
        # Save model
        exported_model.save(output_path)
        
        # Save metadata
        metadata = {
            'export_method': method,
            'input_shape': list(example_input.shape),
            'model_type': type(model).__name__
        }
        
        metadata_path = output_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model exported to {output_path}")
        return output_path


class ONNXExporter:
    """Export models to ONNX format."""
    
    @staticmethod
    def export_model(
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str,
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> str:
        """
        Export model to ONNX format.
        
        Args:
            model: Model to export
            example_input: Example input for export
            output_path: Output file path
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            
        Returns:
            Path to exported model
        """
        model.eval()
        
        # Default dynamic axes for sequence models
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify exported model
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported and verified: {output_path}")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")
        
        return output_path
    
    @staticmethod
    def create_onnx_session(
        onnx_path: str,
        providers: Optional[List[str]] = None
    ) -> ort.InferenceSession:
        """
        Create ONNX Runtime inference session.
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers
            
        Returns:
            ONNX Runtime session
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        return session
    
    @staticmethod
    def benchmark_onnx_model(
        session: ort.InferenceSession,
        input_data: np.ndarray,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark ONNX model performance.
        
        Args:
            session: ONNX Runtime session
            input_data: Input data for benchmarking
            num_iterations: Number of iterations
            
        Returns:
            Performance metrics
        """
        import numpy as np
        
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_name: input_data})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = session.run(None, {input_name: input_data})
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        
        return {
            'avg_latency_ms': avg_time * 1000,
            'throughput_fps': 1.0 / avg_time,
            'total_time_s': total_time
        }


class InferenceOptimizer:
    """High-level optimizer for inference pipelines."""
    
    def __init__(self, model: nn.Module, example_input: torch.Tensor):
        """
        Initialize inference optimizer.
        
        Args:
            model: Model to optimize
            example_input: Example input tensor
        """
        self.model = model
        self.example_input = example_input
        self.optimizations = {}
    
    def apply_all_optimizations(self, device: str = 'cuda') -> Dict[str, Any]:
        """
        Apply all available optimizations and compare performance.
        
        Args:
            device: Device to run optimizations on
            
        Returns:
            Comparison of all optimization methods
        """
        results = {}
        
        # Original model
        original_metrics = ModelOptimizer.benchmark_model(
            self.model, self.example_input.shape, device
        )
        results['original'] = original_metrics
        
        # Torch compile
        try:
            compiled_model = ModelOptimizer.apply_torch_compile(self.model)
            compiled_metrics = ModelOptimizer.benchmark_model(
                compiled_model, self.example_input.shape, device
            )
            results['compiled'] = compiled_metrics
            self.optimizations['compiled'] = compiled_model
        except Exception as e:
            print(f"Torch compile failed: {e}")
        
        # TorchScript
        try:
            scripted = CompiledInference(self.model, self.example_input)
            scripted_metrics = ModelOptimizer.benchmark_model(
                scripted.scripted_model, self.example_input.shape, device
            )
            results['torchscript'] = scripted_metrics
            self.optimizations['torchscript'] = scripted
        except Exception as e:
            print(f"TorchScript compilation failed: {e}")
        
        # Quantization (CPU only)
        if device == 'cpu':
            try:
                quantized = QuantizedInference(self.model)
                quantized_metrics = ModelOptimizer.benchmark_model(
                    quantized.quantized_model, self.example_input.shape, 'cpu'
                )
                results['quantized'] = quantized_metrics
                self.optimizations['quantized'] = quantized
            except Exception as e:
                print(f"Quantization failed: {e}")
        
        # Calculate speedups
        for opt_name, metrics in results.items():
            if opt_name != 'original':
                speedup = original_metrics['avg_latency_ms'] / metrics['avg_latency_ms']
                metrics['speedup'] = speedup
        
        return results
    
    def get_best_optimization(self, results: Dict[str, Any]) -> Tuple[str, nn.Module]:
        """
        Get the best optimization based on performance.
        
        Args:
            results: Results from apply_all_optimizations
            
        Returns:
            Name and model of best optimization
        """
        best_name = 'original'
        best_latency = results['original']['avg_latency_ms']
        
        for opt_name, metrics in results.items():
            if opt_name != 'original' and metrics['avg_latency_ms'] < best_latency:
                best_latency = metrics['avg_latency_ms']
                best_name = opt_name
        
        if best_name == 'original':
            return best_name, self.model
        else:
            return best_name, self.optimizations[best_name]
    
    def export_optimized_model(
        self,
        optimization_name: str,
        output_dir: str,
        formats: List[str] = ['torchscript', 'onnx']
    ) -> Dict[str, str]:
        """
        Export optimized model in various formats.
        
        Args:
            optimization_name: Name of optimization to export
            output_dir: Output directory
            formats: Export formats
            
        Returns:
            Dictionary mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_paths = {}
        
        if optimization_name in self.optimizations:
            model = self.optimizations[optimization_name]
        else:
            model = self.model
        
        for fmt in formats:
            if fmt == 'torchscript':
                output_path = output_dir / f"model_{optimization_name}.pt"
                if hasattr(model, 'scripted_model'):
                    model.save(str(output_path))
                else:
                    TorchScriptExporter.export_model(
                        model, self.example_input, str(output_path)
                    )
                exported_paths['torchscript'] = str(output_path)
            
            elif fmt == 'onnx':
                output_path = output_dir / f"model_{optimization_name}.onnx"
                if hasattr(model, 'original_model'):
                    export_model = model.original_model
                else:
                    export_model = model
                
                ONNXExporter.export_model(
                    export_model, self.example_input, str(output_path)
                )
                exported_paths['onnx'] = str(output_path)
        
        return exported_paths