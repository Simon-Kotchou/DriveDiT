"""
High-level inference pipelines for different use cases.
Provides easy-to-use interfaces for model inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import time
import numpy as np
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass

from .rollout import StreamingRollout, InferenceConfig
from ..models.vae3d import VAE3D
from ..models.dit_student import DiTStudent
from ..data.preprocessing import VideoPreprocessor


@dataclass
class PipelineConfig:
    """Configuration for inference pipelines."""
    model_path: str
    vae_path: str
    device: str = 'cuda'
    batch_size: int = 1
    max_sequence_length: int = 64
    context_window: int = 8
    temperature: float = 0.8
    use_mixed_precision: bool = True
    use_torch_compile: bool = False
    cache_models: bool = True


class InferencePipeline:
    """Base inference pipeline with model loading and preprocessing."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize inference pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Load models
        self.student_model = self._load_student_model(config.model_path)
        self.vae_model = self._load_vae_model(config.vae_path)
        
        # Initialize preprocessor
        self.preprocessor = VideoPreprocessor(
            target_size=(256, 256),
            normalize=True,
            dtype=torch.float16 if config.use_mixed_precision else torch.float32
        )
        
        # Create inference config
        self.inference_config = InferenceConfig(
            max_sequence_length=config.max_sequence_length,
            context_window=config.context_window,
            temperature=config.temperature,
            mixed_precision=config.use_mixed_precision,
            use_kv_cache=True
        )
        
        # Initialize rollout
        self.rollout = StreamingRollout(
            self.student_model,
            self.vae_model,
            self.inference_config,
            device=str(self.device)
        )
        
        # Optimize models if requested
        if config.use_torch_compile:
            self._compile_models()
    
    def _load_student_model(self, model_path: str) -> DiTStudent:
        """Load and initialize student model."""
        if not Path(model_path).exists():
            # Create default model for testing
            model = DiTStudent(
                latent_dim=64,
                d_model=512,
                n_layers=12,
                n_heads=8
            )
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Infer model config from state dict
            model = self._create_model_from_state_dict(state_dict)
            model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        if self.config.use_mixed_precision:
            model = model.half()
        
        return model
    
    def _load_vae_model(self, vae_path: str) -> VAE3D:
        """Load and initialize VAE model."""
        if not Path(vae_path).exists():
            # Create default VAE for testing
            model = VAE3D(
                in_channels=3,
                latent_dim=64,
                hidden_dims=[64, 128, 256, 512]
            )
        else:
            checkpoint = torch.load(vae_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            model = VAE3D(
                in_channels=3,
                latent_dim=64,
                hidden_dims=[64, 128, 256, 512]
            )
            model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        if self.config.use_mixed_precision:
            model = model.half()
        
        return model
    
    def _create_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> DiTStudent:
        """Create model from state dict by inferring dimensions."""
        # Extract dimensions from state dict keys
        d_model = 512  # Default
        n_layers = 12  # Default
        n_heads = 8    # Default
        
        # Try to infer from embedding layer
        if 'embedding.weight' in state_dict:
            d_model = state_dict['embedding.weight'].shape[1]
        
        # Try to infer from attention layers
        attn_keys = [k for k in state_dict.keys() if 'attn.qkv.weight' in k]
        if attn_keys:
            qkv_dim = state_dict[attn_keys[0]].shape[0]
            d_model = qkv_dim // 3
        
        # Count layers
        layer_keys = [k for k in state_dict.keys() if 'layers.' in k]
        if layer_keys:
            layer_nums = [int(k.split('.')[1]) for k in layer_keys if k.split('.')[1].isdigit()]
            if layer_nums:
                n_layers = max(layer_nums) + 1
        
        return DiTStudent(
            latent_dim=64,  # Will be adjusted by loading
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads
        )
    
    def _compile_models(self):
        """Compile models for faster inference."""
        try:
            self.student_model = torch.compile(self.student_model, mode='reduce-overhead')
            self.vae_model = torch.compile(self.vae_model, mode='reduce-overhead')
            print("Models compiled successfully")
        except Exception as e:
            print(f"Failed to compile models: {e}")
    
    def preprocess_input(
        self,
        frames: Union[torch.Tensor, np.ndarray],
        controls: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocess input frames and controls.
        
        Args:
            frames: Input frames, various formats supported
            controls: Optional control signals
            
        Returns:
            Preprocessed frames and controls
        """
        # Convert to tensor if needed
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        
        # Ensure correct shape [B, T, C, H, W]
        if frames.dim() == 3:  # [C, H, W]
            frames = frames.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
        elif frames.dim() == 4:  # [T, C, H, W] or [B, C, H, W]
            if frames.shape[1] == 3:  # Assume [T, C, H, W]
                frames = frames.unsqueeze(0)  # [1, T, C, H, W]
            else:  # Assume [B, C, H, W]
                frames = frames.unsqueeze(1)  # [B, 1, C, H, W]
        
        # Preprocess frames
        frames = self.preprocessor.preprocess_batch(frames)
        frames = frames.to(self.device)
        
        # Process controls if provided
        if controls is not None:
            if isinstance(controls, np.ndarray):
                controls = torch.from_numpy(controls)
            controls = controls.to(self.device)
            
            # Ensure correct shape [B, T, control_dim]
            if controls.dim() == 1:  # [control_dim]
                controls = controls.unsqueeze(0).unsqueeze(0)  # [1, 1, control_dim]
            elif controls.dim() == 2:  # [T, control_dim] or [B, control_dim]
                if controls.shape[0] == frames.shape[0]:  # [B, control_dim]
                    controls = controls.unsqueeze(1)  # [B, 1, control_dim]
                else:  # [T, control_dim]
                    controls = controls.unsqueeze(0)  # [1, T, control_dim]
        
        return frames, controls
    
    def postprocess_output(self, result: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Postprocess inference output.
        
        Args:
            result: Raw inference result
            
        Returns:
            Processed result with numpy arrays and metadata
        """
        processed = {}
        
        # Convert tensors to numpy
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                # Denormalize frames
                if 'frames' in key:
                    value = self.preprocessor.denormalize(value)
                    value = torch.clamp(value, 0, 1)
                
                processed[key] = value.detach().cpu().numpy()
            else:
                processed[key] = value
        
        return processed
    
    def generate(
        self,
        context_frames: Union[torch.Tensor, np.ndarray],
        controls: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_frames: int = 16,
        return_dict: bool = True
    ) -> Union[Dict[str, Any], torch.Tensor]:
        """
        Generate video sequence.
        
        Args:
            context_frames: Context frames for generation
            controls: Optional control signals
            num_frames: Number of frames to generate
            return_dict: Whether to return full result dict or just frames
            
        Returns:
            Generated sequence(s)
        """
        # Preprocess inputs
        frames, controls = self.preprocess_input(context_frames, controls)
        
        # Ensure we have enough control signals
        if controls is not None:
            total_length = frames.shape[1] + num_frames
            if controls.shape[1] < total_length:
                # Repeat last control signal
                last_control = controls[:, -1:].repeat(1, total_length - controls.shape[1], 1)
                controls = torch.cat([controls, last_control], dim=1)
        else:
            # Create dummy controls
            controls = torch.zeros(frames.shape[0], frames.shape[1] + num_frames, 4, device=self.device)
        
        # Generate sequence
        with torch.inference_mode():
            result = self.rollout.generate_sequence(
                context_frames=frames,
                control_sequence=controls,
                max_new_frames=num_frames,
                return_intermediates=False
            )
        
        if return_dict:
            return self.postprocess_output(result)
        else:
            return result['generated_frames']


class BatchInferencePipeline(InferencePipeline):
    """Pipeline optimized for batch inference."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.batch_size = config.batch_size
    
    def generate_batch(
        self,
        batch_context_frames: List[Union[torch.Tensor, np.ndarray]],
        batch_controls: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
        num_frames: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Generate sequences for a batch of inputs.
        
        Args:
            batch_context_frames: List of context frame tensors
            batch_controls: Optional list of control tensors
            num_frames: Number of frames to generate
            
        Returns:
            List of generation results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(batch_context_frames), self.batch_size):
            batch_end = min(i + self.batch_size, len(batch_context_frames))
            
            # Prepare batch
            batch_frames = []
            batch_ctrls = []
            
            for j in range(i, batch_end):
                frames, controls = self.preprocess_input(
                    batch_context_frames[j],
                    batch_controls[j] if batch_controls else None
                )
                batch_frames.append(frames)
                batch_ctrls.append(controls)
            
            # Stack into batch tensors
            stacked_frames = torch.cat(batch_frames, dim=0)
            stacked_controls = torch.cat(batch_ctrls, dim=0) if batch_ctrls[0] is not None else None
            
            # Generate for batch
            batch_result = self.generate(
                stacked_frames,
                stacked_controls,
                num_frames=num_frames,
                return_dict=True
            )
            
            # Split batch results
            for k in range(batch_end - i):
                individual_result = {}
                for key, value in batch_result.items():
                    if isinstance(value, np.ndarray) and value.ndim > 0:
                        individual_result[key] = value[k:k+1]
                    else:
                        individual_result[key] = value
                results.append(individual_result)
        
        return results


class RealtimeInferencePipeline(InferencePipeline):
    """Pipeline optimized for real-time streaming inference."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.frame_buffer = []
        self.max_buffer_size = config.context_window
        self.is_streaming = False
    
    def start_streaming(self):
        """Start streaming mode."""
        self.is_streaming = True
        self.frame_buffer.clear()
        self.rollout.reset_state()
    
    def stop_streaming(self):
        """Stop streaming mode."""
        self.is_streaming = False
        self.frame_buffer.clear()
    
    def process_frame(
        self,
        frame: Union[torch.Tensor, np.ndarray],
        control: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Optional[np.ndarray]:
        """
        Process single frame in streaming mode.
        
        Args:
            frame: Input frame
            control: Optional control signal
            
        Returns:
            Generated next frame or None if not ready
        """
        if not self.is_streaming:
            raise RuntimeError("Pipeline not in streaming mode. Call start_streaming() first.")
        
        # Preprocess frame
        processed_frame, processed_control = self.preprocess_input(frame, control)
        
        # Add to buffer
        self.frame_buffer.append(processed_frame.squeeze(0))  # Remove batch dim
        
        # Maintain buffer size
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # Generate next frame when buffer is full
        if len(self.frame_buffer) == self.max_buffer_size:
            # Stack buffer frames
            context_frames = torch.stack(self.frame_buffer, dim=1)  # [1, T, C, H, W]
            
            # Prepare control sequence
            if processed_control is not None:
                control_seq = processed_control.repeat(1, 2, 1)  # Extend for prediction
            else:
                control_seq = torch.zeros(1, 2, 4, device=self.device)
            
            # Generate next frame
            with torch.inference_mode():
                result = self.rollout.generate_sequence(
                    context_frames=context_frames,
                    control_sequence=control_seq,
                    max_new_frames=1,
                    return_intermediates=False
                )
            
            if result['generated_frames'].numel() > 0:
                next_frame = result['generated_frames'][0, 0]  # [C, H, W]
                
                # Postprocess
                next_frame = self.preprocessor.denormalize(next_frame.unsqueeze(0).unsqueeze(0))
                next_frame = torch.clamp(next_frame, 0, 1)
                
                return next_frame.squeeze().detach().cpu().numpy()
        
        return None
    
    async def async_process_frame(
        self,
        frame: Union[torch.Tensor, np.ndarray],
        control: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Optional[np.ndarray]:
        """Async version of process_frame."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_frame, frame, control)


class MultiModelPipeline:
    """Pipeline for ensemble inference with multiple models."""
    
    def __init__(self, model_configs: List[PipelineConfig]):
        """
        Initialize multi-model pipeline.
        
        Args:
            model_configs: List of configurations for different models
        """
        self.pipelines = []
        for config in model_configs:
            self.pipelines.append(InferencePipeline(config))
        
        self.device = torch.device(model_configs[0].device)
    
    def generate_ensemble(
        self,
        context_frames: Union[torch.Tensor, np.ndarray],
        controls: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_frames: int = 16,
        ensemble_method: str = 'average'
    ) -> Dict[str, Any]:
        """
        Generate with ensemble of models.
        
        Args:
            context_frames: Context frames
            controls: Optional control signals
            num_frames: Number of frames to generate
            ensemble_method: Method for combining results ('average', 'vote', 'best')
            
        Returns:
            Ensemble generation result
        """
        # Generate with each model
        results = []
        for pipeline in self.pipelines:
            result = pipeline.generate(
                context_frames, controls, num_frames, return_dict=True
            )
            results.append(result)
        
        # Combine results
        if ensemble_method == 'average':
            # Average generated frames
            generated_frames = []
            for result in results:
                generated_frames.append(torch.from_numpy(result['generated_frames']))
            
            avg_frames = torch.stack(generated_frames).mean(dim=0)
            
            # Use first result as template and replace frames
            ensemble_result = results[0].copy()
            ensemble_result['generated_frames'] = avg_frames.numpy()
            ensemble_result['ensemble_method'] = ensemble_method
            ensemble_result['num_models'] = len(self.pipelines)
            
            return ensemble_result
        
        elif ensemble_method == 'best':
            # Select best result based on some criteria (e.g., lowest loss)
            best_idx = 0
            best_score = float('inf')
            
            for i, result in enumerate(results):
                # Use generation time as simple scoring metric
                score = result.get('performance', {}).get('avg_fps', 0)
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            results[best_idx]['ensemble_method'] = ensemble_method
            results[best_idx]['selected_model'] = best_idx
            return results[best_idx]
        
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")


def create_pipeline(
    model_path: str,
    vae_path: str,
    pipeline_type: str = 'standard',
    device: str = 'cuda',
    **kwargs
) -> InferencePipeline:
    """
    Factory function to create inference pipelines.
    
    Args:
        model_path: Path to model checkpoint
        vae_path: Path to VAE checkpoint
        pipeline_type: Type of pipeline ('standard', 'batch', 'realtime')
        device: Device to run on
        **kwargs: Additional configuration
        
    Returns:
        Configured inference pipeline
    """
    config = PipelineConfig(
        model_path=model_path,
        vae_path=vae_path,
        device=device,
        **kwargs
    )
    
    if pipeline_type == 'standard':
        return InferencePipeline(config)
    elif pipeline_type == 'batch':
        return BatchInferencePipeline(config)
    elif pipeline_type == 'realtime':
        return RealtimeInferencePipeline(config)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")