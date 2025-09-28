"""
Data collation utilities for batching video sequences.
Handles variable-length sequences and memory-efficient batching.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import random


class VideoCollator:
    """Collator for video sequences with padding and batching."""
    
    def __init__(
        self,
        pad_to_max: bool = True,
        max_sequence_length: Optional[int] = None,
        padding_value: float = 0.0,
        drop_incomplete: bool = False
    ):
        """
        Initialize video collator.
        
        Args:
            pad_to_max: Whether to pad sequences to max length in batch
            max_sequence_length: Maximum allowed sequence length
            padding_value: Value to use for padding
            drop_incomplete: Whether to drop incomplete sequences
        """
        self.pad_to_max = pad_to_max
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        self.drop_incomplete = drop_incomplete
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        if not batch:
            return {}
        
        # Filter incomplete sequences if requested
        if self.drop_incomplete:
            batch = self._filter_incomplete_sequences(batch)
        
        collated = {}
        
        # Handle video frames
        if 'frames' in batch[0]:
            frames_list = [sample['frames'] for sample in batch]
            collated['frames'] = self._collate_video_sequences(frames_list)
        
        # Handle controls
        if 'controls' in batch[0]:
            controls_list = [sample['controls'] for sample in batch]
            collated['controls'] = self._collate_sequences(controls_list)
        
        # Handle depth
        if 'depth' in batch[0]:
            depth_list = [sample['depth'] for sample in batch]
            collated['depth'] = self._collate_video_sequences(depth_list)
        
        # Handle metadata
        for key in ['sequence_id', 'start_idx']:
            if key in batch[0]:
                collated[key] = [sample[key] for sample in batch]
        
        # Handle other tensor fields
        for key, value in batch[0].items():
            if key not in collated and isinstance(value, torch.Tensor):
                if value.dim() == 0:  # Scalar tensors
                    collated[key] = torch.stack([sample[key] for sample in batch])
                else:
                    tensor_list = [sample[key] for sample in batch]
                    collated[key] = self._collate_sequences(tensor_list)
        
        return collated
    
    def _filter_incomplete_sequences(self, batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Filter out incomplete sequences."""
        if self.max_sequence_length is None:
            return batch
        
        filtered_batch = []
        for sample in batch:
            frames = sample.get('frames')
            if frames is not None:
                seq_len = frames.shape[0] if frames.dim() == 4 else frames.shape[1]
                if seq_len >= self.max_sequence_length:
                    filtered_batch.append(sample)
        
        return filtered_batch if filtered_batch else batch  # Fallback to original batch
    
    def _collate_video_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Collate video sequences with proper padding."""
        if not sequences:
            return torch.empty(0)
        
        # Get sequence properties
        max_length = max(seq.shape[0] if seq.dim() == 4 else seq.shape[1] for seq in sequences)
        
        # Apply max length constraint
        if self.max_sequence_length is not None:
            max_length = min(max_length, self.max_sequence_length)
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if seq.dim() == 4:  # [T, C, H, W]
                seq_len = seq.shape[0]
                if seq_len > max_length:
                    # Truncate
                    seq = seq[:max_length]
                elif seq_len < max_length and self.pad_to_max:
                    # Pad
                    pad_length = max_length - seq_len
                    padding = torch.full((pad_length, *seq.shape[1:]), self.padding_value, dtype=seq.dtype, device=seq.device)
                    seq = torch.cat([seq, padding], dim=0)
                
                padded_sequences.append(seq.unsqueeze(0))  # Add batch dimension
            
            elif seq.dim() == 5:  # [B, T, C, H, W] - already batched
                padded_sequences.append(seq)
            
            else:
                raise ValueError(f"Unsupported sequence dimensions: {seq.dim()}")
        
        return torch.cat(padded_sequences, dim=0)
    
    def _collate_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Collate general tensor sequences."""
        if not sequences:
            return torch.empty(0)
        
        # Handle different dimensions
        if sequences[0].dim() == 1:  # [T] -> stack to [B, T]
            max_length = max(seq.shape[0] for seq in sequences)
            if self.max_sequence_length is not None:
                max_length = min(max_length, self.max_sequence_length)
            
            padded_sequences = []
            for seq in sequences:
                if len(seq) > max_length:
                    seq = seq[:max_length]
                elif len(seq) < max_length and self.pad_to_max:
                    pad_length = max_length - len(seq)
                    padding = torch.full((pad_length,), self.padding_value, dtype=seq.dtype, device=seq.device)
                    seq = torch.cat([seq, padding], dim=0)
                padded_sequences.append(seq)
            
            return torch.stack(padded_sequences, dim=0)
        
        elif sequences[0].dim() == 2:  # [T, D] -> stack to [B, T, D]
            max_length = max(seq.shape[0] for seq in sequences)
            if self.max_sequence_length is not None:
                max_length = min(max_length, self.max_sequence_length)
            
            padded_sequences = []
            for seq in sequences:
                if seq.shape[0] > max_length:
                    seq = seq[:max_length]
                elif seq.shape[0] < max_length and self.pad_to_max:
                    pad_length = max_length - seq.shape[0]
                    padding = torch.full((pad_length, seq.shape[1]), self.padding_value, dtype=seq.dtype, device=seq.device)
                    seq = torch.cat([seq, padding], dim=0)
                padded_sequences.append(seq)
            
            return torch.stack(padded_sequences, dim=0)
        
        else:
            # For higher dimensions, just stack
            return torch.stack(sequences, dim=0)


class LatentCollator:
    """Specialized collator for latent representations."""
    
    def __init__(
        self,
        tokenize: bool = True,
        max_tokens: Optional[int] = None,
        pad_token_id: int = 0
    ):
        """
        Initialize latent collator.
        
        Args:
            tokenize: Whether to tokenize latents into sequences
            max_tokens: Maximum number of tokens per sequence
            pad_token_id: Padding token ID
        """
        self.tokenize = tokenize
        self.max_tokens = max_tokens
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate latent batch."""
        if not batch:
            return {}
        
        collated = {}
        
        # Handle latent representations
        if 'latents' in batch[0]:
            latents_list = [sample['latents'] for sample in batch]
            
            if self.tokenize:
                # Tokenize and pad
                collated['latent_tokens'] = self._collate_tokenized_latents(latents_list)
            else:
                # Regular collation
                collated['latents'] = self._collate_latent_tensors(latents_list)
        
        # Handle other fields
        for key, value in batch[0].items():
            if key not in collated:
                if isinstance(value, torch.Tensor):
                    tensor_list = [sample[key] for sample in batch]
                    collated[key] = torch.stack(tensor_list, dim=0)
                else:
                    collated[key] = [sample[key] for sample in batch]
        
        return collated
    
    def _collate_tokenized_latents(self, latents_list: List[torch.Tensor]) -> torch.Tensor:
        """Collate tokenized latent representations."""
        tokenized_list = []
        
        for latents in latents_list:
            # Tokenize: [T, C, H, W] -> [T, C*H*W]
            if latents.dim() == 4:  # [T, C, H, W]
                T, C, H, W = latents.shape
                tokens = latents.view(T, C * H * W)
            elif latents.dim() == 3:  # [C, H, W]
                C, H, W = latents.shape
                tokens = latents.view(1, C * H * W)  # Add time dimension
            else:
                tokens = latents
            
            tokenized_list.append(tokens)
        
        # Pad to max length
        max_length = max(tokens.shape[0] for tokens in tokenized_list)
        if self.max_tokens is not None:
            max_length = min(max_length, self.max_tokens)
        
        padded_tokens = []
        for tokens in tokenized_list:
            if tokens.shape[0] > max_length:
                tokens = tokens[:max_length]
            elif tokens.shape[0] < max_length:
                pad_length = max_length - tokens.shape[0]
                padding = torch.full((pad_length, tokens.shape[1]), self.pad_token_id, dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=0)
            
            padded_tokens.append(tokens)
        
        return torch.stack(padded_tokens, dim=0)  # [B, T, D]
    
    def _collate_latent_tensors(self, latents_list: List[torch.Tensor]) -> torch.Tensor:
        """Collate latent tensors without tokenization."""
        # Find common shape
        shapes = [latents.shape for latents in latents_list]
        
        if all(shape == shapes[0] for shape in shapes):
            # All same shape - simple stack
            return torch.stack(latents_list, dim=0)
        else:
            # Different shapes - need padding
            max_dims = []
            for dim_idx in range(len(shapes[0])):
                max_dim = max(shape[dim_idx] for shape in shapes)
                max_dims.append(max_dim)
            
            padded_latents = []
            for latents in latents_list:
                # Pad to max dimensions
                padding = []
                for dim_idx in range(len(latents.shape) - 1, -1, -1):
                    pad_amount = max_dims[dim_idx] - latents.shape[dim_idx]
                    padding.extend([0, pad_amount])
                
                if any(p > 0 for p in padding):
                    latents = torch.nn.functional.pad(latents, padding)
                
                padded_latents.append(latents)
            
            return torch.stack(padded_latents, dim=0)


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that automatically chooses appropriate collation strategy."""
    if not batch:
        return {}
    
    # Determine data types in batch
    has_video = 'frames' in batch[0]
    has_latents = 'latents' in batch[0]
    has_controls = 'controls' in batch[0]
    
    # Choose appropriate collator
    if has_latents and not has_video:
        collator = LatentCollator()
    else:
        collator = VideoCollator()
    
    return collator(batch)


class MemoryEfficientCollator:
    """Memory-efficient collator for large video datasets."""
    
    def __init__(
        self,
        max_batch_frames: int = 128,
        adaptive_batching: bool = True,
        frame_memory_limit_mb: float = 1000.0
    ):
        """
        Initialize memory-efficient collator.
        
        Args:
            max_batch_frames: Maximum total frames per batch
            adaptive_batching: Whether to adapt batch size based on sequence length
            frame_memory_limit_mb: Memory limit for frames in MB
        """
        self.max_batch_frames = max_batch_frames
        self.adaptive_batching = adaptive_batching
        self.frame_memory_limit_mb = frame_memory_limit_mb
        
        self.base_collator = VideoCollator()
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate with memory efficiency."""
        if not batch:
            return {}
        
        # Estimate memory usage
        if self.adaptive_batching:
            batch = self._filter_by_memory_limit(batch)
        
        return self.base_collator(batch)
    
    def _filter_by_memory_limit(self, batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Filter batch to stay within memory limits."""
        if not batch:
            return batch
        
        total_frames = 0
        estimated_memory = 0.0
        filtered_batch = []
        
        for sample in batch:
            frames = sample.get('frames')
            if frames is not None:
                seq_frames = frames.shape[0] if frames.dim() == 4 else frames.shape[1]
                frame_memory = self._estimate_frame_memory(frames)
                
                if (total_frames + seq_frames <= self.max_batch_frames and 
                    estimated_memory + frame_memory <= self.frame_memory_limit_mb):
                    
                    filtered_batch.append(sample)
                    total_frames += seq_frames
                    estimated_memory += frame_memory
                else:
                    break
        
        return filtered_batch if filtered_batch else batch[:1]  # At least one sample
    
    def _estimate_frame_memory(self, frames: torch.Tensor) -> float:
        """Estimate memory usage of frames in MB."""
        if frames.dtype == torch.float32:
            bytes_per_element = 4
        elif frames.dtype == torch.float16:
            bytes_per_element = 2
        elif frames.dtype == torch.uint8:
            bytes_per_element = 1
        else:
            bytes_per_element = 4  # Default assumption
        
        total_elements = frames.numel()
        total_bytes = total_elements * bytes_per_element
        total_mb = total_bytes / (1024 * 1024)
        
        return total_mb


class ChunkedBatchCollator:
    """Collator that processes large sequences in chunks."""
    
    def __init__(
        self,
        chunk_size: int = 32,
        overlap: int = 4,
        shuffle_chunks: bool = True
    ):
        """
        Initialize chunked batch collator.
        
        Args:
            chunk_size: Size of each chunk
            overlap: Overlap between consecutive chunks
            shuffle_chunks: Whether to shuffle chunks within batch
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.shuffle_chunks = shuffle_chunks
        
        self.base_collator = VideoCollator()
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate with chunking."""
        if not batch:
            return {}
        
        # Create chunks from sequences
        chunked_samples = []
        
        for sample in batch:
            chunks = self._create_chunks(sample)
            chunked_samples.extend(chunks)
        
        # Shuffle chunks if requested
        if self.shuffle_chunks:
            random.shuffle(chunked_samples)
        
        # Collate chunks
        return self.base_collator(chunked_samples)
    
    def _create_chunks(self, sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Create chunks from a single sample."""
        frames = sample.get('frames')
        if frames is None:
            return [sample]
        
        seq_len = frames.shape[0] if frames.dim() == 4 else frames.shape[1]
        
        if seq_len <= self.chunk_size:
            return [sample]
        
        chunks = []
        step = self.chunk_size - self.overlap
        
        for start_idx in range(0, seq_len - self.chunk_size + 1, step):
            end_idx = start_idx + self.chunk_size
            
            chunk_sample = {}
            
            # Chunk frames
            if frames.dim() == 4:  # [T, C, H, W]
                chunk_sample['frames'] = frames[start_idx:end_idx]
            else:  # [B, T, C, H, W]
                chunk_sample['frames'] = frames[:, start_idx:end_idx]
            
            # Chunk other temporal data
            for key, value in sample.items():
                if key == 'frames':
                    continue
                elif isinstance(value, torch.Tensor) and value.dim() >= 1:
                    if value.shape[0] == seq_len:  # Temporal dimension
                        chunk_sample[key] = value[start_idx:end_idx]
                    elif value.dim() >= 2 and value.shape[1] == seq_len:  # Batch, temporal
                        chunk_sample[key] = value[:, start_idx:end_idx]
                    else:
                        chunk_sample[key] = value
                else:
                    chunk_sample[key] = value
            
            chunks.append(chunk_sample)
        
        return chunks