"""
Tensor utility functions.
Pure PyTorch implementations for common tensor operations.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


def safe_cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Safely concatenate tensors, handling empty lists.
    
    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate
    
    Returns:
        Concatenated tensor or empty tensor if list is empty
    """
    if not tensors:
        return torch.empty(0)
    
    # Filter out None tensors
    valid_tensors = [t for t in tensors if t is not None]
    
    if not valid_tensors:
        return torch.empty(0)
    
    return torch.cat(valid_tensors, dim=dim)


def safe_stack(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Safely stack tensors, handling empty lists.
    
    Args:
        tensors: List of tensors to stack
        dim: Dimension along which to stack
    
    Returns:
        Stacked tensor or empty tensor if list is empty
    """
    if not tensors:
        return torch.empty(0)
    
    # Filter out None tensors
    valid_tensors = [t for t in tensors if t is not None]
    
    if not valid_tensors:
        return torch.empty(0)
    
    return torch.stack(valid_tensors, dim=dim)


def pad_sequence(
    sequences: List[torch.Tensor],
    batch_first: bool = True,
    padding_value: float = 0.0,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Pad sequences to same length.
    
    Args:
        sequences: List of tensors with varying lengths
        batch_first: If True, output shape is [B, T, ...], else [T, B, ...]
        padding_value: Value to use for padding
        max_length: Maximum length to pad to (if None, use longest sequence)
    
    Returns:
        Padded tensor
    """
    if not sequences:
        return torch.empty(0)
    
    # Get maximum length
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    # Pad each sequence
    padded_sequences = []
    for seq in sequences:
        seq_len = seq.size(0)
        if seq_len < max_length:
            # Calculate padding
            pad_size = max_length - seq_len
            padding = [0, 0] * (seq.dim() - 1) + [0, pad_size]
            padded_seq = F.pad(seq, padding, value=padding_value)
        else:
            # Truncate if longer than max_length
            padded_seq = seq[:max_length]
        
        padded_sequences.append(padded_seq)
    
    # Stack sequences
    result = torch.stack(padded_sequences, dim=0)
    
    if not batch_first:
        result = result.transpose(0, 1)
    
    return result


def chunk_tensor(
    tensor: torch.Tensor,
    chunk_size: int,
    dim: int = 0,
    overlap: int = 0
) -> List[torch.Tensor]:
    """
    Split tensor into chunks with optional overlap.
    
    Args:
        tensor: Input tensor to chunk
        chunk_size: Size of each chunk
        dim: Dimension along which to chunk
        overlap: Number of elements to overlap between chunks
    
    Returns:
        List of tensor chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    tensor_size = tensor.size(dim)
    chunks = []
    
    start = 0
    while start < tensor_size:
        end = min(start + chunk_size, tensor_size)
        
        # Select chunk along the specified dimension
        indices = torch.arange(start, end, device=tensor.device)
        chunk = torch.index_select(tensor, dim, indices)
        chunks.append(chunk)
        
        # Move start position considering overlap
        start += chunk_size - overlap
        
        # Break if we've reached the end
        if end == tensor_size:
            break
    
    return chunks


def flatten_dict_tensors(tensor_dict: dict, prefix: str = '') -> dict:
    """
    Flatten nested dictionary of tensors.
    
    Args:
        tensor_dict: Dictionary containing tensors or nested dicts
        prefix: Prefix for flattened keys
    
    Returns:
        Flattened dictionary
    """
    flattened = {}
    
    for key, value in tensor_dict.items():
        new_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_dict_tensors(value, new_key))
        elif isinstance(value, torch.Tensor):
            flattened[new_key] = value
        else:
            flattened[new_key] = value
    
    return flattened


def unflatten_dict_tensors(flattened_dict: dict) -> dict:
    """
    Unflatten dictionary of tensors back to nested structure.
    
    Args:
        flattened_dict: Flattened dictionary
    
    Returns:
        Nested dictionary
    """
    unflattened = {}
    
    for key, value in flattened_dict.items():
        keys = key.split('.')
        current_dict = unflattened
        
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        
        current_dict[keys[-1]] = value
    
    return unflattened


def tensor_memory_usage(tensor: torch.Tensor) -> float:
    """
    Calculate memory usage of tensor in MB.
    
    Args:
        tensor: Input tensor
    
    Returns:
        Memory usage in megabytes
    """
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def batch_tensor_memory_usage(tensors: List[torch.Tensor]) -> float:
    """
    Calculate total memory usage of list of tensors in MB.
    
    Args:
        tensors: List of tensors
    
    Returns:
        Total memory usage in megabytes
    """
    return sum(tensor_memory_usage(t) for t in tensors if t is not None)


def move_to_device(
    obj: Union[torch.Tensor, dict, list, tuple],
    device: torch.device,
    non_blocking: bool = False
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Recursively move tensors to device.
    
    Args:
        obj: Object containing tensors
        device: Target device
        non_blocking: Whether to use non-blocking transfer
    
    Returns:
        Object with tensors moved to device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device, non_blocking) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in obj)
    else:
        return obj


def detach_tensors(
    obj: Union[torch.Tensor, dict, list, tuple]
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Recursively detach tensors from computation graph.
    
    Args:
        obj: Object containing tensors
    
    Returns:
        Object with detached tensors
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_tensors(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_tensors(item) for item in obj)
    else:
        return obj


def interpolate_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Linear interpolation between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        alpha: Interpolation factor (0 = tensor1, 1 = tensor2)
    
    Returns:
        Interpolated tensor
    """
    return (1 - alpha) * tensor1 + alpha * tensor2


def normalize_tensor(
    tensor: torch.Tensor,
    dim: Optional[int] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    L2 normalize tensor along specified dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize along (if None, normalize entire tensor)
        eps: Small value for numerical stability
    
    Returns:
        Normalized tensor
    """
    if dim is None:
        norm = tensor.norm()
    else:
        norm = tensor.norm(dim=dim, keepdim=True)
    
    return tensor / (norm + eps)