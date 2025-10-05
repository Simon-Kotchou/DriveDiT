"""
Model utility functions.
Helper functions for model management and operations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Iterator
import random
import numpy as np


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_parameters(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze all parameters in a model.
    
    Args:
        model: PyTorch model
        freeze: Whether to freeze (True) or unfreeze (False) parameters
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def freeze_layers(model: nn.Module, layer_names: List[str]) -> None:
    """
    Freeze specific layers in a model.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break


def get_device(model: nn.Module) -> torch.device:
    """
    Get the device of a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Device of the model
    """
    return next(model.parameters()).device


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    return total_size / (1024 * 1024)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_weights(model: nn.Module, init_type: str = 'normal', gain: float = 0.02) -> None:
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        init_type: Type of initialization ('normal', 'xavier', 'kaiming', 'orthogonal')
        gain: Gain factor for initialization
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'Initialization method {init_type} not implemented')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    model.apply(init_func)


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip gradients of model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    
    Returns:
        Total norm of gradients before clipping
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_gradient_norm(model: nn.Module) -> float:
    """
    Get the total gradient norm of model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** (1. / 2)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    **kwargs
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load checkpoint to
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def exponential_moving_average(
    model: nn.Module,
    ema_model: nn.Module,
    decay: float = 0.999
) -> None:
    """
    Update exponential moving average of model parameters.
    
    Args:
        model: Current model
        ema_model: EMA model to update
        decay: EMA decay factor
    """
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
    
    Returns:
        Model summary string
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {}
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1
            
            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params
        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    # Create properties
    summary = {}
    hooks = []
    
    # Register hook
    model.apply(register_hook)
    
    # Make a forward pass
    device = get_device(model)
    x = torch.randn(1, *input_size).to(device)
    
    try:
        model(x)
    except Exception as e:
        # Clean up hooks
        for h in hooks:
            h.remove()
        raise e
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Create summary string
    summary_str = "Model Summary\n"
    summary_str += "=" * 70 + "\n"
    summary_str += f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}\n"
    summary_str += "=" * 70 + "\n"
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        # Input_shape, output_shape, trainable, nb_params
        line_new = f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['nb_params']:>15,}\n"
        total_params += summary[layer]['nb_params']
        
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable']:
                trainable_params += summary[layer]['nb_params']
        
        summary_str += line_new
    
    summary_str += "=" * 70 + "\n"
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {trainable_params:,}\n"
    summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"
    summary_str += "=" * 70
    
    return summary_str


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, bool]:
    """
    Compare two models for parameter equality.
    
    Args:
        model1: First model
        model2: Second model
    
    Returns:
        Dictionary with comparison results
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(model1.state_dict().items(), model2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print(f"Mismatch found at {key_item_1[0]}")
            else:
                raise Exception
    
    return {'models_differ': models_differ > 0, 'num_differences': models_differ}


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Get layer by name from model.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
    
    Returns:
        Layer module
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    
    raise ValueError(f"Layer {layer_name} not found in model")


def replace_layer(
    model: nn.Module,
    layer_name: str,
    new_layer: nn.Module
) -> None:
    """
    Replace a layer in the model.
    
    Args:
        model: PyTorch model
        layer_name: Name of layer to replace
        new_layer: New layer module
    """
    # Split the layer name
    names = layer_name.split('.')
    
    # Navigate to parent module
    parent = model
    for name in names[:-1]:
        parent = getattr(parent, name)
    
    # Replace the layer
    setattr(parent, names[-1], new_layer)