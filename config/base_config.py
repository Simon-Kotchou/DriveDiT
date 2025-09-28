"""
Base configuration classes and utilities.
Zero-dependency configuration management using only Python standard library.
"""

import json
from dataclasses import dataclass, asdict, fields
from typing import Dict, Any, Type, TypeVar, Optional
from pathlib import Path

T = TypeVar('T', bound='BaseConfig')


@dataclass
class BaseConfig:
    """
    Base configuration class with JSON serialization support.
    Uses only Python standard library for zero dependencies.
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert config to JSON string or save to file.
        
        Args:
            filepath: Optional path to save JSON file
            
        Returns:
            JSON string representation
        """
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config instance
        """
        # Filter only fields that exist in the dataclass
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls: Type[T], json_str_or_path: str) -> T:
        """
        Create config from JSON string or file.
        
        Args:
            json_str_or_path: JSON string or path to JSON file
            
        Returns:
            Config instance
        """
        if Path(json_str_or_path).exists():
            # Load from file
            with open(json_str_or_path, 'r') as f:
                config_dict = json.load(f)
        else:
            # Parse as JSON string
            config_dict = json.loads(json_str_or_path)
        
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> None:
        """
        Update config with new values.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config key: {key}")
    
    def merge(self: T, other: T) -> T:
        """
        Merge with another config instance.
        
        Args:
            other: Other config instance
            
        Returns:
            New merged config instance
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Update with non-None values from other
        for key, value in other_dict.items():
            if value is not None:
                self_dict[key] = value
        
        return self.__class__.from_dict(self_dict)
    
    def validate(self) -> None:
        """
        Validate configuration values.
        Override in subclasses for specific validation.
        """
        pass
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()


def load_config(config_class: Type[T], config_path: str) -> T:
    """
    Load configuration from file.
    
    Args:
        config_class: Configuration class type
        config_path: Path to configuration file
        
    Returns:
        Configuration instance
    """
    return config_class.from_json(config_path)


def save_config(config: BaseConfig, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration instance
        config_path: Path to save configuration
    """
    config.to_json(config_path)


def merge_configs(*configs: BaseConfig) -> BaseConfig:
    """
    Merge multiple configurations.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration instances to merge
        
    Returns:
        Merged configuration
    """
    if not configs:
        raise ValueError("At least one config must be provided")
    
    result = configs[0]
    for config in configs[1:]:
        result = result.merge(config)
    
    return result


def create_config_from_args(config_class: Type[T], args_dict: Dict[str, Any]) -> T:
    """
    Create configuration from command line arguments dictionary.
    
    Args:
        config_class: Configuration class type
        args_dict: Dictionary from argparse or similar
        
    Returns:
        Configuration instance
    """
    # Convert argparse namespace to dict if needed
    if hasattr(args_dict, '__dict__'):
        args_dict = vars(args_dict)
    
    # Filter None values
    filtered_args = {k: v for k, v in args_dict.items() if v is not None}
    
    return config_class.from_dict(filtered_args)


def get_config_diff(config1: BaseConfig, config2: BaseConfig) -> Dict[str, tuple]:
    """
    Get differences between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary of differences {key: (value1, value2)}
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    diff = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in all_keys:
        val1 = dict1.get(key, None)
        val2 = dict2.get(key, None)
        
        if val1 != val2:
            diff[key] = (val1, val2)
    
    return diff