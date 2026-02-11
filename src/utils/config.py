"""
Configuration Loading and Management

Handles YAML config files with inheritance and merging
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Supports inheritance from base configs using comment syntax:
        # Inherits from: ../base.yaml
    
    Args:
        config_path: Path to config YAML file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Check for inheritance
    base_config_path = _find_base_config(config_path)
    
    if base_config_path:
        logger.info(f"Loading base config from {base_config_path}")
        base_config = load_config(base_config_path)
        config = merge_configs(base_config, config)
    
    logger.info(f"Loaded config from {config_path}")
    
    return config


def _find_base_config(config_path: Path) -> Path:
    """
    Find base config from inheritance comment in YAML file
    
    Looks for: # Inherits from: ../base.yaml
    """
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') and 'Inherits from:' in line:
                # Extract path after "Inherits from:"
                base_path = line.split('Inherits from:')[1].strip()
                base_path = config_path.parent / base_path
                
                if base_path.exists():
                    return base_path
                else:
                    logger.warning(f"Base config not found: {base_path}")
    
    return None


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries
    
    Values in 'override' take precedence over values in 'base'
    
    Args:
        base: Base configuration
        override: Override configuration
    
    Returns:
        Merged configuration
    """
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value
    
    return merged


def save_config(config: Dict, filepath: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved config to {filepath}")


def validate_config(config: Dict) -> bool:
    """
    Validate configuration has all required fields
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ['data', 'model', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data config
    required_data_keys = ['root_dir', 'num_bands', 'image_size']
    for key in required_data_keys:
        if key not in config['data']:
            raise ValueError(f"Missing required data config: {key}")
    
    # Validate model config
    if 'encoder' not in config['model'] or 'decoder' not in config['model']:
        raise ValueError("Model config must have 'encoder' and 'decoder' sections")
    
    # Validate training config
    required_training_keys = ['epochs', 'micro_batch_size']
    for key in required_training_keys:
        if key not in config['training']:
            raise ValueError(f"Missing required training config: {key}")
    
    logger.info("Configuration validation passed")
    return True


def print_config(config: Dict, indent: int = 0):
    """
    Pretty print configuration dictionary
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(' ' * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(' ' * indent + f"{key}: {value}")


def get_project_root() -> Path:
    """Return the project root path by searching for a marker file (setup.py or .git).

    Falls back to the current working directory if none found.
    """
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / 'setup.py').exists() or (parent / '.git').exists():
            return parent
    return Path.cwd()
