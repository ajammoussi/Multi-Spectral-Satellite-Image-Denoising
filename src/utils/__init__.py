"""
Utility Module

Helper functions for configuration, checkpointing, visualization, and downloads
"""

from .config import load_config, merge_configs, validate_config
from .checkpointing import CheckpointManager
from .visualization import visualize_restoration, plot_spectral_signatures, plot_training_curves
from .download import (
    download_eurosat_dataset,
    download_satmae_weights,
    setup_project_data,
    verify_downloads
)
from .setup_helpers import (
    setup_config,
    setup_device,
    create_model_from_config,
    create_training_components,
    load_checkpoint,
    setup_training_session,
    print_config_summary
)
from .notebook_helpers import (
    visualize_sample_batch,
    visualize_restoration_comparison,
    plot_training_progress,
    plot_spectral_comparison,
    print_dataset_info,
    print_evaluation_summary
)

__all__ = [
    'load_config',
    'merge_configs',
    'validate_config',
    'CheckpointManager',
    'visualize_restoration',
    'plot_spectral_signatures',
    'plot_training_curves',
    'download_eurosat_dataset',
    'download_satmae_weights',
    'setup_project_data',
    'verify_downloads',
    # Setup helpers
    'setup_config',
    'setup_device',
    'create_model_from_config',
    'create_training_components',
    'load_checkpoint',
    'setup_training_session',
    'print_config_summary',
    # Notebook helpers
    'visualize_sample_batch',
    'visualize_restoration_comparison',
    'plot_training_progress',
    'plot_spectral_comparison',
    'print_dataset_info',
    'print_evaluation_summary'
]
