"""Parameter handling utilities for AlphaMask"""

import yaml
from pathlib import Path
from typing import Dict, Any

def get_default_params_path() -> Path:
    """Get the path to the default parameters file"""
    return Path(__file__).parent.parent / "config" / "defaults.yaml"

def load_defaults() -> Dict[str, Any]:
    """Load default parameters from YAML file"""
    try:
        params_path = get_default_params_path()
        if not params_path.exists():
            # Return minimal defaults if file doesn't exist
            return {
                "masking_mode": "list",
                "mask_msa": True,
                "mask_deletion_matrix": True,
                "mask_identity": "X",
                "num_recycles": 2,
                "num_seeds": 2,
                "msa_method": "mmseqs2",
                "run_control": True,
                "run_only_control": False,
                "unified_memory": False,
                "debug": False,
                "pipeline_type": "default"
            }
        
        with open(params_path) as f:
            defaults = yaml.safe_load(f)
        
        return defaults
        
    except Exception as e:
        # Return minimal defaults on error
        return {
            "masking_mode": "list",
            "mask_msa": True,
            "mask_deletion_matrix": True,
            "mask_identity": "X",
            "num_recycles": 2,
            "num_seeds": 2,
            "msa_method": "mmseqs2",
            "run_control": True,
            "run_only_control": False,
            "unified_memory": False,
            "debug": False,
            "pipeline_type": "default"
        }

def create_common_params(config):
    """Create common parameters for all runs."""
    # Load defaults
    params = load_defaults()
    
    # Override with config values
    overrides = {
        "sequence": config.sequence,
        "custom_a3m_path": config.custom_a3m_path,
        "msa_method": config.msa_method,
        "num_recycles": config.num_recycles,
        "num_seeds": config.num_seeds,
        "parent_path": config.parent_path,
        "unified_memory": config.unified_memory,
        "mutations": config.mutations if config.mutations else [],
    }
    
    # Update defaults with overrides
    params.update(overrides)
    
    return params
