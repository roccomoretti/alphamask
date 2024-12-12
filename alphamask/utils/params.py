import os
from pathlib import Path
import yaml

def load_defaults() -> dict:
    """Load default parameters from YAML file."""
    defaults_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
    with open(defaults_path, 'r') as f:
        return yaml.safe_load(f)

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
