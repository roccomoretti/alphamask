"""Types module for AlphaMask"""

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Callable, Any, TypedDict, Dict, Union
from enum import Enum
from pathlib import Path
import os
import yaml

class AlphaFoldAuxData(TypedDict, total=False):
    """Type definition for AlphaFold auxiliary data"""
    plddt: Any  # Predicted LDDT scores
    pae: Optional[Any]  # Predicted Aligned Error
    ptm: Optional[float]  # Predicted TM-score
    confidence: Optional[float]  # Confidence score
    atom_positions: Optional[Any]  # Atom positions

class AlphaFoldTempData(TypedDict):
    """Type definition for AlphaFold temporary data"""
    best: Dict[str, Any]  # Contains best prediction data
    aux: AlphaFoldAuxData  # Auxiliary data
    traj: Dict[str, List]  # Trajectory data
    log: List[Any]  # Log data

@dataclass
class AlphaFoldResult:
    """Result from AlphaFold prediction"""
    success: bool
    error: Optional[str]
    data: Optional[Dict[str, Any]]

def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

class MaskingStrategy(str, Enum):
    """Enum for masking strategies"""
    NONE = "none"
    ITERATIVE_SINGLE = "iterative_single"
    ITERATIVE_SINGLE_MASK_MUTATE = "iterative_single_mask_mutate"
    MASK_POSITIONS = "mask_positions"
    UNMASK_POSITIONS = "unmask_positions"
    MUTATE_AND_MASK = "mutate_and_mask"

@dataclass
class ExperimentConfig:
    """Configuration for AlphaMask experiments."""
    # Required fields
    sequence: str
    jobname_prefix: str
    parent_path: str
    masking_strategy: MaskingStrategy
    
    # Optional fields - defaults will be loaded from defaults.yaml
    masking_mode: str = None
    mask_msa: bool = None
    mask_deletion_matrix: bool = None
    mask_identity: str = None
    positions: Optional[List[int]] = None
    num_recycles: int = None
    num_seeds: int = None
    msa_method: str = None
    custom_a3m_path: str = None
    mutations: Optional[List[str]] = None
    run_control: bool = None
    run_only_control: bool = None
    unified_memory: bool = None
    setup_path: Optional[str] = None
    debug: bool = None
    pipeline_type: str = None
    callback_fn: Optional[Callable[[Any, Optional[str]], None]] = None
    create_control: bool = True
    
    def __post_init__(self):
        """Load defaults after initialization."""
        from .params import load_defaults
        defaults = load_defaults()
        
        # Set defaults for any None values
        for field in self.__dataclass_fields__:
            if field != 'callback_fn' and getattr(self, field) is None:
                if field in defaults:
                    setattr(self, field, defaults[field])
    
    def copy(self) -> 'ExperimentConfig':
        """Create a deep copy of the config."""
        return ExperimentConfig(
            sequence=self.sequence,
            jobname_prefix=self.jobname_prefix,
            parent_path=self.parent_path,
            masking_strategy=self.masking_strategy,
            masking_mode=self.masking_mode,
            mask_msa=self.mask_msa,
            mask_deletion_matrix=self.mask_deletion_matrix,
            mask_identity=self.mask_identity,
            positions=self.positions.copy() if self.positions else None,
            num_recycles=self.num_recycles,
            num_seeds=self.num_seeds,
            msa_method=self.msa_method,
            custom_a3m_path=self.custom_a3m_path,
            mutations=self.mutations.copy() if self.mutations else None,
            run_control=self.run_control,
            run_only_control=self.run_only_control,
            unified_memory=self.unified_memory,
            setup_path=self.setup_path,
            debug=self.debug,
            pipeline_type=self.pipeline_type,
            callback_fn=self.callback_fn
        )
    
    def get_setup_path(self) -> Path:
        """Get the setup directory path for this experiment."""
        setup_base = Path(self.setup_path or os.path.expanduser("~/alphamask_setup"))
        return setup_base

    def ensure_setup_directory(self) -> None:
        """Ensure the setup directory exists."""
        setup_path = self.get_setup_path()
        setup_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert the config to a dictionary, handling special types and field name conversion."""
        from .params import load_defaults
        
        d = asdict(self)
        # Convert Enum to string
        d['masking_strategy'] = self.masking_strategy.value
        # Remove callback function as it's not serializable
        d.pop('callback_fn', None)
        
        # Load defaults from YAML
        defaults = load_defaults()
        
        # Convert field names from snake_case to camelCase
        converted = {}
        for key, value in d.items():
            if key == 'parent_path':
                parent_path = Path(value)
                converted['parent_path'] = str(parent_path)
                # Create setup directory and add its path
                setup_path = self.get_setup_path()
                self.ensure_setup_directory()
                converted['setup_path'] = str(setup_path)  # Now points to the shared setup directory
            elif key == 'jobname_prefix':
                # Keep both jobname and jobname_prefix for compatibility
                converted['jobname'] = value if value else ""  # Use empty string if None
                converted['jobname_prefix'] = value if value else ""
            elif key == 'unified_memory':
                converted['unified_memory'] = value  # Keep as snake_case to match schema
            elif key == 'msa_method':
                converted['msa_method'] = value  # Keep as snake_case to match schema
            elif key == 'custom_a3m_path':
                converted['custom_a3m_path'] = value  # Keep as snake_case to match schema
            elif key == 'num_recycles':
                converted['num_recycles'] = value  # Keep as snake_case to match schema
            elif key == 'num_seeds':
                converted['num_seeds'] = value  # Keep as snake_case to match schema
            elif key == 'num_msa':
                converted['num_msa'] = value  # Keep as snake_case to match schema
            elif key == 'num_extra_msa':
                converted['num_extra_msa'] = value  # Keep as snake_case to match schema
            elif key == 'mutations':
                converted['mutations'] = value if value is not None else []  # Ensure mutations is a list
            elif key == 'positions':
                converted['positions'] = value if value is not None else []  # Ensure positions is a list
            elif key == 'setup_path':
                pass  # Skip this field as it's only used internally
            else:
                converted[key] = value  # Keep other fields as is
                
        # Add any missing fields from defaults
        for key, value in defaults.items():
            if key not in converted:
                converted[key] = value
                
        return converted

    def save(self, path: Union[str, Path]) -> None:
        """Save the configuration to a YAML file.
        
        Args:
            path: Path to save the YAML file
        """
        # Convert path to Path object
        path = Path(path)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary
        config_dict = self.to_dict()
        
        # Save to YAML file
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

@dataclass
class ExperimentPaths:
    """Paths for experiment outputs and resources"""
    vanilla_jobname: Optional[str] = None
    vanilla_path: Optional[str] = None
    vanilla_pipeline: Optional[Any] = None
    mask_mutate_jobname: Optional[str] = None
    mask_mutate_path: Optional[str] = None
    mask_mutate_pipeline: Optional[Any] = None
    mask_jobname: Optional[str] = None
    mask_path: Optional[str] = None
    mask_pipeline: Optional[Any] = None

# Export all types
__all__ = [
    'AlphaFoldAuxData',
    'AlphaFoldTempData',
    'AlphaFoldResult',
    'MaskingStrategy',
    'ExperimentConfig',
    'ExperimentPaths',
    'snake_to_camel'
]
