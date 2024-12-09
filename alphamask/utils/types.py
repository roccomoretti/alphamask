from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional, Callable, Any
from pathlib import Path
import os


def snake_to_camel(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


class MaskingStrategy(Enum):
    MASK_POSITIONS = "mask_positions"
    UNMASK_POSITIONS = "unmask_positions"
    ITERATIVE_SINGLE = "iterative_single"
    ITERATIVE_SINGLE_MASK_MUTATE = "iterative_single_mask_mutate"


@dataclass
class ExperimentConfig:
    sequence: str
    jobname_prefix: str
    parent_path: str
    masking_strategy: MaskingStrategy
    positions: Optional[List[int]] = None
    num_recycles: int = 12
    num_seeds: int = 12
    msa_method: str = "custom_a3m"
    custom_a3m_path: str = ""
    mutations: Optional[List[str]] = None
    run_control: bool = True
    run_only_control: bool = False
    unified_memory: bool = False
    setup_base_path: Optional[str] = None
    callback_fn: Optional[Callable[[Any, Optional[str]], None]] = None

    def copy(self) -> 'ExperimentConfig':
        """Create a deep copy of the config."""
        return ExperimentConfig(
            sequence=self.sequence,
            jobname_prefix=self.jobname_prefix,
            parent_path=self.parent_path,
            masking_strategy=self.masking_strategy,
            positions=self.positions.copy() if self.positions else None,
            num_recycles=self.num_recycles,
            num_seeds=self.num_seeds,
            msa_method=self.msa_method,
            custom_a3m_path=self.custom_a3m_path,
            mutations=self.mutations.copy() if self.mutations else None,
            run_control=self.run_control,
            run_only_control=self.run_only_control,
            unified_memory=self.unified_memory,
            setup_base_path=self.setup_base_path,
            callback_fn=self.callback_fn
        )

    def get_setup_path(self) -> Path:
        """Get the setup directory path for this experiment."""
        setup_base = Path(self.setup_base_path or os.path.expanduser("~/alphamask_setup"))
        return setup_base

    def ensure_setup_directory(self) -> None:
        """Ensure the setup directory exists."""
        setup_path = self.get_setup_path()
        setup_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert the config to a dictionary, handling special types and field name conversion."""
        d = asdict(self)
        # Convert Enum to string
        d['masking_strategy'] = self.masking_strategy.value
        # Remove callback function as it's not serializable
        d.pop('callback_fn', None)
        
        # Convert field names from snake_case to camelCase
        converted = {}
        for key, value in d.items():
            if key == 'parent_path':
                parent_path = Path(value)
                converted['parentPath'] = str(parent_path)
                # Create setup directory and add its path
                setup_path = self.get_setup_path()
                self.ensure_setup_directory()
                converted['setupPath'] = str(setup_path)  # Now points to the shared setup directory
            elif key == 'jobname_prefix':
                converted['jobname'] = value  # Special case: jobname_prefix -> jobname
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
            elif key == 'setup_base_path':
                pass  # Skip this field as it's only used internally
            else:
                converted[key] = value  # Keep other fields as is
                
        # Add required fields with default values if not present
        defaults = {
            'copies': 1,
            'pair_mode': 'unpaired',
            'cov': 0,
            'id': 90,
            'qid': 0,
            'do_not_filter': False,
            'template_mode': 'none',
            'pdb': '',
            'chain': '',
            'rm_template_seq': False,
            'propagate_to_copies': False,
            'do_not_align': False,
            'model_type': 'monomer (ptm)',
            'rank_by': 'plddt',
            'debug': False,
            'use_initial_guess': False,
            'num_msa': 512,
            'num_extra_msa': 1024,
            'use_cluster_profile': True,
            'model': 'all',
            'recycle_early_stop_tolerance': 0.5,
            'select_best_across_recycles': True,
            'use_mlm': False,
            'use_dropout': True,
            'seed': 42,
            'show_images': True,
            'masking_mode': 'off',
            'mask_msa': False,
            'mask_deletion_matrix': False,
            'cols': [],
            'cols_range': [],
            'mask_identity': 'X',
            'overwrite': True,
            'show_figures': True  # Add show_figures with default value True
        }
        
        for key, value in defaults.items():
            if key not in converted:
                converted[key] = value
                
        return converted


@dataclass
class ExperimentPaths:
    vanilla_jobname: Optional[str] = None
    vanilla_path: Optional[str] = None
    vanilla_pipeline: Optional[Any] = None
    mask_mutate_jobname: Optional[str] = None
    mask_mutate_path: Optional[str] = None
    mask_mutate_pipeline: Optional[Any] = None
    mask_jobname: Optional[str] = None
    mask_path: Optional[str] = None
    mask_pipeline: Optional[Any] = None
