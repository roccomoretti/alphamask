from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

from ..utils.types import ExperimentConfig, MaskingStrategy, AlphaFoldResult
from ..utils.slurm import SlurmJobManager, SlurmJobConfig
from ..utils.params import load_defaults

class ControlType(Enum):
    VANILLA = "vanilla"
    MUTATION = "mutation"
    DOUBLE_MUTATION = "double_mutation"

@dataclass
class ProteinSystem:
    name: str
    sequence: str
    mutations: Optional[List[str]] = None
    known_positions: Optional[List[int]] = None
    frustra_positions: Optional[List[int]] = None
    msa_method: str = "mmseqs2"
    custom_a3m_path: str = ""
    parent_path: str = "/path/to/experiments"

    def __post_init__(self):
        """Initialize paths after dataclass initialization"""
        self.parent_path = str(Path(self.parent_path).resolve())

class Control:
    def __init__(
        self, 
        name: str, 
        control_type: ControlType, 
        protein: ProteinSystem,
        slurm_config: SlurmJobConfig,
        mutations: Optional[List[str]] = None
    ):
        self.name = name
        self.type = control_type
        self.protein = protein
        self.slurm_config = slurm_config
        self.mutations = mutations
        self.working_dir = Path(self.protein.parent_path).resolve() / self.protein.name / "controls" / name
        self.config_dir = self.working_dir / "configs"
        self.script_dir = self.working_dir / "scripts"
        self.log_dir = self.working_dir / "logs"

    def _create_config(self) -> ExperimentConfig:
        """Create configuration based on control type."""
        # Load defaults
        defaults = load_defaults()
        
        # Override with control-specific values
        config = {
            "sequence": self.protein.sequence,
            "jobname_prefix": f"{self.protein.name}_{self.name}",
            "parent_path": str(self.working_dir),
            "msa_method": self.protein.msa_method,
            "custom_a3m_path": self.protein.custom_a3m_path,
            "run_only_control": True,
            "pipeline_type": "default"  # Controls use default pipeline by default
        }
        
        # Handle mutations based on control type
        if self.type in [ControlType.MUTATION, ControlType.DOUBLE_MUTATION]:
            if not self.mutations:
                raise ValueError(f"{self.type.value} control requires mutations")
            config["mutations"] = self.mutations
            config["pipeline_type"] = "mutate"  # Mutation controls use mutate pipeline
        
        # Update defaults with control config
        defaults.update(config)
        
        # Filter out fields that aren't in ExperimentConfig
        valid_fields = {
            'sequence', 'jobname_prefix', 'parent_path', 'masking_strategy',
            'masking_mode', 'mask_msa', 'mask_deletion_matrix', 'mask_identity',
            'positions', 'num_recycles', 'num_seeds', 'msa_method', 'custom_a3m_path',
            'mutations', 'run_control', 'run_only_control', 'unified_memory',
            'setup_path', 'debug', 'pipeline_type', 'callback_fn'
        }
        
        filtered_defaults = {k: v for k, v in defaults.items() if k in valid_fields}
        
        # Convert masking_strategy string to enum
        if 'masking_strategy' in filtered_defaults:
            strategy_str = filtered_defaults['masking_strategy']
            try:
                filtered_defaults['masking_strategy'] = MaskingStrategy(strategy_str)
            except ValueError:
                # If conversion fails, use ITERATIVE_SINGLE as default
                filtered_defaults['masking_strategy'] = MaskingStrategy.ITERATIVE_SINGLE
        else:
            filtered_defaults['masking_strategy'] = MaskingStrategy.ITERATIVE_SINGLE
        
        return ExperimentConfig(**filtered_defaults)

    def run(self) -> bool:
        """Run control experiment"""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.script_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        config = self._create_config()
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success

class Experiment:
    def __init__(
        self, 
        name: str, 
        protein: ProteinSystem,
        slurm_config: Optional[SlurmJobConfig] = None
    ):
        self.name = name
        self.protein = protein
        self.slurm_config = slurm_config or SlurmJobConfig()
        self.controls: List[Control] = []
        self.working_dir = Path(self.protein.parent_path) / self.protein.name / name
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Validate experiment configuration"""
        if not self.protein.sequence:
            raise ValueError("Protein sequence is required")
        
        if self.protein.mutations:
            for mut in self.protein.mutations:
                if not (len(mut) >= 3 and mut[0].isalpha() and mut[-1].isalpha() and mut[1:-1].isdigit()):
                    raise ValueError(f"Invalid mutation format: {mut}")
        
        if self.protein.custom_a3m_path and not Path(self.protein.custom_a3m_path).exists():
            raise ValueError(f"Custom MSA file not found: {self.protein.custom_a3m_path}")

    def run(self) -> bool:
        """Run experiment and its controls"""
        self.validate()
        
        # Run controls first
        for control in self.controls:
            if not control.run():
                return False
        
        return True 