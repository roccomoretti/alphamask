from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.slurm import SlurmJobManager, SlurmJobConfig

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
        return ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_{self.name}",
            masking_mode="on",
            masking_strategy=MaskingStrategy.ITERATIVE_SINGLE,
            mask_msa=True,
            parent_path=str(self.working_dir),
            num_recycles=12,
            num_seeds=12,
            msa_method=self.protein.msa_method,
            custom_a3m_path=self.protein.custom_a3m_path,
            mutations=self.mutations,
            run_only_control=True
        )

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