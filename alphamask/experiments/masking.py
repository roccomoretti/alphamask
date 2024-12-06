from typing import List, Optional
from pathlib import Path

from .base import Experiment, Control, ControlType, ProteinSystem
from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.slurm import SlurmJobManager, SlurmJobConfig

class IterativeMaskingExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("iterative_masking", protein, slurm_config)
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)

    def run_iterative_masking(self) -> bool:
        """Run iterative masking experiment"""
        config = ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_iterative",
            masking_strategy=MaskingStrategy.ITERATIVE_SINGLE,
            parent_path=str(self.working_dir),
            num_recycles=12,
            num_seeds=12,
            msa_method=self.protein.msa_method,
            custom_a3m_path=self.protein.custom_a3m_path
        )
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success

    def run(self) -> bool:
        """Run experiment and control"""
        if not super().run():  # This runs the controls
            return False
            
        return self.run_iterative_masking()

class AprioriMaskingExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("apriori_masking", protein, slurm_config)
        
        if not protein.known_positions:
            raise ValueError("Known positions are required for a priori masking")
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)

    def run_apriori_masking(self) -> bool:
        """Run a priori masking experiment"""
        config = ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_apriori",
            masking_strategy=MaskingStrategy.MASK_POSITIONS,
            positions=self.protein.known_positions,
            parent_path=str(self.working_dir),
            num_recycles=12,
            num_seeds=12,
            msa_method=self.protein.msa_method,
            custom_a3m_path=self.protein.custom_a3m_path
        )
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success

    def run(self) -> bool:
        """Run experiment and control"""
        if not super().run():  # This runs the controls
            return False
            
        return self.run_apriori_masking() 