from typing import List, Optional
from pathlib import Path

from .base import Experiment, Control, ControlType, ProteinSystem
from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.slurm import SlurmJobManager, SlurmJobConfig

class FrustraMaskingExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("frustra_masking", protein, slurm_config)
        
        if not protein.frustra_positions:
            raise ValueError("Frustra positions must be provided")
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)

    def run_frustra_masking(self) -> bool:
        """Run Frustra-guided masking experiment"""
        config = ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_frustra",
            masking_strategy=MaskingStrategy.MASK_POSITIONS,
            positions=self.protein.frustra_positions,
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
            
        return self.run_frustra_masking() 