from typing import List, Optional
from pathlib import Path
import yaml

from .base import Experiment, Control, ControlType, ProteinSystem
from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.slurm import SlurmJobManager, SlurmJobConfig
from ..utils.params import load_defaults

# Load defaults at module level
DEFAULTS = load_defaults()

class FrustraMaskingExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("frustra_masking", protein, slurm_config)
        
        if not protein.frustra_positions:
            raise ValueError("Frustra positions must be provided")
            
        self.frustra_positions = protein.frustra_positions
        # Default mutations for I89 experiment
        self.frustra_mutations = ["I89S", "D72D", "G73G", "S74S", "G75G", "T76T", "E81E"]
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)
        
        # Add mutation controls for each mutation without masking
        for mutation in self.frustra_mutations:
            mutation_control = Control(
                f"mutation_{mutation}", 
                ControlType.MUTATION,  # Using MUTATION type will automatically set pipeline_type="mutate"
                protein,
                self.slurm_config,
                mutations=[mutation]
            )
            self.controls.append(mutation_control)

    def run_frustra_masking(self) -> bool:
        """Run Frustra-guided masking experiment"""
        config = ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_frustra",
            masking_strategy=MaskingStrategy.MASK_POSITIONS,
            positions=self.frustra_positions,
            parent_path=str(self.working_dir),
            num_recycles=DEFAULTS.get('num_recycles', 2),
            num_seeds=DEFAULTS.get('num_seeds', 2),
            msa_method=self.protein.msa_method,
            custom_a3m_path=self.protein.custom_a3m_path,
            setup_path=str(self.slurm_config.setup_path),
            mutations=self.frustra_mutations,  # Add mutations to be applied
            pipeline_type="mutate_and_mask"  # Use mutate_and_mask pipeline
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