from typing import List, Optional
from pathlib import Path
from itertools import combinations

from .base import Experiment, Control, ControlType, ProteinSystem
from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.slurm import SlurmJobManager, SlurmJobConfig

class MutationExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, mutations: List[str], slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("mutation_masking", protein, slurm_config)
        
        if not mutations:
            raise ValueError("Mutations must be provided")
        
        self.mutations = mutations
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)
        
        # Add mutation control
        self.mutation_control = Control(
            "mutation", 
            ControlType.MUTATION, 
            protein,
            self.slurm_config,
            mutations=mutations
        )
        self.controls.append(self.mutation_control)

    def run_mutation_masking(self) -> bool:
        """Run mutation masking experiment"""
        config = ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_mutation",
            masking_strategy=MaskingStrategy.ITERATIVE_SINGLE_MASK_MUTATE,
            parent_path=str(self.working_dir),
            num_recycles=12,
            num_seeds=12,
            msa_method=self.protein.msa_method,
            custom_a3m_path=self.protein.custom_a3m_path,
            mutations=self.mutations
        )
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success

    def run(self) -> bool:
        """Run experiment and controls"""
        if not super().run():  # This runs the controls
            return False
            
        return self.run_mutation_masking()

class DoubleMutationExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, mutations: List[str], slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("double_mutation_masking", protein, slurm_config)
        
        if len(mutations) < 2:
            raise ValueError("At least two mutations must be provided")
        
        self.mutations = mutations
        self.mutation_pairs = list(combinations(mutations, 2))
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)
        
        # Add single mutation controls
        self.single_mutation_controls = []
        for mutation in mutations:
            control = Control(
                f"mutation_{mutation}", 
                ControlType.MUTATION, 
                protein,
                self.slurm_config,
                mutations=[mutation]
            )
            self.single_mutation_controls.append(control)
            self.controls.append(control)

    def run_double_mutation_masking(self) -> bool:
        """Run double mutation masking experiments"""
        success = True
        
        for mut_pair in self.mutation_pairs:
            config = ExperimentConfig(
                sequence=self.protein.sequence,
                jobname_prefix=f"{self.protein.name}_double_mutation_{'_'.join(mut_pair)}",
                masking_strategy=MaskingStrategy.ITERATIVE_SINGLE_MASK_MUTATE,
                parent_path=str(self.working_dir),
                num_recycles=12,
                num_seeds=12,
                msa_method=self.protein.msa_method,
                custom_a3m_path=self.protein.custom_a3m_path,
                mutations=list(mut_pair)
            )
            
            job_manager = SlurmJobManager(
                experiment_config=config,
                slurm_config=self.slurm_config,
                working_dir=str(self.working_dir / f"pair_{'_'.join(mut_pair)}")
            )
            
            pair_success, failed_jobs = job_manager.run_experiment()
            success = success and pair_success
        
        return success

    def run(self) -> bool:
        """Run experiment and controls"""
        if not super().run():  # This runs the controls
            return False
            
        return self.run_double_mutation_masking() 