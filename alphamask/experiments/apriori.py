from pathlib import Path
import logging
from typing import Optional, List

from ..utils.slurm import SlurmJobManager, SlurmJobConfig
from ..utils.types import ExperimentConfig
from ..utils.params import load_defaults

from .base import BaseExperiment, Control, ExperimentError
from .types import ProteinConfig, Condition, AprioriExperiment as AprioriConfig

logger = logging.getLogger(__name__)

class AprioriExperiment(BaseExperiment):
    """Implementation of a priori masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None
    ):
        super().__init__("apriori_masking", protein_config, slurm_config, working_dir)
        
        if not protein_config.apriori_masking.enabled:
            raise ExperimentError("A priori masking is not enabled in configuration")
            
        self.experiments = protein_config.apriori_masking.experiments
    
    def validate(self) -> None:
        """Validate a priori masking configuration"""
        if not self.experiments:
            raise ExperimentError("No experiments specified for a priori masking")
        
        for experiment in self.experiments:
            # Validate experiment name
            if not experiment.name:
                raise ExperimentError("Experiment name cannot be empty")
            
            # Validate positions
            if not experiment.positions:
                raise ExperimentError(f"No positions specified for experiment {experiment.name}")
            
            for pos in experiment.positions:
                if pos < 1 or pos > len(self.protein_config.sequence):
                    raise ExperimentError(
                        f"Invalid position {pos} in experiment {experiment.name}"
                    )
            
            # Validate mutations and conditions
            has_mutations = bool(experiment.mutations)
            required_conditions = [
                (False, False), (True, False), (False, True), (True, True)
            ] if has_mutations else [
                (False, False), (True, False), (False, True)
            ]
            
            if has_mutations:
                for mutation in experiment.mutations:
                    if not self._validate_mutation(mutation):
                        raise ExperimentError(
                            f"Invalid mutation {mutation} in experiment {experiment.name}"
                        )
            
            experiment_conditions = [
                (c.mask, c.mutate) for c in experiment.conditions
            ]
            if sorted(experiment_conditions) != sorted(required_conditions):
                raise ExperimentError(
                    f"Experiment {experiment.name} requires exactly "
                    f"{'4 conditions when mutations are specified' if has_mutations else '3 conditions when no mutations are specified'}"
                )
    
    def _validate_mutation(self, mutation: str) -> bool:
        """Validate a single mutation against the protein sequence"""
        if len(mutation) < 3:
            return False
            
        orig_aa = mutation[0]
        new_aa = mutation[-1]
        pos = int(mutation[1:-1])
        
        if pos < 1 or pos > len(self.protein_config.sequence):
            return False
            
        if self.protein_config.sequence[pos-1] != orig_aa:
            return False
            
        return True
    
    def setup(self) -> None:
        """No separate setup needed as conditions are handled directly in run"""
        pass
    
    def _create_config(self, experiment: AprioriConfig) -> ExperimentConfig:
        """Create configuration for a specific a priori experiment"""
        return ExperimentConfig(
            sequence=self.protein_config.sequence,
            jobname_prefix=f"{self.name}_{experiment.name}",
            parent_path=str(self.working_dir / experiment.name),
            positions=experiment.positions,
            mutations=experiment.mutations,
            num_recycles=self.defaults.get('num_recycles', 2),
            num_seeds=self.defaults.get('num_seeds', 2),
            setup_path=str(self.slurm_config.setup_path),
            pipeline_type="mutate_and_mask" if experiment.mutations else "masking"
        )
    
    def submit(self) -> bool:
        """Submit a priori masking experiment jobs to SLURM
        
        This method validates the experiment configuration and submits jobs for each
        experiment with their associated conditions. For each experiment, it:
        1. Creates the experiment configuration
        2. Sets up a SLURM job manager
        3. Submits the jobs through the job manager
        
        Returns:
            bool: True if all jobs were submitted successfully, False otherwise
        """
        self.validate()
        
        # Submit each experiment with its conditions
        success = True
        for experiment in self.experiments:
            config = self._create_config(experiment)
            
            job_manager = SlurmJobManager(
                experiment_config=config,
                slurm_config=self.slurm_config,
                working_dir=str(self.working_dir / experiment.name)
            )
            
            exp_success, failed_jobs = job_manager.run_experiment()
            if not exp_success:
                logger.error(f"Failed to submit experiment {experiment.name}")
                success = False
        
        return success 