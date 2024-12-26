from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
import logging

from ..utils.slurm import SlurmJobManager, SlurmJobConfig
from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.params import load_defaults

from .types import (
    ProteinConfig, Condition, AprioriExperiment,
    IterativeMasking, AprioriMasking, FrustraMasking
)

# Get logger for this module
logger = logging.getLogger("alphamask.experiments.base")

class ExperimentError(Exception):
    """Base class for experiment-related errors"""
    pass

class BaseExperiment(ABC):
    """Abstract base class for all experiments"""
    
    def __init__(
        self,
        name: str,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None
    ):
        self.name = name
        self.protein_config = protein_config
        self.slurm_config = slurm_config or SlurmJobConfig()
        self.working_dir = (working_dir or Path.cwd() / "experiments" / name).resolve()
        self.dry_run = dry_run
        self.schema_path = schema_path.resolve() if schema_path else None
        
        # Create the main working directory
        if not self.dry_run:
            self.working_dir.mkdir(parents=True, exist_ok=True) 
        else:
            logger.info(f"[DRY RUN] Would create directory: {self.working_dir}")
        
        # Load default parameters
        self.defaults = load_defaults()
        
        # Initialize controls list
        self.controls: List[Control] = []
        
    @abstractmethod
    def validate(self) -> None:
        """Validate experiment configuration"""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Set up experiment directories and resources"""
        pass
    
    @abstractmethod
    def submit(self) -> bool:
        """Submit the experiment jobs to SLURM
        
        Returns:
            bool: True if all jobs were submitted successfully, False otherwise
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources after experiment"""
        pass

class Control:
    """Base class for experiment controls"""
    
    def __init__(
        self,
        name: str,
        protein_config: ProteinConfig,
        slurm_config: SlurmJobConfig,
        conditions: List[Condition],
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None
    ):
        self.name = name
        self.protein_config = protein_config
        self.slurm_config = slurm_config
        self.conditions = conditions
        self.dry_run = dry_run
        self.schema_path = schema_path
        
        # Set working directory, ensuring it's under the experiment's controls directory
        if working_dir is None:
            raise ValueError("working_dir must be specified for Control")
            
        # Create the control directory under the specific mode's directory
        self.working_dir = (Path(working_dir) / name).resolve()
        self.defaults = load_defaults()
        
        # Create standard directory structure
        self.config_dir = self.working_dir / "configs"
        self.script_dir = self.working_dir / "scripts"
        self.log_dir = self.working_dir / "logs"
        self.input_dir = self.working_dir / "in"
        self.output_dir = self.working_dir / "out"
        self.schema_dir = self.working_dir / "schema"
        
        if not self.dry_run:
            # Create all directories
            for dir_path in [
                self.working_dir,
                self.config_dir,
                self.script_dir,
                self.log_dir,
                self.input_dir,
                self.input_dir / "msa",
                self.output_dir,
                self.output_dir / "pdbs",
                self.schema_dir
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
        else:
            # Log what would be created in dry run mode
            for dir_path in [
                self.working_dir,
                self.config_dir,
                self.script_dir,
                self.log_dir,
                self.input_dir,
                self.input_dir / "msa",
                self.output_dir,
                self.output_dir / "pdbs",
                self.schema_dir
            ]:
                logger.info(f"[DRY RUN] Would create directory: {dir_path}")
    
    def create_experiment_config(self, condition: Condition) -> ExperimentConfig:
        """Create ExperimentConfig for a specific condition"""
        # Create a unique name for this condition
        condition_name = f"{self.name}_{condition.mask}_{condition.mutate}"
        
        # Get the experiment positions if this is part of an AprioriExperiment
        experiment_positions = []
        mutations = []
        
        if isinstance(self.protein_config.apriori_masking, AprioriMasking):
            for exp in self.protein_config.apriori_masking.experiments:
                if exp.name in self.name:
                    experiment_positions = exp.positions
                    mutations = exp.mutations if condition.mutate else []
                    break
        elif isinstance(self.protein_config.iterative_masking, IterativeMasking):
            mutations = self.protein_config.iterative_masking.mutations if condition.mutate else []
        
        # Map pipeline type - 'vanilla' should be mapped to 'default'
        pipeline_type = "default" if not (condition.mask or condition.mutate) else "mutate_and_mask"
        
        # Create the config
        config = ExperimentConfig(
            sequence=self.protein_config.sequence,
            jobname_prefix=condition_name,
            parent_path=str(self.working_dir),
            masking_strategy=MaskingStrategy.MASK_POSITIONS if condition.mask else MaskingStrategy.NONE,
            positions=experiment_positions if condition.mask else [],
            mutations=mutations,
            num_recycles=self.defaults.get('num_recycles', 2),
            num_seeds=self.defaults.get('num_seeds', 2),
            setup_path=str(self.slurm_config.setup_path),
            pipeline_type=pipeline_type
        )
        
        # Save the config to a file or log it in dry run mode
        config_path = self.config_dir / f"config_{condition_name}.yaml"
        if not self.dry_run:
            config.save(config_path)
            logger.info(f"Created config file at: {config_path}")
            logger.info(f"Config details: masking={condition.mask}, mutate={condition.mutate}, positions={experiment_positions}, mutations={mutations}")
        else:
            logger.info(f"[DRY RUN] Would create config file at: {config_path}")
            logger.info(f"[DRY RUN] Config contents would be:")
            logger.info(f"[DRY RUN] - Sequence: {config.sequence[:20]}...")
            logger.info(f"[DRY RUN] - Masking strategy: {config.masking_strategy}")
            logger.info(f"[DRY RUN] - Positions to mask: {config.positions}")
            logger.info(f"[DRY RUN] - Mutations: {config.mutations}")
            logger.info(f"[DRY RUN] - Pipeline type: {config.pipeline_type}")
        
        return config
    
    def run(self) -> bool:
        """Run control experiments for all conditions"""
        try:
            for condition in self.conditions:
                config = self.create_experiment_config(condition)
                
                job_manager = SlurmJobManager(
                    experiment_config=config,
                    slurm_config=self.slurm_config,
                    working_dir=str(self.working_dir)
                )
                
                success, failed_jobs = job_manager.run_experiment()
                if not success:
                    logger.error(f"Failed to run condition {condition}: {failed_jobs}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error running control: {str(e)}")
            return False

class IterativeExperiment(BaseExperiment):
    """Base class for iterative masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None
    ):
        super().__init__("iterative_masking", protein_config, slurm_config, working_dir, dry_run, schema_path)
        
        if not protein_config.iterative_masking.enabled:
            raise ExperimentError("Iterative masking is not enabled in configuration")
    
    def validate(self) -> None:
        """Validate iterative masking configuration"""
        if not self.protein_config.iterative_masking.mutations:
            raise ExperimentError("No mutations specified for iterative masking")
    
    def setup(self) -> None:
        """Set up iterative masking experiment"""
        # Create a controls subdirectory specifically for iterative experiments
        self.controls_dir = self.working_dir / "controls"
        if not self.dry_run:
            self.controls_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"[DRY RUN] Would create directory: {self.controls_dir}")

        # Add vanilla control
        self.controls.append(Control(
            "vanilla",
            self.protein_config,
            self.slurm_config,
            [Condition(mask=False, mutate=False)],
            working_dir=self.controls_dir,
            dry_run=self.dry_run,
            schema_path=self.schema_path
        ))
        
        # Add mutation controls
        for mutation_set in self.protein_config.iterative_masking.mutations:
            self.controls.append(Control(
                f"mutation_{'_'.join(mutation_set)}",
                self.protein_config,
                self.slurm_config,
                [Condition(mask=False, mutate=True)],
                working_dir=self.controls_dir,
                dry_run=self.dry_run,
                schema_path=self.schema_path
            ))
    
    def submit(self) -> bool:
        """Submit iterative masking experiment jobs to SLURM"""
        self.validate()
        self.setup()
        
        # Run controls first
        for control in self.controls:
            if not control.run():
                return False
        
        # Submit main experiment
        config = self._create_config()
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success

    def _create_config(self) -> ExperimentConfig:
        """Create configuration for iterative masking experiment"""
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": f"{self.name}",
            "parent_path": str(self.working_dir),
            "masking_strategy": MaskingStrategy.ITERATIVE_SINGLE,
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": "mutate_and_mask",
            "mutations": self.protein_config.iterative_masking.mutations
        }
        
        return ExperimentConfig(**config)

class AprioriExperiment(BaseExperiment):
    """Implementation of a priori masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None
    ):
        super().__init__("apriori_masking", protein_config, slurm_config, working_dir, dry_run, schema_path)
        
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
    
    def setup(self) -> None:
        """No separate setup needed as conditions are handled directly in submit"""
        pass
    
    def _create_config(self, experiment: AprioriExperiment, condition: Condition) -> ExperimentConfig:
        """Create configuration for a specific a priori experiment condition"""
        return ExperimentConfig(
            sequence=self.protein_config.sequence,
            jobname_prefix=f"{self.name}_{experiment.name}",
            parent_path=str(self.working_dir / experiment.name),
            masking_strategy=MaskingStrategy.MASK_POSITIONS if condition.mask else MaskingStrategy.NONE,
            positions=experiment.positions if condition.mask else [],
            mutations=experiment.mutations if condition.mutate else [],
            num_recycles=self.defaults.get('num_recycles', 2),
            num_seeds=self.defaults.get('num_seeds', 2),
            setup_path=str(self.slurm_config.setup_path),
            pipeline_type="mutate_and_mask" if condition.mutate and condition.mask else 
                         "mask_only" if condition.mask else
                         "mutate" if condition.mutate else
                         "default"
        )
    
    def submit(self) -> bool:
        """Submit a priori masking experiment jobs to SLURM
        
        This method validates the experiment configuration and submits jobs for each
        experiment with their associated conditions. For each experiment, it:
        1. Creates the experiment configuration for each condition
        2. Sets up a SLURM job manager
        3. Submits the jobs through the job manager
        
        Returns:
            bool: True if all jobs were submitted successfully, False otherwise
        """
        self.validate()
        
        # Submit each experiment with its conditions
        success = True
        for experiment in self.experiments:
            # Create a job for each condition
            for condition in experiment.conditions:
                config = self._create_config(experiment, condition)
                
                # Create descriptive name for this condition
                condition_name = f"{experiment.name}_{'masked' if condition.mask else 'unmasked'}_{'mutated' if condition.mutate else 'unmutated'}"
                working_dir = self.working_dir / experiment.name / condition_name
                
                job_manager = SlurmJobManager(
                    experiment_config=config,
                    slurm_config=self.slurm_config,
                    working_dir=str(working_dir)
                )
                
                exp_success, failed_jobs = job_manager.run_experiment()
                if not exp_success:
                    logger.error(f"Failed to submit experiment {condition_name}")
                    success = False
        
        return success

    def _validate_mutation(self, mutation: str) -> bool:
        """Validate a single mutation against the protein sequence"""
        if len(mutation) < 3:
            return False
            
        orig_aa = mutation[0]
        new_aa = mutation[-1]
        try:
            pos = int(mutation[1:-1])
        except ValueError:
            return False
        
        if pos < 1 or pos > len(self.protein_config.sequence):
            return False
            
        if self.protein_config.sequence[pos-1] != orig_aa:
            return False
            
        return True

class FrustraExperiment(BaseExperiment):
    """Implementation of Frustra-guided masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None
    ):
        super().__init__("frustra_masking", protein_config, slurm_config, working_dir, dry_run, schema_path)
        
        if not protein_config.frustra_masking.enabled:
            raise ExperimentError("Frustra masking is not enabled in configuration")
    
    def validate(self) -> None:
        """Validate Frustra masking configuration"""
        if self.protein_config.frustra_masking.top_positions < 1:
            raise ExperimentError("Invalid number of top positions for Frustra analysis")
    
    def setup(self) -> None:
        """Set up Frustra masking experiment"""
        # Create a controls subdirectory specifically for Frustra
        self.controls_dir = self.working_dir / "controls"
        if not self.dry_run:
            self.controls_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"[DRY RUN] Would create directory: {self.controls_dir}")

        # Add vanilla control
        self.controls.append(Control(
            "vanilla",
            self.protein_config,
            self.slurm_config,
            [Condition(mask=False, mutate=False)],
            working_dir=self.controls_dir,
            dry_run=self.dry_run,
            schema_path=self.schema_path
        ))
    
    def _create_config(self) -> ExperimentConfig:
        """Create configuration for Frustra masking experiment"""
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": f"{self.name}",
            "parent_path": str(self.working_dir),
            "masking_strategy": MaskingStrategy.MASK_POSITIONS,
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": "frustra",
            "top_positions": self.protein_config.frustra_masking.top_positions
        }
        
        return ExperimentConfig(**config)
    
    def submit(self) -> bool:
        """Submit Frustra masking experiment jobs to SLURM"""
        self.validate()
        self.setup()
        
        # Run controls first
        for control in self.controls:
            if not control.run():
                return False
        
        # Submit main experiment
        config = self._create_config()
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success