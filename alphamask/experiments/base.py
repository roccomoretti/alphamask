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

logger = logging.getLogger(__name__)

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
        dry_run: bool = False
    ):
        self.name = name
        self.protein_config = protein_config
        self.slurm_config = slurm_config or SlurmJobConfig()
        self.working_dir = working_dir or Path.cwd() / "experiments" / name
        self.dry_run = dry_run
        
        if not self.dry_run:
            self.working_dir.mkdir(parents=True, exist_ok=True)
            
            # Create controls directory under this specific mode's directory
            self.controls_dir = self.working_dir / "controls"
            self.controls_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.controls_dir = self.working_dir / "controls"
            logger.info(f"[DRY RUN] Would create directory: {self.working_dir}")
            logger.info(f"[DRY RUN] Would create directory: {self.controls_dir}")
        
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
    def run(self) -> bool:
        """Run the experiment"""
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
        dry_run: bool = False
    ):
        self.name = name
        self.protein_config = protein_config
        self.slurm_config = slurm_config
        self.conditions = conditions
        self.dry_run = dry_run
        
        # Set working directory, ensuring it's under the experiment's controls directory
        if working_dir is None:
            raise ValueError("working_dir must be specified for Control")
            
        # Create the control directory under the specific mode's directory
        self.working_dir = Path(working_dir) / name
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
        success = True
        
        for condition in self.conditions:
            try:
                config = self.create_experiment_config(condition)
                
                # Create job manager with proper working directory
                job_manager = SlurmJobManager(
                    experiment_config=config,
                    slurm_config=self.slurm_config,
                    working_dir=str(self.working_dir)
                )
                
                if not self.dry_run:
                    try:
                        # Run the experiment
                        condition_success, failed_jobs = job_manager.run_experiment()
                        if not condition_success:
                            logger.error(f"Failed to run condition {condition} for control {self.name}")
                            if failed_jobs:
                                for job in failed_jobs:
                                    logger.error(f"Failed job details: {job}")
                            success = False
                    except Exception as e:
                        logger.error(f"Error running job for condition {condition} in control {self.name}: {str(e)}")
                        success = False
                else:
                    # Log what would happen in dry run mode
                    logger.info(f"[DRY RUN] Would submit job for condition: {condition}")
                    logger.info(f"[DRY RUN] - Working directory: {self.working_dir}")
                    logger.info(f"[DRY RUN] - Job prefix: {config.jobname_prefix}")
                    if condition.mask:
                        logger.info(f"[DRY RUN] - Would mask positions: {config.positions}")
                    if condition.mutate:
                        logger.info(f"[DRY RUN] - Would apply mutations: {config.mutations}")
                
            except Exception as e:
                logger.error(f"Error {'simulating' if self.dry_run else 'running'} condition {condition} for control {self.name}: {str(e)}")
                logger.exception("Full traceback:")
                success = False
                continue  # Continue with next condition even if this one fails
        
        return success

class IterativeExperiment(BaseExperiment):
    """Base class for iterative masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False
    ):
        super().__init__("iterative_masking", protein_config, slurm_config, working_dir, dry_run)
        
        if not protein_config.iterative_masking.enabled:
            raise ExperimentError("Iterative masking is not enabled in configuration")
    
    def validate(self) -> None:
        """Validate iterative masking configuration"""
        if not self.protein_config.iterative_masking.mutations:
            raise ExperimentError("No mutations specified for iterative masking")
    
    def setup(self) -> None:
        """Set up iterative masking experiment"""
        # Add vanilla control
        self.controls.append(Control(
            "vanilla",
            self.protein_config,
            self.slurm_config,
            [Condition(mask=False, mutate=False)],
            working_dir=self.controls_dir,
            dry_run=self.dry_run
        ))
        
        # Add mutation controls
        for mutation_set in self.protein_config.iterative_masking.mutations:
            self.controls.append(Control(
                f"mutation_{'_'.join(mutation_set)}",
                self.protein_config,
                self.slurm_config,
                [Condition(mask=False, mutate=True)],
                working_dir=self.controls_dir,
                dry_run=self.dry_run
            ))
    
    def run(self) -> bool:
        """Run iterative masking experiment"""
        self.validate()
        self.setup()
        
        # Run controls first
        for control in self.controls:
            if not control.run():
                return False
        
        # Run main experiment
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
        dry_run: bool = False
    ):
        super().__init__("apriori_masking", protein_config, slurm_config, working_dir, dry_run)
        
        if not protein_config.apriori_masking.enabled:
            raise ExperimentError("A priori masking is not enabled in configuration")
    
    def validate(self) -> None:
        """Validate a priori masking configuration"""
        if not self.protein_config.apriori_masking.experiments:
            raise ExperimentError("No experiments specified for a priori masking")
        
        for experiment in self.protein_config.apriori_masking.experiments:
            if len(experiment.conditions) != 4:
                raise ExperimentError(
                    f"Experiment {experiment.name} must have exactly 4 conditions"
                )
    
    def setup(self) -> None:
        """Set up a priori masking experiments"""
        # Add vanilla control
        self.controls.append(Control(
            "vanilla",
            self.protein_config,
            self.slurm_config,
            [Condition(mask=False, mutate=False)],
            working_dir=self.controls_dir,
            dry_run=self.dry_run
        ))
        
        # Add experiment-specific controls
        for experiment in self.protein_config.apriori_masking.experiments:
            self.controls.append(Control(
                f"experiment_{experiment.name}",
                self.protein_config,
                self.slurm_config,
                experiment.conditions,
                working_dir=self.controls_dir,
                dry_run=self.dry_run
            ))
    
    def _create_config(self, experiment: AprioriExperiment) -> ExperimentConfig:
        """Create configuration for a specific a priori experiment"""
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": f"{self.name}_{experiment.name}",
            "parent_path": str(self.working_dir / experiment.name),
            "masking_strategy": MaskingStrategy.MASK_POSITIONS,
            "positions": experiment.positions,
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": "mutate_and_mask",
            "mutations": experiment.mutations
        }
        
        return ExperimentConfig(**config)
    
    def run(self) -> bool:
        """Run a priori masking experiments"""
        self.validate()
        self.setup()
        
        # Run controls first
        for control in self.controls:
            if not control.run():
                return False
        
        # Run each experiment
        success = True
        for experiment in self.protein_config.apriori_masking.experiments:
            config = self._create_config(experiment)
            
            job_manager = SlurmJobManager(
                experiment_config=config,
                slurm_config=self.slurm_config,
                working_dir=str(self.working_dir / experiment.name)
            )
            
            exp_success, failed_jobs = job_manager.run_experiment()
            if not exp_success:
                logger.error(f"Failed to run experiment {experiment.name}")
                success = False
        
        return success

class FrustraExperiment(BaseExperiment):
    """Implementation of Frustra-guided masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False
    ):
        super().__init__("frustra_masking", protein_config, slurm_config, working_dir, dry_run)
        
        if not protein_config.frustra_masking.enabled:
            raise ExperimentError("Frustra masking is not enabled in configuration")
    
    def validate(self) -> None:
        """Validate Frustra masking configuration"""
        if self.protein_config.frustra_masking.top_positions < 1:
            raise ExperimentError("Invalid number of top positions for Frustra analysis")
    
    def setup(self) -> None:
        """Set up Frustra masking experiment"""
        # Add vanilla control
        self.controls.append(Control(
            "vanilla",
            self.protein_config,
            self.slurm_config,
            [Condition(mask=False, mutate=False)],
            working_dir=self.controls_dir,
            dry_run=self.dry_run
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
    
    def run(self) -> bool:
        """Run Frustra masking experiment"""
        self.validate()
        self.setup()
        
        # Run controls first
        for control in self.controls:
            if not control.run():
                return False
        
        # Run main experiment
        config = self._create_config()
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success