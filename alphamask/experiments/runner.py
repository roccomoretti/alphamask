from typing import Optional, Dict, Type
from pathlib import Path
import logging

from .base import (
    BaseExperiment,
    IterativeExperiment,
    AprioriExperiment,
    FrustraExperiment
)
from .types import ValidationError
from .config import process_configuration
from ..utils.slurm import SlurmJobConfig

# Get logger for this module
logger = logging.getLogger("alphamask.experiments.runner")

class ExperimentRunner:
    """
    Main class for running protein experiments.
    Handles experiment creation, validation, and execution.
    """
    
    def __init__(
        self,
        config_path: str,
        slurm_config: Optional[SlurmJobConfig] = None,
        base_dir: Optional[Path] = None
    ):
        # Convert config_path to absolute path
        config_path = Path(config_path).resolve()
        self.config = process_configuration(str(config_path))

        if slurm_config is None:
            slurm_config = SlurmJobConfig()
        # Ensure schema_path is absolute
        schema_path = Path(self.config.schema_path).resolve()
        slurm_config.schema_path = str(schema_path)
        self.slurm_config = slurm_config
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        
        # Map of experiment types to their implementations
        self.experiment_types: Dict[str, Type[BaseExperiment]] = {
            "iterative": IterativeExperiment,
            "apriori": AprioriExperiment,
            "frustra": FrustraExperiment
        }
    
    def create_experiments(self, protein_id: str) -> list[BaseExperiment]:
        """Create experiment instances for a protein"""
        if protein_id not in self.config.proteins:
            raise ValueError(f"No configuration found for protein {protein_id}")
        
        protein_config = self.config.proteins[protein_id]
        experiments = []
        working_dir = self.base_dir / protein_id
        schema_path = Path(self.config.schema_path)
        
        if not schema_path.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        
        # Create iterative masking experiment if enabled
        if protein_config.iterative_masking.enabled:
            experiments.append(
                IterativeExperiment(
                    protein_config,
                    self.slurm_config,
                    working_dir / "iterative",
                    schema_path=schema_path
                )
            )
        
        # Create a priori masking experiment if enabled
        if protein_config.apriori_masking.enabled:
            experiments.append(
                AprioriExperiment(
                    protein_config,
                    self.slurm_config,
                    working_dir / "apriori",
                    schema_path=schema_path
                )
            )
        
        # Create Frustra masking experiment if enabled
        if protein_config.frustra_masking.enabled:
            experiments.append(
                FrustraExperiment(
                    protein_config,
                    self.slurm_config,
                    working_dir / "frustra",
                    schema_path=schema_path
                )
            )
        
        return experiments
    
    def run_protein_experiments(self, protein_id: str) -> bool:
        """Submit all enabled experiments for a protein to SLURM"""
        try:
            # Validate schema path before creating experiments
            if not hasattr(self.config, 'schema_path') or not self.config.schema_path:
                raise ValueError("Schema path not found in configuration")
            
            schema_path = Path(self.config.schema_path)
            if not schema_path.exists():
                raise ValueError(f"Schema file not found: {schema_path}")
            if not schema_path.is_file():
                raise ValueError(f"Schema path is not a file: {schema_path}")

            experiments = self.create_experiments(protein_id)
            
            if not experiments:
                logger.warning(f"No experiments enabled for protein {protein_id}")
                return True
            
            success = True
            for experiment in experiments:
                logger.info(f"Submitting {experiment.name} for {protein_id}")
                if not experiment.submit():
                    logger.error(f"Failed to submit {experiment.name} for {protein_id}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error submitting experiments for {protein_id}: {str(e)}")
            return False
    
    def run_all_experiments(self) -> bool:
        """Run experiments for all configured proteins"""
        overall_success = True
        
        for protein_id in self.config.proteins:
            logger.info(f"Starting experiments for protein {protein_id}")
            if not self.run_protein_experiments(protein_id):
                overall_success = False
        
        return overall_success

def run_experiments(
    config_path: str,
    slurm_config: Optional[SlurmJobConfig] = None,
    base_dir: Optional[Path] = None,
    protein_ids: Optional[list[str]] = None
) -> bool:
    """
    Main entry point for running experiments.
    
    Args:
        config_path: Path to configuration file
        slurm_config: Optional SLURM configuration
        base_dir: Optional base directory for experiments
        protein_ids: Optional list of specific proteins to run
    
    Returns:
        bool: True if all experiments succeeded, False otherwise
    """
    try:
        runner = ExperimentRunner(config_path, slurm_config, base_dir)
        
        if protein_ids:
            # Run experiments for specific proteins
            success = True
            for protein_id in protein_ids:
                if not runner.run_protein_experiments(protein_id):
                    success = False
            return success
        else:
            # Run all experiments
            return runner.run_all_experiments()
            
    except Exception as e:
        logger.error(f"Error in experiment runner: {str(e)}")
        return False 