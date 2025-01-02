from typing import Optional, Dict, Type
from pathlib import Path
import logging
from types import SimpleNamespace

from .base import (
    BaseExperiment,
    IterativeExperiment,
    AprioriExperiment,
    FrustraExperiment
)
from .types import ValidationError
from .config import process_configuration
from ..utils.slurm import SlurmJobConfig

from colabdesign.af.contrib import predict

from ..core.model import CompressionConfig

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
        base_dir: Optional[Path] = None,
        compression_config: Optional['CompressionConfig'] = None
    ):
        # Convert config_path to absolute path
        config_path = Path(config_path).resolve()
        self.config = process_configuration(str(config_path))
        logger.debug(f"Loaded configuration: {self.config}")

        if slurm_config is None:
            slurm_config = SlurmJobConfig()

        # Use provided schema path from slurm_config if available
        if slurm_config.schema_path:
            self.config.schema_path = slurm_config.schema_path
        elif not hasattr(self.config, 'schema_path'):
            # Use default schema path if none provided
            default_schema = Path(__file__).parent.parent / "config" / "schema_validation.json"
            self.config.schema_path = str(default_schema.resolve())
            logger.info(f"Using default schema path: {self.config.schema_path}")

        # Update slurm config with schema path
        slurm_config.schema_path = self.config.schema_path
        self.slurm_config = slurm_config
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()
        self.compression_config = compression_config
        
        # Add compression settings to global config if provided
        if compression_config:
            if not hasattr(self.config, 'global_settings'):
                self.config.global_settings = SimpleNamespace()
            
            # Create compression settings namespace
            compression_settings = {
                'format': compression_config.compress_format,
                'level': compression_config.compression_level,
                'store_uncompressed': compression_config.store_uncompressed_pdbs,
                'store_best_pdb': compression_config.store_best_pdb
            }
            
            # Set compression settings as namespace
            self.config.global_settings.compression = SimpleNamespace(**compression_settings)
        
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
        
        # Get hash for the sequence
        seq_hash = predict.get_hash(self.config.proteins[protein_id].sequence)[:5]
        
        protein_config = self.config.proteins[protein_id]
        experiments = []
        working_dir = self.base_dir / f"{protein_id}_{seq_hash}"
        schema_path = Path(self.config.schema_path)
        
        if not schema_path.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        
        # Create experiments with compression config
        experiment_kwargs = {
            'protein_config': protein_config,
            'slurm_config': self.slurm_config,
            'schema_path': schema_path,
            'compression_config': self.compression_config
        }
        
        
        if hasattr(protein_config, 'apriori_masking') and protein_config.apriori_masking.enabled:
            experiments.append(
                AprioriExperiment(
                    working_dir=working_dir / "apriori",
                    **experiment_kwargs
                )
            )
        
        # Check each experiment type's configuration properly
        if hasattr(protein_config, 'iterative_masking') and protein_config.iterative_masking.enabled:
            experiments.append(
                IterativeExperiment(
                    working_dir=working_dir / "iterative",
                    **experiment_kwargs
                )
            )
        

        
        if hasattr(protein_config, 'frustra_masking') and protein_config.frustra_masking.enabled:
            experiments.append(
                FrustraExperiment(
                    working_dir=working_dir / "frustra",
                    **experiment_kwargs
                )
            )
        
        return experiments
    
    def run_protein_experiments(self, protein_id: str) -> bool:
        """Submit all enabled experiments for a protein to SLURM"""
        try:
            logger.debug(f"Creating experiments for protein {protein_id}")
            experiments = self.create_experiments(protein_id)
            
            if not experiments:
                logger.warning(f"No experiments enabled for protein {protein_id}")
                return True
            
            success = True
            for experiment in experiments:
                logger.info(f"Submitting {experiment.name} for {protein_id}")
                try:
                    if not experiment.submit():
                        logger.error(f"Failed to submit {experiment.name} for {protein_id}")
                        success = False
                except Exception as e:
                    logger.error(f"Error submitting experiment {experiment.name}: {str(e)}")
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error submitting experiments for {protein_id}: {str(e)}")
            return False
    
    def run_all_experiments(self) -> bool:
        """Run experiments for all configured proteins, starting with shorter sequences"""
        overall_success = True
        
        try:
            # Create a list of (protein_id, length) tuples
            protein_lengths = [
                (protein_id, len(self.config.proteins[protein_id].sequence))
                for protein_id in self.config.proteins
            ]
            
            # Sort by sequence length
            protein_lengths.sort(key=lambda x: x[1])
            
            # Initialize partition manager
            from ..utils.partition import PartitionManager
            partition_manager = PartitionManager()
            
            # Log initial partition assignment plan
            logger.info("Initial partition assignment plan:")
            logger.info(partition_manager.get_usage_summary())
            
            # Run experiments in order of increasing length
            for protein_id, length in protein_lengths:
                # Get next partition and GPU type
                partition, gpu_type = partition_manager.get_partition_for_sequence(self.config.proteins[protein_id].sequence)
                
                # Update slurm config for this protein
                self.slurm_config.partition = partition
                self.slurm_config.gpu_type = gpu_type
                
                logger.info(f"Starting experiments for protein {protein_id} (length: {length}) on {partition}/{gpu_type}")
                if not self.run_protein_experiments(protein_id):
                    overall_success = False
                
                # Log updated partition usage after each protein
                logger.info(f"After processing {protein_id}:")
                logger.info(partition_manager.get_usage_summary())
            
            return overall_success
        except Exception as e:
            logger.error(f"Error in run_all_experiments: {str(e)}")
            return False

def run_experiments(
    config_path: str,
    slurm_config: Optional[SlurmJobConfig] = None,
    base_dir: Optional[Path] = None,
    protein_ids: Optional[list[str]] = None,
    compression_config: Optional['CompressionConfig'] = None
) -> bool:
    """
    Main entry point for running experiments.
    
    Args:
        config_path: Path to configuration file
        slurm_config: Optional SLURM configuration
        base_dir: Optional base directory for experiments
        protein_ids: Optional list of specific proteins to run
        compression_config: Optional compression configuration
    
    Returns:
        bool: True if all experiments succeeded, False otherwise
    """
    try:
        runner = ExperimentRunner(
            config_path=config_path, 
            slurm_config=slurm_config, 
            base_dir=base_dir,
            compression_config=compression_config
        )
        
        if protein_ids:
            # Sort proteins by sequence length
            protein_lengths = [
                (protein_id, len(runner.config.proteins[protein_id].sequence))
                for protein_id in protein_ids
            ]
            protein_lengths.sort(key=lambda x: x[1])
            
            # Initialize partition manager
            from alphamask.utils.partition import PartitionManager
            partition_manager = PartitionManager()
            
            # Log partition details at debug level
            logger.debug("Initial partition assignment plan:")
            logger.debug(partition_manager.get_usage_summary())
            
            # Run experiments
            success = True
            for protein_id, length in protein_lengths:
                partition, gpu_type = partition_manager.get_partition_for_sequence(runner.config.proteins[protein_id].sequence)
                runner.slurm_config.partition = partition
                runner.slurm_config.gpu_type = gpu_type
                
                logger.info(f"Processing {protein_id} (length: {length}) on {partition}/{gpu_type}")
                if not runner.run_protein_experiments(protein_id):
                    success = False
                
                # Log partition updates at debug level
                logger.debug(f"After processing {protein_id}:")
                logger.debug(partition_manager.get_usage_summary())
            
            return success
        else:
            # Run all experiments
            return runner.run_all_experiments()
            
    except Exception as e:
        logger.error(f"Error in experiment runner: {str(e)}")
        return False 