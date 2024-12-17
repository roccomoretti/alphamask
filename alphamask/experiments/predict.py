"""
Module for running prediction pipelines in AlphaMask
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.types import ExperimentConfig, MaskingStrategy
from .base import BaseExperiment
from .apriori import AprioriExperiment
from .frustra import FrustraExperiment
from .iterative import IterativeExperiment
from .types import ProteinConfig, IterativeMasking, AprioriMasking, FrustraMasking

logger = logging.getLogger(__name__)

def setup_experiment_directories(config: ExperimentConfig) -> Dict[str, Path]:
    """Set up standard directory structure for experiments"""
    parent_dir = Path(config.parent_path)
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Create standard directory structure
    dirs = {
        'input': parent_dir / "in",
        'output': parent_dir / "out",
        'msa': parent_dir / "in" / "msa",
        'pdb': parent_dir / "out" / "pdbs",
        'logs': parent_dir / "logs"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return dirs

def get_experiment_class(pipeline_type: str) -> type:
    """Get the appropriate experiment class based on pipeline type"""
    # Map 'vanilla' to 'default'
    if pipeline_type == "vanilla":
        pipeline_type = "default"
        
    pipeline_map = {
        "default": FrustraExperiment,  # Basic prediction without masking
        "masking": FrustraExperiment,  # Frustra-guided masking
        "mutate": IterativeExperiment,  # Mutation-based experiments
        "mutate_and_mask": IterativeExperiment,  # Combined mutation and masking
        "apriori": AprioriExperiment  # A priori masking experiments - should use AprioriExperiment
    }
    
    if pipeline_type not in pipeline_map:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Valid choices are: {list(pipeline_map.keys())}")
    
    return pipeline_map[pipeline_type]

def create_experiment_config(config: Dict[str, Any], pipeline_type: str) -> ExperimentConfig:
    """Create ExperimentConfig from dictionary config"""
    # Extract protein config if it exists
    if "proteins" in config:
        # Take the first protein's config
        protein_id = next(iter(config["proteins"]))
        protein_config = config["proteins"][protein_id]
        sequence = protein_config["sequence"]
    else:
        sequence = config["sequence"]
    
    # Create experiment config
    return ExperimentConfig(
        sequence=sequence,
        jobname_prefix=config.get("jobname_prefix", "prediction"),
        parent_path=config.get("parent_path", str(Path.cwd())),
        masking_strategy=MaskingStrategy(config.get("masking_strategy", "none")),
        positions=config.get("positions", []),
        mutations=config.get("mutations", []),
        num_recycles=config.get("num_recycles", 2),
        num_seeds=config.get("num_seeds", 2),
        setup_path=config.get("setup_path", str(Path.cwd())),
        pipeline_type=pipeline_type
    )

def create_protein_config(experiment_config: ExperimentConfig) -> ProteinConfig:
    """Convert ExperimentConfig to ProteinConfig"""
    # Create appropriate masking configurations based on pipeline type
    iterative_masking = IterativeMasking(
        enabled=experiment_config.pipeline_type in ["mutate", "mutate_and_mask"],
        mutations=[experiment_config.mutations] if isinstance(experiment_config.mutations, str) else experiment_config.mutations,
        mask_token="X"
    )
    
    apriori_masking = AprioriMasking(
        enabled=experiment_config.pipeline_type == "apriori",
        experiments=[]
    )
    
    frustra_masking = FrustraMasking(
        enabled=experiment_config.pipeline_type in ["default", "masking"],
        top_positions=len(experiment_config.positions) if experiment_config.positions else 10
    )
    
    return ProteinConfig(
        sequence=experiment_config.sequence,
        iterative_masking=iterative_masking,
        apriori_masking=apriori_masking,
        frustra_masking=frustra_masking
    )

def run_prediction_pipeline(
    config: Dict[str, Any],
    pipeline_type: str = "default"
) -> None:
    """Run the prediction pipeline with the given configuration"""
    logger.info(f"Starting prediction pipeline: {pipeline_type}")
    
    # Convert dictionary config to ExperimentConfig
    experiment_config = create_experiment_config(config, pipeline_type)
    
    # Set up directories
    dirs = setup_experiment_directories(experiment_config)
    
    # Convert ExperimentConfig to ProteinConfig
    protein_config = create_protein_config(experiment_config)
    
    # Get appropriate experiment class
    experiment_class = get_experiment_class(pipeline_type)
    
    # Initialize experiment with name parameter
    experiment = experiment_class(
        name=experiment_config.jobname_prefix,
        protein_config=protein_config,
        working_dir=dirs['output']
    )
    
    try:
        # Run the experiment
        experiment.run()
        logger.info("Prediction pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise