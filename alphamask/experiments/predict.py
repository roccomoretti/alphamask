"""
Module for running prediction pipelines in AlphaMask
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..utils.types import ExperimentConfig
from ..core.pipeline import DefaultPipeline, MaskingPipeline, MutatePipeline, MutateAndMaskingPipeline

logger = logging.getLogger(__name__)

def run_prediction_pipeline(
    config: Dict[str, Any],
    pipeline_type: str = "default"
) -> None:
    """Run the prediction pipeline with the given configuration"""
    logger.info(f"Starting prediction pipeline: {pipeline_type}")
    
    # Map 'vanilla' to 'default'
    if pipeline_type == "vanilla":
        pipeline_type = "default"
        
    pipeline_map = {
        "default": DefaultPipeline,  # Basic prediction without masking
        "masking": MaskingPipeline,  # Masking operations
        "mutate": MutatePipeline,  # Mutation operations
        "mutate_and_mask": MutateAndMaskingPipeline  # Combined mutation and masking
    }
    
    if pipeline_type not in pipeline_map:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Valid choices are: {list(pipeline_map.keys())}")
    
    pipeline_class = pipeline_map[pipeline_type]
    
    try:
        # Initialize and run pipeline
        pipeline = pipeline_class(params=config)
        result = pipeline.run()
        
        if not result.success:
            raise RuntimeError(f"Pipeline failed: {result.error}")
            
        logger.info("Prediction pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise