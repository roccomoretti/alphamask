#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import yaml
from pathlib import Path
import logging

# Configure logging first
logger = logging.getLogger("alphamask.cli.commands")

def setup_imports():
    """Setup imports by adding the project root to sys.path"""
    # Get the absolute path to the project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Setup imports before importing project modules
setup_imports()

from alphamask.utils.config import load_config, validate_config
from alphamask.utils.logging import setup_logging
from alphamask.core.pipeline import DefaultPipeline, MaskingPipeline, MutatePipeline, MutateAndMaskingPipeline

def get_pipeline_class(pipeline_type: str) -> type:
    """Get the appropriate pipeline class based on pipeline type"""
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
    
    return pipeline_map[pipeline_type]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run protein predictions")
    parser.add_argument(
        "--yaml_file",
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--json_schema",
        required=True,
        help="Path to JSON schema for validation"
    )
    parser.add_argument(
        "--pipeline",
        choices=["default", "vanilla", "masking", "mutate", "mutate_and_mask"],
        default="default",
        help="Pipeline type ('vanilla' is an alias for 'default')"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable logging output"
    )
    args = parser.parse_args()
    
    # Map 'vanilla' to 'default'
    if args.pipeline == "vanilla":
        args.pipeline = "default"
    
    return args

def main():
    """Main entry point for prediction script"""
    args = parse_args()
    
    # Setup logging
    setup_logging(debug=args.debug, disable=args.quiet)
    logger = logging.getLogger(__name__)
    
    try:
        # Log environment information
        working_dir = os.getcwd()
        logger.info(f"Working directory: {working_dir}")
        logger.info(f"Config path: {os.path.abspath(args.yaml_file)}")
        logger.info(f"Script path: {os.path.abspath(__file__)}")
        logger.info(f"Schema path: {os.path.abspath(args.json_schema)}")
        
        # Log directory contents
        logger.info("Directory structure:")
        for root, dirs, files in os.walk(working_dir, topdown=True, followlinks=False):
            level = root.replace(working_dir, '').count(os.sep)
            indent = '  ' * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            for f in files:
                logger.info(f"{indent}  {f}")
        
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.yaml_file}")
        config = load_config(args.yaml_file)
        
        logger.info(f"Validating configuration against schema {args.json_schema}")
        validate_config(config, args.json_schema)
        
        # Get appropriate pipeline class
        pipeline_class = get_pipeline_class(args.pipeline)
        
        # Initialize and run pipeline
        logger.info(f"Running prediction pipeline: {args.pipeline}")
        
        # Log the config in human readable format with indentation
        logger.info(f"Config: {yaml.dump(config, default_flow_style=False, indent=4)}")
        
        pipeline = pipeline_class(params=config)
        result = pipeline.run()
        
        if not result.success:
            raise RuntimeError(f"Pipeline failed: {result.error}")
            
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        if args.debug:
            logger.exception("Detailed error trace:")
        sys.exit(1)

if __name__ == "__main__":
    main() 