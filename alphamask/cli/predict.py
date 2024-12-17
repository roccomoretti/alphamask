#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import yaml
from pathlib import Path

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
from alphamask.experiments.predict import run_prediction_pipeline
from alphamask.utils.types import MaskingStrategy

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
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.yaml_file}")
        config = load_config(args.yaml_file)
        
        logger.info(f"Validating configuration against schema {args.json_schema}")
        validate_config(config, args.json_schema)
        
        # Run prediction pipeline
        logger.info(f"Running prediction pipeline: {args.pipeline}")
        run_prediction_pipeline(config, pipeline_type=args.pipeline)
        
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        if args.debug:
            logger.exception("Detailed error trace:")
        sys.exit(1)

if __name__ == "__main__":
    main() 