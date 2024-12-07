#!/usr/bin/env python3
import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

from alphamask.core.pipeline import (
    DefaultPipeline,
    MaskingPipeline,
    MutatePipeline,
    MutateAndMaskingPipeline
)

def setup_logging(log_dir: Path):
    """Set up logging configuration"""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "prediction.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError("Config file must be either JSON or YAML")

def validate_config(config: Dict[str, Any], schema_path: str) -> None:
    """Validate configuration against JSON schema"""
    import jsonschema
    
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run AlphaFold predictions")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    config_group.add_argument(
        "--yaml_file",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--schema",
        "--json_schema",  # Add alias for backward compatibility
        type=str,
        required=True,
        help="Path to JSON schema file"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["default", "masking", "mutate", "mutate_and_mask"],
        default="default",
        help="Pipeline type to use"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(Path("logs"))
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate configuration
        config_path = args.config if args.config else args.yaml_file
        config = load_config(config_path)
        validate_config(config, args.schema)
        
        # Select pipeline based on argument
        pipeline_classes = {
            "default": DefaultPipeline,
            "masking": MaskingPipeline,
            "mutate": MutatePipeline,
            "mutate_and_mask": MutateAndMaskingPipeline
        }
        
        PipelineClass = pipeline_classes[args.pipeline]
        logger.info(f"Using {PipelineClass.__name__}")
        
        # Initialize and run pipeline
        pipeline = PipelineClass(params=config)
        result = pipeline.run()
        
        logger.info(f"Prediction completed successfully: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 