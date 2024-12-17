import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError as JsonSchemaError

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def validate_config(config: Dict[str, Any], schema_path: str) -> None:
    """Validate configuration against JSON schema"""
    try:
        # Load schema
        with open(schema_path) as f:
            schema = json.load(f)
        
        # Validate against schema
        validate(instance=config, schema=schema)
        
    except JsonSchemaError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise

def get_config_dir() -> Path:
    """Get the directory containing configuration files"""
    return Path(__file__).parent.parent / "config"

def get_default_schema_path() -> Path:
    """Get the path to the default JSON schema file"""
    return get_config_dir() / "config_schema.json"

def get_example_config_path() -> Path:
    """Get the path to the example configuration file"""
    return get_config_dir() / "example_config.yaml" 