from pathlib import Path
import yaml
from typing import Dict, Any, Tuple
import logging
from dataclasses import asdict
import os
import json
from types import SimpleNamespace

from .types import (
    ValidationError, LogLevel, Condition, AprioriExperiment,
    IterativeMasking, AprioriMasking, FrustraMasking,
    LoggingConfig, GlobalSettings, ProteinConfig, MaskingConfiguration
)

# Get logger for this module
logger = logging.getLogger("alphamask.experiments.config")

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its contents as a dictionary.
    Handles potential YAML parsing errors with informative messages.
    """
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(f"Error parsing YAML file: {e}")
    except FileNotFoundError:
        raise ValidationError(f"Configuration file not found: {file_path}")

def create_condition_from_dict(data: Dict[str, bool]) -> Condition:
    """Convert dictionary to Condition object with validation"""
    return Condition(
        mask=data['mask'],
        mutate=data['mutate']
    )

def create_experiment_from_dict(data: Dict[str, Any]) -> AprioriExperiment:
    """Convert dictionary to AprioriExperiment object with validation"""
    return AprioriExperiment(
        name=data['name'],
        positions=data['positions'],
        mutations=data['mutations'],
        conditions=[create_condition_from_dict(c) for c in data['conditions']]
    )

def create_protein_config_from_dict(data: Dict[str, Any]) -> ProteinConfig:
    """Convert dictionary to ProteinConfig object with validation"""
    return ProteinConfig(
        sequence=data['sequence'],
        iterative_masking=IterativeMasking(
            enabled=data['iterative_masking']['enabled'],
            mutations=data['iterative_masking']['mutations'],
            mask_token=data['iterative_masking'].get('mask_token', 'X')
        ),
        apriori_masking=AprioriMasking(
            enabled=data['apriori_masking']['enabled'],
            experiments=[
                create_experiment_from_dict(exp) 
                for exp in data['apriori_masking']['experiments']
            ]
        ),
        frustra_masking=FrustraMasking(
            enabled=data['frustra_masking']['enabled'],
            top_positions=data['frustra_masking']['top_positions']
        ),
        uniprot_id=data.get('uniprot_id')
    )

def create_config_from_dict(data: Dict[str, Any]) -> MaskingConfiguration:
    """Convert dictionary to MaskingConfiguration object with validation"""
    return MaskingConfiguration(
        schema_path=data['schema_path'],
        proteins={
            protein_id: create_protein_config_from_dict(protein_data)
            for protein_id, protein_data in data['proteins'].items()
        },
        global_settings=GlobalSettings(
            mask_token=data['global_settings']['mask_token'],
            output_dir=data['global_settings']['output_dir'],
            logging=LoggingConfig(
                enabled=data['global_settings']['logging']['enabled'],
                level=LogLevel[data['global_settings']['logging']['level']],
                file=data['global_settings']['logging']['file']
            )
        )
    )

def setup_logging(config: LoggingConfig) -> None:
    """Set up logging based on configuration"""
    logging.basicConfig(
        filename=config.file,
        level=config.level.value,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_configuration(config: MaskingConfiguration) -> Tuple[bool, str]:
    """
    Validate the complete configuration.
    Returns (is_valid, message) tuple.
    """
    try:
        # Validate protein configurations
        for protein_id, protein_config in config.proteins.items():
            # Basic validation
            if not protein_config.sequence:
                return False, f"Empty sequence for protein {protein_id}"
            
            # Validate iterative masking
            if protein_config.iterative_masking.enabled:
                if not protein_config.iterative_masking.mutations:
                    return False, f"No mutations specified for iterative masking in {protein_id}"
            
            # Validate apriori masking
            if protein_config.apriori_masking.enabled:
                if not protein_config.apriori_masking.experiments:
                    return False, f"No experiments specified for apriori masking in {protein_id}"
            
            # Validate frustra masking
            if protein_config.frustra_masking.enabled:
                if protein_config.frustra_masking.top_positions < 1:
                    return False, f"Invalid top_positions for frustra masking in {protein_id}"

        # Validate global settings
        if not config.global_settings.output_dir:
            return False, "Output directory not specified"
        
        return True, "Configuration is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_configuration(config_path: str) -> Any:
    """Process and validate configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add schema path if not present
        if 'schema_path' not in config:
            default_schema = Path(__file__).parent.parent / "config" / "schema_validation.json"
            config['schema_path'] = str(default_schema.resolve())
            logger.info(f"Using default schema path: {config['schema_path']}")
        
        # Convert to object for easier access
        config = SimpleNamespace(**config)
        
        # Ensure proteins is a dictionary
        if not hasattr(config, 'proteins') or not isinstance(config.proteins, dict):
            raise ValidationError("Configuration must contain a 'proteins' dictionary")
        
        # Convert each protein config to object
        for protein_id, protein_config in config.proteins.items():
            config.proteins[protein_id] = SimpleNamespace(**protein_config)
            
            # Convert masking configurations to objects
            if hasattr(config.proteins[protein_id], 'iterative_masking'):
                iterative = config.proteins[protein_id].iterative_masking
                if isinstance(iterative, dict):
                    config.proteins[protein_id].iterative_masking = SimpleNamespace(**iterative)
            
            if hasattr(config.proteins[protein_id], 'apriori_masking'):
                apriori = config.proteins[protein_id].apriori_masking
                if isinstance(apriori, dict):
                    # Convert experiments list
                    if 'experiments' in apriori:
                        apriori['experiments'] = [
                            SimpleNamespace(**exp) if isinstance(exp, dict) else exp
                            for exp in apriori['experiments']
                        ]
                        # Convert conditions in each experiment
                        for exp in apriori['experiments']:
                            if hasattr(exp, 'conditions'):
                                exp.conditions = [
                                    SimpleNamespace(**cond) if isinstance(cond, dict) else cond
                                    for cond in exp.conditions
                                ]
                    config.proteins[protein_id].apriori_masking = SimpleNamespace(**apriori)
            
            if hasattr(config.proteins[protein_id], 'frustra_masking'):
                frustra = config.proteins[protein_id].frustra_masking
                if isinstance(frustra, dict):
                    config.proteins[protein_id].frustra_masking = SimpleNamespace(**frustra)
        
        # Initialize global settings
        if not hasattr(config, 'global_settings'):
            config.global_settings = SimpleNamespace()
        elif isinstance(config.global_settings, dict):
            config.global_settings = SimpleNamespace(**config.global_settings)
            
            # Convert nested dictionaries to SimpleNamespace
            for key, value in vars(config.global_settings).items():
                if isinstance(value, dict):
                    setattr(config.global_settings, key, SimpleNamespace(**value))
        
        return config
        
    except Exception as e:
        logger.error(f"Error processing configuration: {str(e)}")
        raise 