from pathlib import Path
import yaml
from typing import Dict, Any, Tuple
import logging
from dataclasses import asdict

from .types import (
    ValidationError, LogLevel, Condition, AprioriExperiment,
    IterativeMasking, AprioriMasking, FrustraMasking,
    LoggingConfig, GlobalSettings, ProteinConfig, MaskingConfiguration
)

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

def process_configuration(file_path: str) -> MaskingConfiguration:
    """
    Main function to load, validate, and process a masking configuration file.
    Returns a validated MaskingConfiguration object ready for use.
    """
    # Load YAML file
    yaml_data = load_yaml_config(file_path)
    
    # Create configuration object
    try:
        config = create_config_from_dict(yaml_data)
    except Exception as e:
        raise ValidationError(f"Error creating configuration objects: {e}")
    
    # Validate configuration
    is_valid, message = validate_configuration(config)
    if not is_valid:
        raise ValidationError(f"Configuration validation failed: {message}")
    
    # Setup logging if enabled
    if config.global_settings.logging.enabled:
        setup_logging(config.global_settings.logging)
        logging.info("Configuration loaded and validated successfully")
    
    return config 