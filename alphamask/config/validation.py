from pathlib import Path
import json
import logging
from typing import Dict, Any, Tuple, Optional
from jsonschema import validate, ValidationError as JsonSchemaError

from ..experiments.types import ValidationError

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Handles configuration validation using JSON schema"""
    
    def __init__(self, schema_path: Optional[str] = None):
        if schema_path is None:
            schema_path = str(Path(__file__).parent / "config_schema.json")
        
        try:
            with open(schema_path) as f:
                self.schema = json.load(f)
        except Exception as e:
            raise ValidationError(f"Failed to load schema from {schema_path}: {str(e)}")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate configuration against schema
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            validate(instance=config, schema=self.schema)
            return self._validate_additional_constraints(config)
        except JsonSchemaError as e:
            return False, f"Schema validation failed: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_additional_constraints(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate additional constraints not covered by JSON schema
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Validate each protein configuration
            for protein_id, protein_config in config["proteins"].items():
                # Validate sequence length
                if len(protein_config["sequence"]) == 0:
                    return False, f"Empty sequence for protein {protein_id}"
                
                # Validate iterative masking
                if protein_config["iterative_masking"]["enabled"]:
                    result = self._validate_iterative_masking(protein_id, protein_config)
                    if not result[0]:
                        return result
                
                # Validate a priori masking
                if protein_config["apriori_masking"]["enabled"]:
                    result = self._validate_apriori_masking(protein_id, protein_config)
                    if not result[0]:
                        return result
                
                # Validate frustra masking
                if protein_config["frustra_masking"]["enabled"]:
                    result = self._validate_frustra_masking(protein_id, protein_config)
                    if not result[0]:
                        return result
            
            return True, "Configuration is valid"
            
        except Exception as e:
            return False, f"Additional validation failed: {str(e)}"
    
    def _validate_iterative_masking(
        self, 
        protein_id: str, 
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate iterative masking configuration"""
        try:
            if not config["iterative_masking"].get("mutations"):
                return False, f"No mutations specified for iterative masking in {protein_id}"
            
            sequence = config["sequence"]
            for mutation_set in config["iterative_masking"]["mutations"]:
                for mutation in mutation_set:
                    if not self._validate_mutation(mutation, sequence):
                        return False, f"Invalid mutation {mutation} in {protein_id}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Iterative masking validation failed: {str(e)}"
    
    def _validate_apriori_masking(
        self, 
        protein_id: str, 
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate a priori masking configuration"""
        try:
            if not config["apriori_masking"].get("experiments"):
                return False, f"No experiments specified for a priori masking in {protein_id}"
            
            sequence = config["sequence"]
            for exp in config["apriori_masking"]["experiments"]:
                # Validate positions
                for pos in exp["positions"]:
                    if pos < 1 or pos > len(sequence):
                        return False, f"Invalid position {pos} in experiment {exp['name']}"
                
                # Validate mutations
                for mutation in exp["mutations"]:
                    if not self._validate_mutation(mutation, sequence):
                        return False, f"Invalid mutation {mutation} in experiment {exp['name']}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"A priori masking validation failed: {str(e)}"
    
    def _validate_frustra_masking(
        self, 
        protein_id: str, 
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate frustra masking configuration"""
        try:
            top_positions = config["frustra_masking"]["top_positions"]
            if top_positions < 1:
                return False, f"Invalid top_positions value in {protein_id}"
            
            if top_positions > len(config["sequence"]):
                return False, f"top_positions exceeds sequence length in {protein_id}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Frustra masking validation failed: {str(e)}"
    
    def _validate_mutation(self, mutation: str, sequence: str) -> bool:
        """Validate a single mutation against a sequence"""
        try:
            if len(mutation) < 3:
                return False
            
            orig_aa = mutation[0]
            new_aa = mutation[-1]
            pos = int(mutation[1:-1])
            
            if pos < 1 or pos > len(sequence):
                return False
            
            if sequence[pos-1] != orig_aa:
                return False
            
            return True
            
        except Exception:
            return False 