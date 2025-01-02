from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Condition:
    """Represents a single experimental condition in protein analysis"""
    mask: bool    # Whether to mask the position
    mutate: bool  # Whether to apply mutation

@dataclass
class AprioriExperiment:
    """
    Represents a complete experiment with known mutations.
    Each experiment consists of positions to study, mutations to apply,
    and a set of controlled conditions.
    """
    name: str                 # Experiment identifier (e.g., "Abdullah_et_al_2023_T150A")
    positions: List[int]      # Positions in sequence to analyze
    mutations: List[str]      # Mutations to apply (format: "A123B")
    conditions: List[Condition]  # Must include all 4 combinations

@dataclass
class IterativeMasking:
    """
    Systematic analysis where each position is masked one at a time.
    Mutations are tested against this systematic masking.
    """
    enabled: bool
    mutations: List[List[str]]  # Sets of mutations to test (e.g., [["I89S"], ["I89N"], ["I89S", "L90A"]])
    mask_token: str = "X"       # Token used for masking
    use_wt_msa: bool = True     # Whether to use WT MSA for all predictions
    msa_reuse: bool = True      # Whether to reuse MSA across jobs
    
    def __post_init__(self):
        """Validate mutation format and settings after initialization"""
        if self.enabled and self.mutations:
            self._validate_mutations()
    
    def _validate_mutations(self):
        """Validate mutation format and compatibility"""
        seen_positions = set()
        for mutation_set in self.mutations:
            positions = []
            for mutation in mutation_set:
                # Check mutation format (e.g., "I89S")
                if len(mutation) < 3:
                    raise ValidationError(f"Invalid mutation format: {mutation}")
                try:
                    pos = int(mutation[1:-1])
                    positions.append(pos)
                except ValueError:
                    raise ValidationError(f"Invalid position in mutation: {mutation}")
                
                # Check for conflicting mutations at same position in a set
                if pos in seen_positions:
                    raise ValidationError(f"Conflicting mutations at position {pos} in set {mutation_set}")
                seen_positions.add(pos)
            
            # Sort positions for consistent handling
            positions.sort()
            
            # Store validated positions for later use
            setattr(self, f"positions_{mutation_set[0]}", positions)

@dataclass
class AprioriMasking:
    """
    Analysis based on known mutations from literature or hypothesis.
    Provides controlled experiments for specific positions.
    """
    enabled: bool
    experiments: List[AprioriExperiment]

@dataclass
class FrustraMasking:
    """
    Automated analysis based on protein frustration patterns.
    Identifies positions and mutations without manual specification.
    """
    enabled: bool
    top_positions: int  # Number of positions to analyze

@dataclass
class LoggingConfig:
    """Logging configuration for analysis tracking"""
    enabled: bool
    level: LogLevel
    file: str

@dataclass
class GlobalSettings:
    """Global parameters affecting all analyses"""
    mask_token: str
    output_dir: str
    logging: LoggingConfig
    msa_settings: Optional[Dict] = None  # Optional MSA-specific settings

@dataclass
class ProteinConfig:
    """
    Complete configuration for analyzing a single protein.
    Combines all three masking strategies with protein information.
    """
    sequence: str
    iterative_masking: IterativeMasking
    apriori_masking: AprioriMasking
    frustra_masking: FrustraMasking
    uniprot_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate protein configuration after initialization"""
        if not self.sequence:
            raise ValidationError("Protein sequence cannot be empty")
        
        # Ensure at least one masking strategy is enabled
        if not any([
            self.iterative_masking.enabled,
            self.apriori_masking.enabled,
            self.frustra_masking.enabled
        ]):
            raise ValidationError("At least one masking strategy must be enabled")

@dataclass
class MaskingConfiguration:
    """
    Root configuration combining multiple proteins and global settings.
    Entry point for the entire analysis system.
    """
    proteins: Dict[str, ProteinConfig]
    global_settings: GlobalSettings
    schema_path: str

    def __post_init__(self):
        """Validate schema path and configuration after initialization"""
        if not self.schema_path:
            raise ValidationError("Schema path not specified")
        schema_path = Path(self.schema_path)
        if not schema_path.exists():
            raise ValidationError(f"Schema file not found: {schema_path}")
        if not schema_path.is_file():
            raise ValidationError(f"Schema path is not a file: {schema_path}")
        
        # Apply global settings to protein configs
        self._apply_global_settings()
    
    def _apply_global_settings(self):
        """Apply global settings to individual protein configurations"""
        for protein_config in self.proteins.values():
            # Apply mask token if not explicitly set
            if protein_config.iterative_masking.mask_token == "X":
                protein_config.iterative_masking.mask_token = self.global_settings.mask_token