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
    mutations: List[List[str]]  # Sets of mutations to test
    mask_token: str = "X"       # Token used for masking

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
        """Validate schema path after initialization"""
        if not self.schema_path:
            raise ValidationError("Schema path not specified")
        schema_path = Path(self.schema_path)
        if not schema_path.exists():
            raise ValidationError(f"Schema file not found: {schema_path}")
        if not schema_path.is_file():
            raise ValidationError(f"Schema path is not a file: {schema_path}")