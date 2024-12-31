from pathlib import Path
import shutil
import logging
from typing import Optional
import os

from .config import process_configuration
from .types import ValidationError, MaskingConfiguration
from colabdesign.af.contrib import predict

# Get logger for this module
logger = logging.getLogger("alphamask.experiments.setup")

class SetupError(Exception):
    """Custom exception for setup-related errors"""
    pass

class ExperimentSetup:
    """Handles experiment directory setup and resource preparation"""
    
    def __init__(
        self,
        config_path: str,
        setup_path: str,
        base_dir: Optional[Path] = None,
        force: bool = False
    ):
        self.config_path = Path(config_path).resolve()
        self.setup_path = Path(setup_path).resolve()
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd()
        self.force = force
        
        # Load and process configuration
        self.config = process_configuration(str(self.config_path))
        
        # Standard directory structure
        self.required_dirs = [
            "config",
            "logs",
            "schema",
            "scripts"
        ]
    
    def validate_setup(self) -> None:
        """Validate setup requirements"""
        # Validate configuration file
        if not self.config_path.exists():
            raise SetupError(f"Configuration file not found: {self.config_path}")
        
        # Validate base directory
        if self.base_dir.exists() and not self.force:
            existing_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            if existing_dirs:
                raise SetupError(
                    f"{self.base_dir} already contains directories. "
                    "Use --force to overwrite existing setup."
                )
        
        # Validate setup base directory
        if not self.setup_path.exists():
            raise SetupError(f"Setup base directory not found: {self.setup_path}")
    
    def create_directory_structure(self) -> None:
        """Create the base directory structure"""
        logger.info(f"Setting up experiment directories in {self.base_dir}")
        
        try:
            # Create standard directories
            for dir_path in self.required_dirs:
                full_path = self.base_dir / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {full_path}")
            
            # Create protein-specific directories
            for protein_id in self.config.proteins:
                # Get hash for the sequence
                seq_hash = predict.get_hash(self.config.proteins[protein_id].sequence)[:5]
                # Add the hash to the protein_id   
                protein_dir = self.base_dir / f"{protein_id}_{seq_hash}"
                protein_config = self.config.proteins[protein_id]
                
                # Create experiment type directories if enabled
                if protein_config.iterative_masking.enabled:
                    (protein_dir / "iterative").mkdir(parents=True, exist_ok=True)
                    (protein_dir / "iterative" / "controls").mkdir(parents=True, exist_ok=True)
                
                if protein_config.frustra_masking.enabled:
                    (protein_dir / "frustra").mkdir(parents=True, exist_ok=True)
                    (protein_dir / "frustra" / "controls").mkdir(parents=True, exist_ok=True)
                
                # Create common directories for each experiment type
                for exp_type in ["iterative", "apriori", "frustra"]:
                    exp_dir = protein_dir / exp_type
                    if exp_dir.exists():
                        for subdir in ["configs", "scripts", "logs", "in/msa", "out/pdbs", "schema"]:
                            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Created directories for protein: {protein_id}")
                
        except Exception as e:
            raise SetupError(f"Failed to create directory structure: {str(e)}")
    
    def setup_shared_resources(self) -> None:
        """Set up shared resources in setup directory"""
        try:
            logger.info(f"Setting up shared resources in {self.setup_path}")
            self.setup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy configuration templates
            pkg_config = Path(__file__).parent.parent / "config"
            config_files = {
                "test.yaml": "config/test.yaml",
                "schema_validation.json": "schema/schema_validation.json"
            }
            
            for src_file, dst_path in config_files.items():
                src_path = pkg_config / src_file
                if src_path.exists():
                    dst_full_path = self.base_dir / dst_path
                    dst_full_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_full_path)
                    logger.info(f"Copied {src_file} to {dst_full_path}")
                else:
                    logger.warning(f"Could not find {src_file} in package config")
                    
        except Exception as e:
            raise SetupError(f"Failed to setup shared resources: {str(e)}")
    
    def setup(self) -> None:
        """Run complete setup process"""
        try:
            self.validate_setup()
            self.create_directory_structure()
            self.setup_shared_resources()
            
            logger.info("\nExperiment directories setup complete!")
            logger.info("\nNext steps:")
            logger.info("1. Review and modify config/proteins.yaml as needed")
            logger.info("2. Run experiments using alphamask run")
            
        except (SetupError, ValidationError) as e:
            logger.error(f"Setup failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during setup: {str(e)}")
            raise 