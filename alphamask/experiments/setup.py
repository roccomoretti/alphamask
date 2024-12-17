from pathlib import Path
import shutil
import logging
from typing import Optional

from .config import process_configuration
from .types import ValidationError

logger = logging.getLogger(__name__)

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
        self.config = process_configuration(config_path)
        self.setup_base = Path(setup_path).resolve()
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.force = force
        
        # Standard directory structure
        self.required_dirs = [
            "config",
            "logs",
            "schema",
            "scripts"
        ]
    
    def validate_setup(self) -> None:
        """Validate setup requirements"""
        if self.base_dir.exists() and not self.force:
            existing_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            if existing_dirs:
                raise SetupError(
                    f"{self.base_dir} already contains directories. "
                    "Use --force to overwrite existing setup."
                )
        
        # Validate setup base directory
        if not self.setup_base.exists():
            raise SetupError(f"Setup base directory not found: {self.setup_base}")
    
    def create_directory_structure(self) -> None:
        """Create the base directory structure"""
        logger.info(f"Setting up experiment directories in {self.base_dir}")
        
        # Create standard directories
        for dir_path in self.required_dirs:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
        
        # Create protein-specific directories
        for protein_id in self.config.proteins:
            protein_dir = self.base_dir / protein_id
            
            # Create experiment type directories if enabled
            protein_config = self.config.proteins[protein_id]
            
            if protein_config.iterative_masking.enabled:
                (protein_dir / "iterative").mkdir(parents=True, exist_ok=True)
            
            if protein_config.apriori_masking.enabled:
                (protein_dir / "apriori").mkdir(parents=True, exist_ok=True)
            
            if protein_config.frustra_masking.enabled:
                (protein_dir / "frustra").mkdir(parents=True, exist_ok=True)
            
            # Create common directories
            (protein_dir / "controls" / "vanilla" / "configs").mkdir(parents=True, exist_ok=True)
            (protein_dir / "controls" / "vanilla" / "scripts").mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Created directories for protein: {protein_id}")
    
    def setup_shared_resources(self) -> None:
        """Set up shared resources in setup directory"""
        logger.info(f"Setting up shared resources in {self.setup_base}")
        self.setup_base.mkdir(parents=True, exist_ok=True)
        
        # Copy configuration templates if they exist
        pkg_config = Path(__file__).parent.parent / "config"
        config_files = {
            "test.yaml": "config/test.yaml",
            "schema_validation.json": "schema/schema_validation.json"
        }
        
        for src_file, dst_path in config_files.items():
            src_path = pkg_config / src_file
            if src_path.exists():
                dst_full_path = self.base_dir / dst_path
                shutil.copy(src_path, dst_full_path)
                logger.info(f"Copied {src_file} to {dst_full_path}")
            else:
                logger.warning(f"Could not find {src_file} in package config")
    
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