import time
import glob
from typing import List, Optional, Dict
from pathlib import Path
import yaml

from ..experiments.base import Experiment, Control, ControlType, ProteinSystem
from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.slurm import SlurmJobManager, SlurmJobConfig
import logging

logger = logging.getLogger(__name__)

def load_defaults() -> Dict:
    """Load default values from defaults.yaml"""
    defaults_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
    try:
        with open(defaults_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load defaults from {defaults_path}: {e}")
        return {}

# Load defaults at module level
DEFAULTS = load_defaults()

def find_msa_file(base_dir: Path, timeout: int = 3600, check_interval: int = 15) -> Optional[Path]:
    """Find MSA file in job output directories.
    
    Args:
        base_dir: Base directory to search in
        timeout: Maximum time to wait in seconds (default: 1 hour)
        check_interval: Time between checks in seconds (default: 1 minute)
        
    Returns:
        Path to MSA file if found, None if not found within timeout
    """
    start_time = time.time()
    while True:
        # Look for job output directories (they end with a hash)
        job_dirs = list(base_dir.glob("*_[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]"))
        
        for job_dir in job_dirs:
            msa_path = job_dir / "in" / "msa.a3m"
            if msa_path.exists():
                return msa_path
        
        if time.time() - start_time > timeout:
            logger.error(f"Timeout waiting for MSA file in {base_dir}")
            return None
            
        logger.info(f"Waiting for control MSA to be available in {base_dir}... (checking every {check_interval} seconds)")
        time.sleep(check_interval)

class IterativeMaskingExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("iterative_masking", protein, slurm_config)
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)

    def run_iterative_masking(self) -> bool:
        """Run iterative masking experiment"""
        # Find the control's MSA file
        msa_path = find_msa_file(self.vanilla_control.working_dir)
        if not msa_path:
            logger.error("Failed to find control MSA - timeout reached")
            return False
        
        logger.info(f"Using control MSA from {msa_path}")
        
        config = ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_iterative",
            masking_strategy=MaskingStrategy.ITERATIVE_SINGLE,
            parent_path=str(self.working_dir),
            num_recycles=DEFAULTS.get('num_recycles', 2),
            num_seeds=DEFAULTS.get('num_seeds', 2),
            msa_method="custom_a3m",  # Use custom MSA from control
            custom_a3m_path=str(msa_path),  # Use control's MSA
            pipeline_type="masking",  # Use masking pipeline
            masking_mode="on",  # Enable masking
            mask_msa=True,  # Enable MSA masking
            mask_deletion_matrix=True,  # Enable deletion matrix masking
            setup_path=str(self.slurm_config.setup_path),  # Add setup path
            run_control=False
        )
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success

    def run(self) -> bool:
        """Run experiment and control"""
        if not super().run():  # This runs the controls
            return False
            
        return self.run_iterative_masking()

class AprioriMaskingExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("apriori_masking", protein, slurm_config)
        
        if not protein.known_positions:
            raise ValueError("Known positions are required for a priori masking")
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)

    def run_apriori_masking(self) -> bool:
        """Run a priori masking experiment"""
        # Find the control's MSA file
        msa_path = find_msa_file(self.vanilla_control.working_dir)
        if not msa_path:
            logger.error("Failed to find control MSA - timeout reached")
            return False
            
        logger.info(f"Using control MSA from {msa_path}")
        
        config = ExperimentConfig(
            sequence=self.protein.sequence,
            jobname_prefix=f"{self.protein.name}_apriori",
            masking_strategy=MaskingStrategy.MASK_POSITIONS,
            positions=self.protein.known_positions,
            parent_path=str(self.working_dir),
            num_recycles=DEFAULTS.get('num_recycles', 2),
            num_seeds=DEFAULTS.get('num_seeds', 2),
            msa_method="custom_a3m",  # Use custom MSA from control
            custom_a3m_path=str(msa_path),  # Use control's MSA
            setup_path=str(self.slurm_config.setup_path)  # Add setup path
        )
        
        job_manager = SlurmJobManager(
            experiment_config=config,
            slurm_config=self.slurm_config,
            working_dir=str(self.working_dir)
        )
        
        success, failed_jobs = job_manager.run_experiment()
        return success

    def run(self) -> bool:
        """Run experiment and control"""
        if not super().run():  # This runs the controls
            return False
            
        return self.run_apriori_masking() 

class I89SMutationMaskingExperiment(Experiment):
    def __init__(self, protein: ProteinSystem, slurm_config: Optional[SlurmJobConfig] = None):
        super().__init__("i89s_masking", protein, slurm_config)
        
        # Add vanilla control
        self.vanilla_control = Control(
            "vanilla", 
            ControlType.VANILLA, 
            protein,
            self.slurm_config
        )
        self.controls.append(self.vanilla_control)

        # Define mutation combinations
        self.mutation_sets = [
            ["I89S"],  # Single mutant
            ["I89S", "D72D"],  # Double mutant
            ["I89S", "D72D", "G73G"],  # Triple mutant
            ["I89S", "D72D", "G73G", "S74S"],  # Quadruple mutant
            ["I89S", "D72D", "G73G", "S74S", "G75G"],  # Quintuple mutant
            ["I89S", "D72D", "G73G", "S74S", "G75G", "T76T"],  # Sextuple mutant 
            ["I89S", "D72D", "G73G", "S74S", "G75G", "T76T", "E81E"],  # Septuple mutant
        ]

    def run_i89s_masking(self) -> bool:
        """Run I89S mutation and masking experiment"""
        # Find the control's MSA file
        msa_path = find_msa_file(self.vanilla_control.working_dir)
        if not msa_path:
            logger.error("Failed to find control MSA - timeout reached")
            return False
        
        logger.info(f"Using control MSA from {msa_path}")
        
        success = True
        for mutation_set in self.mutation_sets:
            mutation_str = "_".join(mutation_set)
            logger.info(f"Processing mutation combination: {mutation_str}")
            
            config = ExperimentConfig(
                sequence=self.protein.sequence,
                jobname_prefix=f"{self.protein.name}_I89S_{mutation_str}",
                masking_strategy=MaskingStrategy.ITERATIVE_SINGLE,
                parent_path=str(self.working_dir),
                num_recycles=DEFAULTS.get('num_recycles', 2),
                num_seeds=DEFAULTS.get('num_seeds', 2),
                msa_method="custom_a3m",  # Use custom MSA from control
                custom_a3m_path=str(msa_path),  # Use control's MSA
                pipeline_type="mutate_and_mask",  # Use mutate and mask pipeline
                masking_mode="list",  # Use list mode for masking
                mask_msa=True,  # Enable MSA masking
                mask_deletion_matrix=True,  # Enable deletion matrix masking
                mutations=mutation_set,  # Set mutations
                mask_identity=DEFAULTS.get('mask_identity', 'X'),  # Use default mask identity
                debug=DEFAULTS.get('debug', False),
                setup_path=str(self.slurm_config.setup_path)  # Add setup path
            )
            
            job_manager = SlurmJobManager(
                experiment_config=config,
                slurm_config=self.slurm_config,
                working_dir=str(self.working_dir / mutation_str)
            )
            
            job_success, failed_jobs = job_manager.run_experiment()
            if not job_success:
                logger.error(f"Failed to run experiment for {mutation_str}")
                success = False
        
        return success

    def run(self) -> bool:
        """Run experiment and control"""
        if not super().run():  # This runs the controls
            return False
            
        return self.run_i89s_masking() 