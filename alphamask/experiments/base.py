from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import logging
import yaml

from ..utils.slurm import SlurmJobManager, SlurmJobConfig
from ..utils.types import ExperimentConfig
from ..utils.params import load_defaults
from colabdesign.af.contrib import predict
from ..core.model import CompressionConfig

from .types import (
    ProteinConfig, Condition, AprioriExperiment,
    IterativeMasking, AprioriMasking, FrustraMasking
)

import frustrapy

# Get logger for this module
logger = logging.getLogger("alphamask.experiments.base")

class ExperimentError(Exception):
    """Base class for experiment-related errors"""
    pass

class BaseExperiment(ABC):
    """Abstract base class for all experiments"""
    
    def __init__(
        self,
        name: str,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None,
        compression_config: Optional[CompressionConfig] = None
    ):
        self.name = name
        self.protein_config = protein_config
        self.slurm_config = slurm_config or SlurmJobConfig()
        self.working_dir = (working_dir or Path.cwd() / "experiments" / name).resolve()
        self.dry_run = dry_run
        self.schema_path = schema_path.resolve() if schema_path else None
        self.compression_config = compression_config
        
        # Only create the working directory if it's needed
        if not self.dry_run and isinstance(self, (IterativeExperiment, FrustraExperiment)):
            self.working_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created working directory: {self.working_dir}")
        
        # Load default parameters
        self.defaults = load_defaults()
        logger.debug("Loaded default parameters")
        
        # Initialize controls list
        self.controls: List[Control] = []
        
        # Configure experiment-specific logging
        self.debug = logging.getLogger().getEffectiveLevel() == logging.DEBUG
        
    @abstractmethod
    def validate(self) -> None:
        """Validate experiment configuration"""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Set up experiment directories and resources"""
        pass
    
    @abstractmethod
    def submit(self) -> bool:
        """Submit the experiment jobs to SLURM
        
        Returns:
            bool: True if all jobs were submitted successfully, False otherwise
        """
        pass
    
    def cleanup(self) -> None:
        """Clean up resources after experiment"""
        pass

    def _log_config_creation(self, config_path: Path):
        """Log config file creation based on debug level"""
        if self.debug:
            logger.debug(f"Created WT position config at {config_path}")
        else:
            # Only show progress periodically for large numbers of configs
            if config_path.name.endswith(('_1.yaml', '_50.yaml', '_100.yaml')):
                logger.info(f"Creating configs... ({config_path.name})")

class Control:
    """Base class for experiment controls"""
    
    def __init__(
        self,
        name: str,
        protein_config: ProteinConfig,
        slurm_config: SlurmJobConfig,
        conditions: List[Condition],
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None,
        compression_config: Optional[CompressionConfig] = None
    ):
        self.name = name
        self.protein_config = protein_config
        self.slurm_config = slurm_config
        self.conditions = conditions
        self.dry_run = dry_run
        self.schema_path = schema_path
        self.compression_config = compression_config
        
        # Set working directory, ensuring it's under the experiment's controls directory
        if working_dir is None:
            raise ValueError("working_dir must be specified for Control")
            
        # Create the control directory under the specific mode's directory
        self.working_dir = (Path(working_dir) / name).resolve()
        self.defaults = load_defaults()
        
        # Create standard directory structure
        self.config_dir = self.working_dir / "configs"
        self.script_dir = self.working_dir / "scripts"
        self.log_dir = self.working_dir / "logs"
        self.input_dir = self.working_dir / "in"
        # self.output_dir = self.working_dir / "out"
        self.schema_dir = self.working_dir / "schema"
        
        if not self.dry_run:
            # Create all directories
            for dir_path in [
                self.working_dir,
                self.config_dir,
                self.script_dir,
                self.log_dir,
                self.input_dir,
                self.input_dir / "msa",
                # self.output_dir,
                # self.output_dir / "pdbs",
                self.schema_dir
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
        else:
            # Log what would be created in dry run mode
            for dir_path in [
                self.working_dir,
                self.config_dir,
                self.script_dir,
                self.log_dir,
                self.input_dir,
                self.input_dir / "msa",
                # self.output_dir,
                # self.output_dir / "pdbs",
                self.schema_dir
            ]:
                logger.info(f"[DRY RUN] Would create directory: {dir_path}")
    
    def create_experiment_config(self, condition: Condition) -> Tuple[ExperimentConfig, str]:
        """Create experiment configuration for a condition"""
        # Create a unique name for this condition
        condition_name = f"{self.name}_{condition.mask}_{condition.mutate}"
        
        # Get the experiment positions if this is part of an AprioriExperiment
        experiment_positions = []
        mutations = []
        
        if isinstance(self.protein_config.apriori_masking, AprioriMasking):
            for exp in self.protein_config.apriori_masking.experiments:
                if exp.name in self.name:
                    experiment_positions = exp.positions
                    mutations = exp.mutations if condition.mutate else []
                    break
        elif isinstance(self.protein_config.iterative_masking, IterativeMasking):
            mutations = self.protein_config.iterative_masking.mutations if condition.mutate else []
        
        # Map pipeline type - 'vanilla' should be mapped to 'default'
        pipeline_type = "default" if not (condition.mask or condition.mutate) else "mutate_and_mask"
        
        # Create the config
        config = ExperimentConfig(
            sequence=self.protein_config.sequence,
            jobname_prefix=self.name,
            parent_path=str(self.working_dir),
            pipeline_type=pipeline_type,
            positions=experiment_positions,
            mutations=mutations,
            num_recycles=self.defaults.get('num_recycles', 2),
            num_seeds=self.defaults.get('num_seeds', 2),
            setup_path=str(self.slurm_config.setup_path)
        )
        
        # Create job name
        job_name = f"{self.name}_{condition.mask}_{condition.mutate}"
        
        # Save the config to a file or log it in dry run mode
        config_path = self.config_dir / f"config_{condition_name}.yaml"
        if not self.dry_run:
            config.save(config_path)
            logger.info(f"Created config file at: {config_path}")
            logger.info(f"Config details: masking={condition.mask}, mutate={condition.mutate}, positions={experiment_positions}, mutations={mutations}")
        else:
            logger.info(f"[DRY RUN] Would create config file at: {config_path}")
            logger.info(f"[DRY RUN] Config contents would be:")
            logger.info(f"[DRY RUN] - Sequence: {config.sequence[:20]}...")
            logger.info(f"[DRY RUN] - Positions to mask: {config.positions}")
            logger.info(f"[DRY RUN] - Mutations: {config.mutations}")
            logger.info(f"[DRY RUN] - Pipeline type: {config.pipeline_type}")
        
        return config, job_name
    
    def run(self) -> bool:
        """Run control experiments for all conditions"""
        try:
            config, job_name = self.create_experiment_config(
                Condition(mask=False, mutate=False)
            )
            
            job_manager = SlurmJobManager(
                experiment_config=config,
                slurm_config=self.slurm_config,
                working_dir=str(self.working_dir),
                job_name=job_name
            )
            
            success, failed_jobs = job_manager.run_experiment()
            return success
            
        except Exception as e:
            logger.error(f"Error running control: {str(e)}")
            return False

class IterativeExperiment(BaseExperiment):
    """Base class for iterative masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None,
        compression_config: Optional[CompressionConfig] = None
    ):
        super().__init__(
            "iterative_masking", 
            protein_config, 
            slurm_config, 
            working_dir, 
            dry_run, 
            schema_path,
            compression_config
        )
        
        if not protein_config.iterative_masking.enabled:
            raise ExperimentError("Iterative masking is not enabled in configuration")
    
    def validate(self) -> None:
        """Validate iterative masking configuration"""
        if not self.protein_config.iterative_masking.mutations:
            raise ExperimentError("No mutations specified for iterative masking")
    
    def setup(self) -> None:
        """Set up iterative masking experiment"""
        # Create directories for iterative masking
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create WT directory
        wt_dir = self.working_dir / "WT"
        wt_dir.mkdir(parents=True, exist_ok=True)
        
        # Get sequence length for masking positions
        sequence_length = len(self.protein_config.sequence)
        
        # Create configs directory for position-specific configs
        configs_dir = wt_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create WT configs for each position
        for pos in range(1, sequence_length + 1):
            config = {
                "sequence": self.protein_config.sequence,
                "jobname_prefix": f"WT_pos_{pos}",
                "parent_path": str(wt_dir),
                "setup_path": str(self.slurm_config.setup_path),
                "pipeline_type": "masking",
                "masking_mode": "list",
                "mask_msa": True,
                "mask_deletion_matrix": True,
                "cols": [pos],
                "mask_identity": self.protein_config.iterative_masking.mask_token,
                "num_recycles": self.defaults.get('num_recycles', 2),
                "num_seeds": self.defaults.get('num_seeds', 2)
            }
            
            config_path = configs_dir / f"config_pos_{pos}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        
        # Handle mutations
        for mutation_set in self.protein_config.iterative_masking.mutations:
            # Create directory for mutation(s)
            mutation_name = '_'.join(mutation_set)
            mutation_dir = self.working_dir / mutation_name
            mutation_dir.mkdir(parents=True, exist_ok=True)
            
            # Create configs directory for mutation
            mut_configs_dir = mutation_dir / "configs"
            mut_configs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create configs for each position with mutation
            for pos in range(1, sequence_length + 1):
                config = {
                    "sequence": self.protein_config.sequence,
                    "jobname_prefix": f"{mutation_name}_pos_{pos}",
                    "parent_path": str(mutation_dir),
                    "setup_path": str(self.slurm_config.setup_path),
                    "pipeline_type": "mutate_and_mask",
                    "masking_mode": "list",
                    "mask_msa": True,
                    "mask_deletion_matrix": True,
                    "cols": [pos],
                    "mask_identity": self.protein_config.iterative_masking.mask_token,
                    "mutations": mutation_set,
                    "wt_msa_path": self.working_dir / "WT/in/msa.a3m",
                    "custom_a3m_path": self.working_dir / "WT/in/msa.a3m",
                    "msa_method": "custom_a3m",
                    "num_recycles": self.defaults.get('num_recycles', 2),
                    "num_seeds": self.defaults.get('num_seeds', 2),
                    "copies": self.defaults.get('copies', 1),
                    "pair_mode": self.defaults.get('pair_mode', 'unpaired_paired'),
                    "cov": self.defaults.get('cov', 75),
                    "id": self.defaults.get('id', 90),
                    "qid": self.defaults.get('qid', 0),
                    "do_not_filter": self.defaults.get('do_not_filter', False),
                    "template_mode": self.defaults.get('template_mode', 'none'),
                    "pdb": self.defaults.get('pdb', ''),
                    "chain": self.defaults.get('chain', 'A'),
                    "rm_template_seq": self.defaults.get('rm_template_seq', False),
                    "propagate_to_copies": self.defaults.get('propagate_to_copies', True),
                    "do_not_align": self.defaults.get('do_not_align', False),
                    "model_type": self.defaults.get('model_type', 'monomer (ptm)'),
                    "rank_by": self.defaults.get('rank_by', 'auto'),
                    "debug": self.defaults.get('debug', False),
                    "use_initial_guess": self.defaults.get('use_initial_guess', False),
                    "num_msa": self.defaults.get('num_msa', 512),
                    "num_extra_msa": self.defaults.get('num_extra_msa', 1024),
                    "use_cluster_profile": self.defaults.get('use_cluster_profile', True),
                    "model": self.defaults.get('model', 'all'),
                    "recycle_early_stop_tolerance": self.defaults.get('recycle_early_stop_tolerance', 0.0),
                    "select_best_across_recycles": self.defaults.get('select_best_across_recycles', False),
                    "use_mlm": self.defaults.get('use_mlm', False),
                    "use_dropout": self.defaults.get('use_dropout', False),
                    "seed": self.defaults.get('seed', 0),
                    "show_images": self.defaults.get('show_images', False),
                    "cols_range": self.defaults.get('cols_range', [])
                }
                
                config_path = mut_configs_dir / f"config_pos_{pos}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
    
    def submit(self) -> bool:
        """Submit iterative masking experiment jobs"""
        try:
            # Get protein root directory for shared MSA
            protein_dir = self.working_dir.parent
            shared_msa_dir = protein_dir / "in" / "msa"
            shared_msa_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize a temporary job manager to handle MSA generation
            temp_job_manager = SlurmJobManager(
                experiment_config=ExperimentConfig(
                    sequence=self.protein_config.sequence,
                    jobname_prefix="msa_generation",
                    parent_path=str(shared_msa_dir.parent),  # Use 'in' directory as parent
                    setup_path=str(self.slurm_config.setup_path),
                    pipeline_type="default",
                    msa_method="mmseqs2"  # Force MSA generation
                ),
                slurm_config=self.slurm_config,
                working_dir=str(shared_msa_dir.parent),  # Use 'in' directory as working dir
                job_name="msa_generation"
            )
            
            # Generate MSA first
            logger.info("Generating shared MSA...")
            temp_job_manager._handle_msa()
            logger.info("MSA generation complete")
            
            # Ensure working directory exists
            self.working_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standard subdirectories at root level
            for subdir in ["configs", "scripts", "logs", "in", "out", "schema"]:
                (self.working_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create MSA directory at root level
            (self.working_dir / "in" / "msa").mkdir(parents=True, exist_ok=True)
            
            # Create WT directory with its subdirectories
            wt_dir = self.working_dir / "WT"
            wt_dir.mkdir(parents=True, exist_ok=True)
            
            for subdir in ["configs", "scripts", "logs", "in", "out", "schema"]:
                (wt_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create MSA directory for WT
            (wt_dir / "in" / "msa").mkdir(parents=True, exist_ok=True)
            (wt_dir / "out" / "pdbs").mkdir(parents=True, exist_ok=True)
            
            # Create WT config and position-specific configs
            sequence_length = len(self.protein_config.sequence)
            
            # Create WT position-specific configs
            for pos in range(1, sequence_length + 1):
                config = {
                    "sequence": self.protein_config.sequence,
                    "jobname": f"WT_pos_{pos}",
                    "parent_path": str(wt_dir),
                    "setup_path": str(self.slurm_config.setup_path),
                    "pipeline_type": "masking",  # Explicitly set pipeline type for WT
                    "masking_mode": "list",
                    "mask_msa": True,
                    "mask_deletion_matrix": True,
                    "cols": [pos],
                    "mask_identity": self.protein_config.iterative_masking.mask_token,
                    "num_recycles": self.defaults.get('num_recycles', 2),
                    "num_seeds": self.defaults.get('num_seeds', 2),
                    # Add all required default values
                    "unified_memory": self.defaults.get('unified_memory', False),
                    "copies": self.defaults.get('copies', 1),
                    "msa_method": "custom_a3m",
                    "custom_a3m_path": str(shared_msa_dir / "msa.a3m"),  # Use shared MSA path
                    "pair_mode": self.defaults.get('pair_mode', 'unpaired_paired'),
                    "cov": self.defaults.get('cov', 75),
                    "id": self.defaults.get('id', 90),
                    "qid": self.defaults.get('qid', 0),
                    "do_not_filter": self.defaults.get('do_not_filter', False),
                    "template_mode": self.defaults.get('template_mode', 'none'),
                    "pdb": self.defaults.get('pdb', ''),
                    "chain": self.defaults.get('chain', 'A'),
                    "rm_template_seq": self.defaults.get('rm_template_seq', False),
                    "propagate_to_copies": self.defaults.get('propagate_to_copies', True),
                    "do_not_align": self.defaults.get('do_not_align', False),
                    "model_type": self.defaults.get('model_type', 'monomer (ptm)'),
                    "rank_by": self.defaults.get('rank_by', 'auto'),
                    "debug": self.defaults.get('debug', False),
                    "use_initial_guess": self.defaults.get('use_initial_guess', False),
                    "num_msa": self.defaults.get('num_msa', 512),
                    "num_extra_msa": self.defaults.get('num_extra_msa', 1024),
                    "use_cluster_profile": self.defaults.get('use_cluster_profile', True),
                    "model": self.defaults.get('model', 'all'),
                    "recycle_early_stop_tolerance": self.defaults.get('recycle_early_stop_tolerance', 0.0),
                    "select_best_across_recycles": self.defaults.get('select_best_across_recycles', False),
                    "use_mlm": self.defaults.get('use_mlm', False),
                    "use_dropout": self.defaults.get('use_dropout', False),
                    "seed": self.defaults.get('seed', 0),
                    "show_images": self.defaults.get('show_images', False),
                    "cols_range": self.defaults.get('cols_range', [])
                }
                
                config_path = wt_dir / "configs" / f"WT_config_pos_{pos}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                logger.info(f"Created WT position config at {config_path}")
            
            # Create mutation directories and their position-specific configs
            for mutation_set in self.protein_config.iterative_masking.mutations:
                mutation_name = '_'.join(mutation_set)
                mutation_dir = self.working_dir / mutation_name
                mutation_dir.mkdir(parents=True, exist_ok=True)
                
                # Create standard subdirectories for mutation
                for subdir in ["configs", "scripts", "logs", "in", "out", "schema"]:
                    (mutation_dir / subdir).mkdir(parents=True, exist_ok=True)
                
                # Create MSA directory for mutation
                (mutation_dir / "in" / "msa").mkdir(parents=True, exist_ok=True)
                (mutation_dir / "out" / "pdbs").mkdir(parents=True, exist_ok=True)
                
                # Create position-specific configs for mutation
                for pos in range(1, sequence_length + 1):
                    config = {
                        "sequence": self.protein_config.sequence,
                        "jobname": f"{mutation_name}_pos_{pos}",
                        "parent_path": str(mutation_dir),
                        "setup_path": str(self.slurm_config.setup_path),
                        "pipeline_type": "mutate_and_mask",  # Explicitly set pipeline type for mutations
                        "masking_mode": "list",
                        "mask_msa": True,
                        "mask_deletion_matrix": True,
                        "cols": [pos],
                        "mask_identity": self.protein_config.iterative_masking.mask_token,
                        "mutations": mutation_set,
                        "wt_msa_path": str(shared_msa_dir / "msa.a3m"),  # Use shared MSA path
                        "custom_a3m_path": str(shared_msa_dir / "msa.a3m"),  # Use shared MSA path
                        "msa_method": "custom_a3m",
                        "num_recycles": self.defaults.get('num_recycles', 2),
                        "num_seeds": self.defaults.get('num_seeds', 2),
                        # Add all required default values
                        "unified_memory": self.defaults.get('unified_memory', False),
                        "copies": self.defaults.get('copies', 1),
                        "pair_mode": self.defaults.get('pair_mode', 'unpaired_paired'),
                        "cov": self.defaults.get('cov', 75),
                        "id": self.defaults.get('id', 90),
                        "qid": self.defaults.get('qid', 0),
                        "do_not_filter": self.defaults.get('do_not_filter', False),
                        "template_mode": self.defaults.get('template_mode', 'none'),
                        "pdb": self.defaults.get('pdb', ''),
                        "chain": self.defaults.get('chain', 'A'),
                        "rm_template_seq": self.defaults.get('rm_template_seq', False),
                        "propagate_to_copies": self.defaults.get('propagate_to_copies', True),
                        "do_not_align": self.defaults.get('do_not_align', False),
                        "model_type": self.defaults.get('model_type', 'monomer (ptm)'),
                        "rank_by": self.defaults.get('rank_by', 'auto'),
                        "debug": self.defaults.get('debug', False),
                        "use_initial_guess": self.defaults.get('use_initial_guess', False),
                        "num_msa": self.defaults.get('num_msa', 512),
                        "num_extra_msa": self.defaults.get('num_extra_msa', 1024),
                        "use_cluster_profile": self.defaults.get('use_cluster_profile', True),
                        "model": self.defaults.get('model', 'all'),
                        "recycle_early_stop_tolerance": self.defaults.get('recycle_early_stop_tolerance', 0.0),
                        "select_best_across_recycles": self.defaults.get('select_best_across_recycles', False),
                        "use_mlm": self.defaults.get('use_mlm', False),
                        "use_dropout": self.defaults.get('use_dropout', False),
                        "seed": self.defaults.get('seed', 0),
                        "show_images": self.defaults.get('show_images', False),
                        "cols_range": self.defaults.get('cols_range', [])
                    }
                    
                    config_path = mutation_dir / "configs" / f"{mutation_name}_config_pos_{pos}.yaml"
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f)
                    logger.debug(f"Created mutation position config at {config_path}")
            
            # Initialize job manager for submitting jobs
            job_manager = SlurmJobManager(
                experiment_config=ExperimentConfig(
                    sequence=self.protein_config.sequence,
                    jobname_prefix=self.name,
                    parent_path=str(self.working_dir),
                    setup_path=str(self.slurm_config.setup_path),
                    pipeline_type="default",  # This doesn't matter as we'll use config-specific pipeline types
                    msa_method="custom_a3m",  # Use custom MSA method
                    custom_a3m_path=str(shared_msa_dir / "msa.a3m")  # Use shared MSA path
                ),
                slurm_config=self.slurm_config,
                working_dir=str(self.working_dir),
                job_name=self.name
            )
            
            # Submit WT position-specific jobs
            for pos in range(1, sequence_length + 1):
                config_path = wt_dir / "configs" / f"WT_config_pos_{pos}.yaml"
                job_id = job_manager.submit_job(
                    str(config_path),
                    f"WT_pos_{pos}",
                    script_dir=str(wt_dir / "scripts"),
                    log_dir=str(wt_dir / "logs")
                )
                logger.debug(f"Submitted WT position {pos} job with ID: {job_id}")
            
            # Submit mutation position-specific jobs
            for mutation_set in self.protein_config.iterative_masking.mutations:
                mutation_name = '_'.join(mutation_set)
                mutation_dir = self.working_dir / mutation_name
                for pos in range(1, sequence_length + 1):
                    config_path = mutation_dir / "configs" / f"{mutation_name}_config_pos_{pos}.yaml"
                    job_id = job_manager.submit_job(
                        str(config_path),
                        f"{mutation_name}_pos_{pos}",
                        script_dir=str(mutation_dir / "scripts"),
                        log_dir=str(mutation_dir / "logs")
                    )
                    logger.debug(f"Submitted {mutation_name} position {pos} job with ID: {job_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit experiment: {str(e)}")
            return False

    def _create_wt_config(self) -> ExperimentConfig:
        """Create configuration for WT job"""
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": f"{self.name}_WT",
            "parent_path": str(self.working_dir / "WT"),
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": "default"
        }
        
        return ExperimentConfig(**config)

    def _create_mutation_config(self, mutation_set: List[str]) -> ExperimentConfig:
        """Create configuration for mutation job"""
        mutation_name = "_".join(mutation_set)
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": f"{self.name}_mutation_{mutation_name}",
            "parent_path": str(self.working_dir / mutation_name),
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": "mutate"
        }
        
        return ExperimentConfig(**config)

    def _create_masking_config(self) -> ExperimentConfig:
        """Create configuration for iterative masking job"""
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": f"{self.name}",
            "parent_path": str(self.working_dir),
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": "mutate_and_mask"
        }
        
        return ExperimentConfig(**config)

class AprioriExperiment(BaseExperiment):
    """Implementation of a priori masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None,
        compression_config: Optional[CompressionConfig] = None
    ):
        super().__init__(
            "apriori_masking", 
            protein_config, 
            slurm_config, 
            working_dir, 
            dry_run, 
            schema_path,
            compression_config
        )
        
        if not protein_config.apriori_masking.enabled:
            raise ExperimentError("A priori masking is not enabled in configuration")
            
        self.experiments = protein_config.apriori_masking.experiments
    
    def validate(self) -> None:
        """Validate a priori masking configuration"""
        if not self.experiments:
            raise ExperimentError("No experiments specified for a priori masking")
        
        for experiment in self.experiments:
            # Validate experiment name
            if not experiment.name:
                raise ExperimentError("Experiment name cannot be empty")
            
            # Validate positions
            if not experiment.positions:
                raise ExperimentError(f"No positions specified for experiment {experiment.name}")
            
            for pos in experiment.positions:
                if pos < 1 or pos > len(self.protein_config.sequence):
                    raise ExperimentError(
                        f"Invalid position {pos} in experiment {experiment.name}"
                    )
            
            # Validate mutations and conditions
            has_mutations = bool(experiment.mutations)
            required_conditions = [
                (False, False), (True, False), (False, True), (True, True)
            ] if has_mutations else [
                (False, False), (True, False), (False, True)
            ]
            
            if has_mutations:
                for mutation in experiment.mutations:
                    if not self._validate_mutation(mutation):
                        raise ExperimentError(
                            f"Invalid mutation {mutation} in experiment {experiment.name}"
                        )
            
            experiment_conditions = [
                (c.mask, c.mutate) for c in experiment.conditions
            ]
            if sorted(experiment_conditions) != sorted(required_conditions):
                raise ExperimentError(
                    f"Experiment {experiment.name} requires exactly "
                    f"{'4 conditions when mutations are specified' if has_mutations else '3 conditions when no mutations are specified'}"
                )
    
    def setup(self) -> None:
        """Set up shared MSA directory and other necessary directories"""
        # Get protein root directory for shared MSA
        protein_dir = self.working_dir.parent
        
        # Create shared MSA directory at protein level
        shared_msa_dir = protein_dir / "in" / "msa"
        shared_msa_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        for subdir in ["configs", "scripts", "logs", "in", "out", "schema"]:
            (self.working_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _create_config(self, experiment: AprioriExperiment, condition: Condition) -> Tuple[ExperimentConfig, str]:
        """Create configuration for a specific a priori experiment condition
        
        Returns:
            Tuple[ExperimentConfig, str]: The config object and job name
        """
        # Get shared MSA path
        shared_msa_dir = self.working_dir.parent / "in" / "msa"
        shared_msa_path = shared_msa_dir / "msa.a3m"
        
        # Create descriptive name for this condition
        condition_name = f"{'masked' if condition.mask else 'unmasked'}_{'mutated' if condition.mutate else 'unmutated'}"
        
        # Get hash for the sequence
        seq_hash = predict.get_hash(self.protein_config.sequence)[:5]
        job_name = f"{experiment.name}_{seq_hash}"
        
        # Set pipeline type based on condition
        if condition.mask and condition.mutate:
            pipeline_type = "mutate_and_mask"
        elif condition.mask:
            pipeline_type = "masking"
        elif condition.mutate:
            pipeline_type = "mutate"
        else:
            pipeline_type = "default"
        
        # Set positions and mutations based on condition
        positions = experiment.positions if condition.mask else []
        mutations = experiment.mutations if condition.mutate else []
        
        # Create working directory for this condition
        condition_dir = self.working_dir / experiment.name / condition_name
        configs_dir = condition_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create config with proper masking settings
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": job_name,  # Use job_name for consistency
            "parent_path": str(condition_dir),
            "positions": positions,
            "cols": positions,
            "mutations": mutations,
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": pipeline_type,
            "create_control": False,  # Never create controls in condition directories
            "mask_msa": condition.mask,
            "mask_deletion_matrix": condition.mask,
            "masking_mode": "list" if condition.mask else "off",
            "mask_identity": "X",
            "msa_method": "custom_a3m",  # Always use custom MSA
            "custom_a3m_path": str(shared_msa_path)  # Use shared MSA path
        }
        
        # Create ExperimentConfig instance
        experiment_config = ExperimentConfig(**config)
        
        return experiment_config, job_name
    
    def submit(self) -> bool:
        """Orchestrate the experiment submission process"""
        try:
            self.validate()
            self.setup()
            
            # Get protein root directory for shared MSA
            protein_dir = self.working_dir.parent
            shared_msa_dir = protein_dir / "in" / "msa"
            shared_msa_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize a temporary job manager to handle MSA generation
            temp_job_manager = SlurmJobManager(
                experiment_config=ExperimentConfig(
                    sequence=self.protein_config.sequence,
                    jobname_prefix="msa_generation",
                    parent_path=str(shared_msa_dir.parent),  # Use 'in' directory as parent
                    setup_path=str(self.slurm_config.setup_path),
                    pipeline_type="default",
                    msa_method="mmseqs2"  # Force MSA generation
                ),
                slurm_config=self.slurm_config,
                working_dir=str(shared_msa_dir.parent),  # Use 'in' directory as working dir
                job_name="msa_generation"
            )
            
            # Generate MSA first
            logger.info("Generating shared MSA...")
            temp_job_manager._handle_msa()
            logger.info("MSA generation complete")
            
            # Submit each experiment with its conditions
            success = True
            for experiment in self.experiments:
                for condition in experiment.conditions:
                    # Create experiment configuration and get job name
                    config, job_name = self._create_config(experiment, condition)
                    
                    # Get condition name for directory structure
                    condition_name = f"{'masked' if condition.mask else 'unmasked'}_{'mutated' if condition.mutate else 'unmutated'}"
                    
                    # Get condition directory paths
                    condition_dir = self.working_dir / experiment.name / condition_name
                    script_dir = condition_dir / "scripts"
                    log_dir = condition_dir / "logs"
                    
                    # Create necessary directories
                    for dir_path in [condition_dir, script_dir, log_dir]:
                        dir_path.mkdir(parents=True, exist_ok=True)
                    
                    # Update config to use shared MSA
                    config.msa_method = "custom_a3m"
                    config.custom_a3m_path = str(shared_msa_dir / "msa.a3m")
                    
                    # Initialize SLURM job manager for this condition
                    job_manager = SlurmJobManager(
                        experiment_config=config,
                        slurm_config=self.slurm_config,
                        working_dir=str(condition_dir),
                        job_name=job_name
                    )
                    
                    # Let SlurmJobManager handle the actual job submission
                    exp_success, failed_jobs = job_manager.run_experiment()
                    if not exp_success:
                        logger.error(f"Failed to submit experiment {experiment.name} condition {condition}")
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error in experiment submission: {str(e)}")
            return False

    def _validate_mutation(self, mutation: str) -> bool:
        """Validate a single mutation against the protein sequence"""
        if len(mutation) < 3:
            return False
            
        orig_aa = mutation[0]
        new_aa = mutation[-1]
        try:
            pos = int(mutation[1:-1])
        except ValueError:
            return False
        
        if pos < 1 or pos > len(self.protein_config.sequence):
            return False
            
        if self.protein_config.sequence[pos-1] != orig_aa:
            return False
            
        return True

class FrustraExperiment(BaseExperiment):
    """Implementation of Frustra-guided masking experiments"""
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None,
        dry_run: bool = False,
        schema_path: Optional[Path] = None,
        compression_config: Optional[CompressionConfig] = None
    ):
        super().__init__(
            "frustra_masking", 
            protein_config, 
            slurm_config, 
            working_dir, 
            dry_run, 
            schema_path,
            compression_config
        )
        
        if not protein_config.frustra_masking.enabled:
            raise ExperimentError("Frustra masking is not enabled in configuration")
        
        # Create analysis directory for storing FrustraPy results
        self.analysis_dir = self.working_dir / "analysis" / "frustra" / "configurational"
        self.results_file = self.analysis_dir / "results.pkl"
        
        # Initialize empty list for storing mutations
        self.mutations = []
    
    def validate(self) -> None:
        """Validate Frustra masking configuration"""
        if self.protein_config.frustra_masking.top_positions < 1:
            raise ExperimentError("Invalid number of top positions for Frustra analysis")
    
    def setup(self) -> None:
        """Set up Frustra masking experiment directories and shared MSA"""
        # Get protein root directory for shared MSA
        protein_dir = self.working_dir.parent
        
        # Create shared MSA directory at protein level
        shared_msa_dir = protein_dir / "in" / "msa"
        shared_msa_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a controls subdirectory specifically for Frustra
        self.controls_dir = self.working_dir / "controls"
        if not self.dry_run:
            self.controls_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standard subdirectories
            for subdir in ["configs", "scripts", "logs", "in", "out", "schema"]:
                (self.working_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create analysis directory structure
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"[DRY RUN] Would create directory: {self.controls_dir}")

        # Add vanilla control
        self.controls.append(Control(
            "vanilla",
            self.protein_config,
            self.slurm_config,
            [Condition(mask=False, mutate=False)],
            working_dir=self.controls_dir,
            dry_run=self.dry_run,
            schema_path=self.schema_path
        ))
    
    def _run_frustra_analysis(self, pdb_path: Path) -> List[int]:
        """Run FrustraPy analysis on the best PDB from control prediction
        
        Args:
            pdb_path: Path to the PDB file to analyze
            
        Returns:
            List of top N positions based on frustration scores
        """
        try:
            # Create FrustraPy analysis directory if it doesn't exist
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a subdirectory for the PDB file
            pdbs_dir = self.analysis_dir / "best_pdb"
            pdbs_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy PDB file to analysis directory
            import shutil
            pdb_file = pdb_path.name
            dest_path = pdbs_dir / pdb_file
            shutil.copy(str(pdb_path), str(dest_path))
            
            # Run FrustraPy analysis in configurational mode
            logger.info(f"Running FrustraPy analysis on {dest_path}")
            pdb_config, plots_config, density_results, _ = frustrapy.calculate_frustration(
                pdb_file=str(dest_path),
                mode="configurational",
                results_dir=str(self.analysis_dir),
                debug="INFO",
                chain="A"  # Assuming chain A, might need to make configurable
            )
            
            # Convert density results to a DataFrame for easier processing
            import pandas as pd
            data = []
            for density in density_results.densities:
                data.append({
                    'Residue': density.residue_number,
                    'Chain': density.chain_id,
                    'Total_Density': density.total_density,
                    'Highly_Frustrated': density.highly_frustrated,
                    'Neutrally_Frustrated': density.neutrally_frustrated,
                    'Minimally_Frustrated': density.minimally_frustrated,
                    'Rel_Highly_Frustrated': density.rel_highly_frustrated,
                    'Rel_Neutrally_Frustrated': density.rel_neutrally_frustrated,
                    'Rel_Minimally_Frustrated': density.rel_minimally_frustrated
                })
            
            df = pd.DataFrame(data)
            
            # Sort by minimally frustrated ratio (descending) to get top positions
            # This can be made configurable based on the metric and direction
            df = df.sort_values(by='Rel_Minimally_Frustrated', ascending=False)
            
            # Get top N positions
            top_n = min(self.protein_config.frustra_masking.top_positions, len(df))
            positions = df.head(top_n)['Residue'].tolist()
            positions.sort()  # Sort positions in ascending order
            
            # Save complete results
            import pickle
            results = {
                'positions': positions,
                'mode': 'configurational',
                'pdb': str(dest_path),
                'density_data': df.to_dict('records'),
                'plots_config': plots_config,
                'pdb_config': pdb_config
            }
            
            with open(self.results_file, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Selected top {len(positions)} positions: {positions}")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to run FrustraPy analysis: {str(e)}")
            raise
    
    def _create_mutation_experiments(self, positions: List[int]) -> None:
        """Create mutation experiments for the identified positions
        
        Args:
            positions: List of positions identified by FrustraPy analysis
        """
        for pos in positions:
            # Get the original amino acid at this position
            orig_aa = self.protein_config.sequence[pos-1]
            
            # Create mutation directory
            mutation_name = f"pos_{pos}"
            mutation_dir = self.working_dir / mutation_name
            
            if not self.dry_run:
                mutation_dir.mkdir(parents=True, exist_ok=True)
                
                # Create standard condition directories
                conditions = [
                    "masked_mutated",
                    "masked_unmutated",
                    "unmasked_mutated",
                    "unmasked_unmutated"
                ]
                
                for condition in conditions:
                    condition_dir = mutation_dir / condition
                    condition_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create standard subdirectories in each condition
                    for subdir in ["configs", "scripts", "logs", "in", "out"]:
                        (condition_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Store mutation information
            self.mutations.append({
                'position': pos,
                'original_aa': orig_aa,
                'directory': mutation_dir
            })
    
    def _create_config(self) -> ExperimentConfig:
        """Create configuration for Frustra masking experiment"""
        # Get shared MSA path
        shared_msa_dir = self.working_dir.parent / "in" / "msa"
        shared_msa_path = shared_msa_dir / "msa.a3m"
        
        config = {
            "sequence": self.protein_config.sequence,
            "jobname_prefix": f"{self.name}",
            "parent_path": str(self.working_dir),
            "num_recycles": self.defaults.get('num_recycles', 2),
            "num_seeds": self.defaults.get('num_seeds', 2),
            "setup_path": str(self.slurm_config.setup_path),
            "pipeline_type": "masking",
            "msa_method": "custom_a3m",  # Always use custom MSA
            "custom_a3m_path": str(shared_msa_path),  # Use shared MSA path
            "masking_mode": "list",
            "mask_msa": True,
            "mask_deletion_matrix": True,
            "mask_identity": "X",
            "positions": [],  # Will be set later for each condition
            "cols": []  # Will be set later for each condition
        }
        
        return ExperimentConfig(**config)
    
    def submit(self) -> bool:
        """Submit Frustra masking experiment jobs to SLURM"""
        try:
            # Ensure we're using the correct partition and GPU type
            logger.info(f"Using partition {self.slurm_config.partition} with GPU type {self.slurm_config.gpu_type}")
            
            self.validate()
            self.setup()
            
            # Get sequence hash
            from colabdesign.af.contrib import predict
            seq_hash = predict.get_hash(self.protein_config.sequence)[:5]
            logger.info(f"Using sequence hash: {seq_hash}")
            
            # Get protein root directory for shared MSA
            protein_dir = self.working_dir.parent
            shared_msa_dir = protein_dir / "in" / "msa"
            shared_msa_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize a temporary job manager to handle MSA generation
            msa_slurm_config = SlurmJobConfig(
                partition=self.slurm_config.partition,
                gpu_type=self.slurm_config.gpu_type,
                setup_path=self.slurm_config.setup_path,
                container_path=self.slurm_config.container_path,
                schema_path=self.slurm_config.schema_path
            )
            
            temp_job_manager = SlurmJobManager(
                experiment_config=ExperimentConfig(
                    sequence=self.protein_config.sequence,
                    jobname_prefix="msa_generation",
                    parent_path=str(shared_msa_dir.parent),  # Use 'in' directory as parent
                    setup_path=str(self.slurm_config.setup_path),
                    pipeline_type="default",
                    msa_method="mmseqs2"  # Force MSA generation
                ),
                slurm_config=msa_slurm_config,  # Use our config with correct partition/GPU
                working_dir=str(shared_msa_dir.parent),  # Use 'in' directory as working dir
                job_name="msa_generation"
            )
            
            # Generate MSA first
            logger.info(f"Generating shared MSA using {msa_slurm_config.partition}/{msa_slurm_config.gpu_type}...")
            temp_job_manager._handle_msa()
            logger.info("MSA generation complete")
            
            # Run controls first to get the best PDB for FrustraPy analysis
            logger.info(f"Submitting control prediction job to {self.slurm_config.partition}/{self.slurm_config.gpu_type}...")
            control_job_id = None
            for control in self.controls:
                # Update control's slurm_config to match parent's config
                control.slurm_config = SlurmJobConfig(
                    partition=self.slurm_config.partition,
                    gpu_type=self.slurm_config.gpu_type,
                    setup_path=self.slurm_config.setup_path,
                    container_path=self.slurm_config.container_path,
                    schema_path=self.slurm_config.schema_path
                )
                if not control.run():
                    return False
                
                # Get the job ID from the most recent submission
                import subprocess
                result = subprocess.run(['squeue', '--me', '--format=%i', '--noheader'], capture_output=True, text=True)
                if result.returncode == 0:
                    jobs = result.stdout.strip().split('\n')
                    if jobs and jobs[0]:  # Check if we got any jobs
                        control_job_id = jobs[0].strip()
                        logger.info(f"Control job submitted with ID: {control_job_id}")
            
            if not control_job_id:
                logger.error("Failed to get control job ID")
                return False
            
            # Wait for control job to complete with 1-hour timeout
            import time
            max_wait = 3600  # 1 hour
            wait_interval = 5  # Check every 30 seconds
            waited = 0
            
            while waited < max_wait:
                # Check if job is still in queue
                result = subprocess.run(['squeue', '--job', control_job_id, '--noheader'], capture_output=True, text=True)
                if result.returncode == 0 and not result.stdout.strip():
                    logger.info(f"Control job {control_job_id} completed")
                    break
                
                logger.info(f"Waiting for control job {control_job_id} to complete... ({waited}s/{max_wait}s)")
                time.sleep(wait_interval)
                waited += wait_interval
            
            if waited >= max_wait:
                logger.error(f"Control job {control_job_id} did not complete within {max_wait} seconds")
                return False
            
            # Wait for the PDB file to appear
            control_pdb = self.controls_dir / "vanilla" / "out" / "pdbs" / f"vanilla_{seq_hash}_best.pdb"
            logger.info(f"Looking for PDB file at: {control_pdb}")
            
            # Additional wait for file to appear (5 minutes)
            max_file_wait = 300  # 5 minutes
            waited = 0
            
            while not control_pdb.exists() and waited < max_file_wait:
                logger.info(f"Waiting for PDB file to be generated... ({waited}s/{max_file_wait}s)")
                time.sleep(wait_interval)
                waited += wait_interval
            
            if not control_pdb.exists():
                raise ExperimentError(f"Control prediction failed - no best PDB found at {control_pdb} after waiting {max_file_wait}s")
            
            logger.info(f"Found control PDB at {control_pdb}")
            
            # Run FrustraPy analysis on the control PDB
            positions = self._run_frustra_analysis(control_pdb)
            
            # Create mutation experiments for identified positions
            self._create_mutation_experiments(positions)
            
            # Submit jobs for each mutation and condition
            for mutation in self.mutations:
                pos = mutation['position']
                mutation_dir = mutation['directory']
                
                # Create and submit jobs for each condition
                conditions = [
                    Condition(mask=True, mutate=True),
                    Condition(mask=True, mutate=False),
                    Condition(mask=False, mutate=True),
                    Condition(mask=False, mutate=False)
                ]
                
                for condition in conditions:
                    # Get condition name
                    condition_name = f"{'masked' if condition.mask else 'unmasked'}_{'mutated' if condition.mutate else 'unmutated'}"
                    condition_dir = mutation_dir / condition_name
                    
                    # Create experiment configuration
                    config = self._create_config()
                    config.parent_path = str(condition_dir)
                    
                    # Set pipeline type based on condition
                    if condition.mask and condition.mutate:
                        config.pipeline_type = "mutate_and_mask"
                    elif condition.mask:
                        config.pipeline_type = "masking"
                    elif condition.mutate:
                        config.pipeline_type = "mutate"
                    else:
                        config.pipeline_type = "default"
                    
                    # Set positions and cols for masking
                    if condition.mask:
                        config.positions = [pos]
                        config.cols = [pos]
                    
                    # Initialize job manager for this condition
                    condition_slurm_config = SlurmJobConfig(
                        partition=self.slurm_config.partition,
                        gpu_type=self.slurm_config.gpu_type,
                        setup_path=self.slurm_config.setup_path,
                        container_path=self.slurm_config.container_path,
                        schema_path=self.slurm_config.schema_path
                    )
                    job_manager = SlurmJobManager(
                        experiment_config=config,
                        slurm_config=condition_slurm_config,  # Use our config with correct partition/GPU
                        working_dir=str(condition_dir),
                        job_name=f"frustra_pos_{pos}_{condition_name}"
                    )
                    
                    # Submit the job
                    success, failed_jobs = job_manager.run_experiment()
                    if not success:
                        logger.error(f"Failed to submit job for position {pos} condition {condition_name}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit Frustra experiment: {str(e)}")
            return False