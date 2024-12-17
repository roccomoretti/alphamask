import os
import yaml
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from .logging import setup_logger
from .types import ExperimentConfig, MaskingStrategy
from .params import load_defaults

logger = setup_logger()

# Load defaults at module level
DEFAULTS = load_defaults()

class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

@dataclass
class SlurmJobConfig:
    time: str = "24:00:00"
    memory: str = "300000"
    cpus_per_task: int = 1
    partition: str = "clara"
    gpu_type: str = "rtx2080ti"
    gpu_count: int = 1
    email: Optional[str] = None
    container_path: str = ""
    script_path: str = ""
    schema_path: str = ""
    setup_path: Optional[str] = None

    def __post_init__(self):
        """Set default setup base path and validate GPU configuration"""
        if self.setup_path is None:
            self.setup_path = os.path.expanduser("~/alphamask_setup")
        
        # Validate GPU configuration based on partition
        if self.partition == "clara":
            valid_gpus = ["rtx2080ti", "v100"]
            if self.gpu_type not in valid_gpus:
                logger.warning(f"GPU type {self.gpu_type} not available on partition {self.partition}. Using rtx2080ti.")
                self.gpu_type = "rtx2080ti"
        elif self.partition == "paula":
            valid_gpus = ["a30"]
            if self.gpu_type not in valid_gpus:
                logger.warning(f"GPU type {self.gpu_type} not available on partition {self.partition}. Using a30.")
                self.gpu_type = "a30"

    def get_gpu_constraint(self) -> str:
        """Get the correct GPU constraint string based on partition and GPU type"""
        if self.partition == "clara":
            return f"gpu:rtx2080ti:{self.gpu_count}"
        elif self.partition == "paula":
            return f"gpu:a30:{self.gpu_count}"
        return f"gpu:{self.gpu_type}:{self.gpu_count}"

class SlurmJob:
    def __init__(self, job_id: int, name: str, output_file: str, error_file: str):
        self.job_id = job_id
        self.name = name
        self.output_file = output_file
        self.error_file = error_file
        self.status = JobStatus.PENDING
        self.start_time = None
        self.end_time = None

    def update_status(self) -> JobStatus:
        """Update job status using sacct command"""
        try:
            cmd = f"sacct -j {self.job_id} --format=State --noheader"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            state = result.stdout.strip().upper()
            
            if "PENDING" in state:
                self.status = JobStatus.PENDING
            elif "RUNNING" in state:
                self.status = JobStatus.RUNNING
            elif "COMPLETED" in state:
                self.status = JobStatus.COMPLETED
            elif any(s in state for s in ["FAILED", "CANCELLED", "NODE_FAIL"]):
                self.status = JobStatus.FAILED
            elif "TIMEOUT" in state:
                self.status = JobStatus.TIMEOUT
            else:
                self.status = JobStatus.UNKNOWN
                
            return self.status
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
            return JobStatus.UNKNOWN

class SlurmJobManager:
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        slurm_config: SlurmJobConfig,
        working_dir: Optional[str] = None
    ):
        self.experiment_config = experiment_config
        self.slurm_config = slurm_config
        # Resolve working directory to absolute path
        self.working_dir = Path(working_dir if working_dir else ".").resolve()
        self.jobs: Dict[int, SlurmJob] = {}
        self.failed_jobs: List[int] = []
        self.has_slurm = self._check_slurm_available()
        
        # Set setup base path in experiment config
        self.experiment_config.setup_path = self.slurm_config.setup_path
        
        # Store schema path
        self.schema_path = self.slurm_config.schema_path
        
        if not self.has_slurm:
            logger.warning("SLURM not detected - jobs will run sequentially")
        
        # Create necessary directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for job management"""
        # Create subdirectories in working directory
        (self.working_dir / "configs").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Store paths for later use
        self.config_dir = self.working_dir / "configs"
        self.script_dir = self.working_dir / "scripts"
        self.log_dir = self.working_dir / "logs"
        
        # Log directory structure
        logger.info(f"Created directory structure:")
        logger.info(f"Working directory: {self.working_dir}")
        logger.info(f"Config directory: {self.config_dir}")
        logger.info(f"Script directory: {self.script_dir}")
        logger.info(f"Log directory: {self.log_dir}")

    def create_job_script(self, config_path: str, job_name: str) -> str:
        """Create a SLURM job script for the given configuration.

        Args:
            config_path (str): Path to the configuration file
            job_name (str): Name of the job

        Returns:
            str: Path to the created job script
        """
        try:
            # Convert all paths to absolute paths
            config_path = str(Path(config_path).resolve())
            container_path = str(Path(self.slurm_config.container_path).resolve())
            script_path = str(Path(self.slurm_config.script_path).resolve())
            working_dir = str(Path(self.working_dir).resolve())
            
            # Create script path
            script_path_obj = Path(self.script_dir) / f"{job_name}.sh"
            script_path_abs = str(script_path_obj.resolve())
            
            # Ensure all required directories exist
            for dir_path in [
                self.working_dir,
                self.config_dir,
                self.script_dir,
                self.log_dir,
                Path(self.working_dir) / "in",
                Path(self.working_dir) / "in" / "msa",
                Path(self.working_dir) / "out",
                Path(self.working_dir) / "out" / "pdbs",
                Path(self.working_dir) / "schema"
            ]:
                dir_path = Path(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensuring directory exists: {dir_path}")
            
            # Get local schema path and ensure schema directory exists
            schema_dir = Path(self.working_dir) / "schema"
            local_schema_path = schema_dir / "schema_validation.json"
            
            # Copy schema file if it exists
            if Path(self.schema_path).exists():
                shutil.copy2(self.schema_path, local_schema_path)
                logger.info(f"Copied schema from {self.schema_path} to {local_schema_path}")
            else:
                logger.warning(f"Schema file not found at {self.schema_path}")
                local_schema_path.touch()
                logger.info(f"Created empty schema file at {local_schema_path}")

            # Verify required files exist
            required_files = {
                "Container": container_path,
                "Script": script_path,
                "Config": config_path
            }
            
            for name, path in required_files.items():
                if not Path(path).exists():
                    raise FileNotFoundError(f"{name} not found at {path}")
                logger.info(f"Verified {name} exists at {path}")

            # Create script content with absolute paths
            script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={str(Path(self.log_dir) / f"{job_name}.out")}
#SBATCH --error={str(Path(self.log_dir) / f"{job_name}.err")}
#SBATCH --time=24:00:00
#SBATCH --mem=300000
#SBATCH --cpus-per-task=1
#SBATCH --partition={self.slurm_config.partition}
#SBATCH --gres=gpu:{self.slurm_config.gpu_type}:1

# Load required modules if needed

# Print paths for debugging
echo "Working directory: {working_dir}"
echo "Config path: {config_path}"
echo "Script path: {script_path}"
echo "Schema path: {local_schema_path}"

# Create required directories
mkdir -p {working_dir}/in/msa
mkdir -p {working_dir}/out/pdbs
mkdir -p {working_dir}/schema

cd {working_dir}

# Debug commands to verify paths and permissions
echo "Listing working directory contents:"
ls -la
echo "Listing input directory contents:"
ls -la in/
echo "Checking MSA file:"
ls -la in/msa.a3m || echo "MSA file not found"
echo "Checking parent directories:"
ls -la ..
echo "Current directory structure:"
pwd
find . -type f -name "msa.a3m"

# Verify paths exist
if [ ! -f "{config_path}" ]; then
    echo "Error: Config file not found at {config_path}"
    exit 1
fi

if [ ! -f "{local_schema_path}" ]; then
    echo "Error: Schema file not found at {local_schema_path}"
    exit 1
fi

if [ ! -f "{container_path}" ]; then
    echo "Error: Container not found at {container_path}"
    exit 1
fi

if [ ! -f "{script_path}" ]; then
    echo "Error: Script not found at {script_path}"
    exit 1
fi

# Function to wait for file to be fully written
wait_for_file() {{
    local file="$1"
    local timeout=3600  # 1 hour timeout
    local start_time=$(date +%s)
    
    while true; do
        if [ -f "$file" ]; then
            # Check if file size is stable (no writes for 5 seconds)
            local size1=$(stat -c %s "$file")
            sleep 5
            local size2=$(stat -c %s "$file")
            if [ "$size1" = "$size2" ]; then
                return 0
            fi
        fi
        
        # Check timeout
        local current_time=$(date +%s)
        if [ $((current_time - start_time)) -gt $timeout ]; then
            echo "Timeout waiting for $file to be fully written"
            return 1
        fi
        
        sleep 10
    done
}}

# If this is a masking job and using custom MSA, wait for the MSA file
if [ "{getattr(self.experiment_config, 'msa_method', 'mmseqs2')}" = "custom_a3m" ] && [ -n "{getattr(self.experiment_config, 'custom_a3m_path', '')}" ]; then
    echo "Waiting for MSA file to be fully written: {getattr(self.experiment_config, 'custom_a3m_path', '')}"
    if ! wait_for_file "{getattr(self.experiment_config, 'custom_a3m_path', '')}"; then
        echo "Error: Failed to get MSA file"
        exit 1
    fi
    echo "MSA file is ready"
fi

# Run the prediction script using singularity
singularity exec --nv --cleanenv \\
    -B /work:/work \\
    -B {working_dir}:{working_dir} \\
    {container_path} \\
    python {script_path} \\
    --yaml_file {config_path} \\
    --json_schema {local_schema_path} \\
    --pipeline {self.experiment_config.pipeline_type or "default"}
"""
            
            # Write script to file
            with open(script_path_abs, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            Path(script_path_abs).chmod(0o755)
            
            # Log script creation
            logger.info(f"Created job script at: {script_path_abs}")
            logger.info(f"Generated SLURM script for {job_name}:")
            logger.info(script_content)
            
            return script_path_abs
            
        except Exception as e:
            logger.error(f"Error creating job script: {str(e)}")
            raise

    def submit_job(self, config_path: str, job_name: str) -> Optional[int]:
        """Submit a job either to SLURM or run it directly"""
        try:
            # Log input parameters
            logger.info(f"Submitting job with config_path: {config_path}, job_name: {job_name}")
            logger.info(f"Working directory: {self.working_dir}")
            logger.info(f"Config file exists: {Path(config_path).exists()}")
            
            # Create the job script first
            script_path = self.create_job_script(config_path, job_name)
            logger.info(f"Created job script at: {script_path}")
            logger.info(f"Job script exists: {Path(script_path).exists()}")
            
            if self.has_slurm:
                # Submit to SLURM
                return self._submit_slurm_job(script_path)
            else:
                # Run locally
                return self._run_job_locally(script_path)
            
        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error traceback: ", exc_info=True)
            return None

    def _submit_slurm_job(self, script_path: str) -> Optional[int]:
        """Submit job to SLURM"""
        try:
            # Check if sbatch exists and get its path
            sbatch_path = shutil.which('sbatch')
            logger.info(f"sbatch command path: {sbatch_path}")
            
            # Ensure script directory exists
            script_dir = Path(script_path).parent
            logger.info(f"Ensuring script directory exists: {script_dir}")
            script_dir.mkdir(parents=True, exist_ok=True)
            
            # Log file existence and permissions
            logger.info(f"Script path exists: {Path(script_path).exists()}")
            if Path(script_path).exists():
                logger.info(f"Script permissions: {oct(Path(script_path).stat().st_mode)}")
            
            # Check if working directory exists before changing to it
            if not Path(self.working_dir).exists():
                logger.error(f"Working directory does not exist: {self.working_dir}")
                return None
            
            # Try to get current directory, use /tmp as fallback
            try:
                original_dir = os.getcwd()
            except FileNotFoundError:
                original_dir = "/tmp"
                logger.warning(f"Current directory not accessible, using fallback: {original_dir}")
            
            logger.info(f"Current directory before cd: {original_dir}")
            os.chdir(str(self.working_dir))
            logger.info(f"Changed to working directory: {os.getcwd()}")
            
            # Log environment PATH
            logger.info(f"PATH environment variable: {os.environ.get('PATH', 'Not found')}")
            
            # Submit job
            cmd = f"sbatch {script_path}"
            logger.info(f"Submitting SLURM job with command: {cmd}")
            logger.info(f"Full script path: {os.path.abspath(script_path)}")
            
            # Try to execute sbatch directly with full path if available
            if sbatch_path:
                cmd = f"{sbatch_path} {script_path}"
            
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            # Try to change back to original directory, but don't fail if we can't
            try:
                os.chdir(original_dir)
            except (FileNotFoundError, PermissionError) as e:
                logger.warning(f"Could not change back to original directory: {e}")
            
            # Check result
            if result.returncode == 0:
                # Parse job ID from output
                job_id = int(result.stdout.strip().split()[-1])
                logger.info(f"Successfully submitted job {job_id}")
                
                # Log SLURM partition status
                # try:
                #     sinfo_result = subprocess.run(["sinfo"], capture_output=True, text=True)
                #     logger.info("Current SLURM partition status:")
                #     logger.info(sinfo_result.stdout)
                # except Exception as e:
                #     logger.warning(f"Could not get SLURM partition status: {e}")
                
                return job_id
            else:
                logger.error(f"Failed to submit SLURM job")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None
            
        except Exception as e:
            logger.error(f"Error submitting job: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error traceback:", exc_info=True)
            return None

    def _run_job_locally(self, script_path: str) -> Optional[int]:
        """Run job locally using bash"""
        try:
            logger.info(f"Running job locally")
            
            # Change to working directory
            original_dir = os.getcwd()
            os.chdir(str(self.working_dir))
            
            # Run the script
            cmd = f"bash {script_path}"
            logger.info(f"Running command: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            # Change back to original directory
            os.chdir(original_dir)
            
            # Check result
            if result.returncode == 0:
                logger.info("Successfully completed local job")
                return 0  # Return 0 as pseudo job ID for local runs
            else:
                logger.error(f"Failed to run job locally")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None
            
        except Exception as e:
            logger.error(f"Error running job locally: {str(e)}")
            return None

    def submit_iterative_masking_jobs(self) -> List[int]:
        """Submit jobs for iterative masking experiments"""
        job_ids = []
        sequence_length = len(self.experiment_config.sequence)
        
        # Create configs directory if it doesn't exist
        configs_dir = self.working_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        
        for pos in range(1, sequence_length + 1):
            # Create config for this position
            config = self.experiment_config.to_dict()
            config["cols"] = [pos]
            # Set masking configuration
            config["masking_mode"] = "list"  # Use list mode for single position masking
            config["mask_msa"] = True
            config["mask_deletion_matrix"] = True
            config["debug"] = DEFAULTS.get('debug', False)
            config["pipeline_type"] = "masking"  # Ensure masking pipeline is used
            config["seed"] = DEFAULTS.get('seed', 0)
            config["num_seeds"] = DEFAULTS.get('num_seeds', 2)  # Use default from defaults.yaml
            
            config_path = configs_dir / f"config_pos_{pos}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            job_id = self.submit_job(
                str(config_path),
                f"{self.experiment_config.jobname_prefix}_pos_{pos}"
            )
            
            if job_id:
                job_ids.append(job_id)
        
        return job_ids

    def submit_iterative_mask_mutate_jobs(self) -> List[int]:
        """Submit jobs for iterative masking and mutation experiments"""
        job_ids = []
        positions = self.experiment_config.positions or range(1, len(self.experiment_config.sequence) + 1)
        
        for pos in positions:
            # Create config for this position
            config = self.experiment_config.to_dict()
            config["cols"] = [pos]
            
            # Add mutations if specified
            if self.experiment_config.mutations:
                config["mutations"] = self.experiment_config.mutations
                mutations_str = "_".join(self.experiment_config.mutations)
            else:
                mutations_str = str(pos)
            
            config_path = self.working_dir / "configs" / f"config_pos_{pos}_mut_{mutations_str}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            job_id = self.submit_job(
                str(config_path),
                f"{self.experiment_config.jobname_prefix}_pos_{pos}_mut_{mutations_str}"
            )
            
            if job_id:
                job_ids.append(job_id)
        
        return job_ids

    def submit_position_masking_jobs(self) -> List[int]:
        """Submit position masking jobs"""
        if not self.experiment_config.positions:
            raise ValueError("Positions must be provided for position-based masking")
        
        job_ids = []
        
        # Create masking config
        masking_config = self.experiment_config.copy()
        masking_config.jobname = f"{masking_config.jobname_prefix}_masked"
        masking_config.parent_path = str(self.working_dir)
        
        # Create necessary directories
        msa_dir = Path(self.working_dir) / "in" / "msa"
        pdb_dir = Path(self.working_dir) / "out" / "pdbs"
        msa_dir.mkdir(parents=True, exist_ok=True)
        pdb_dir.mkdir(parents=True, exist_ok=True)
        
        # Save masking config
        config_path = Path(self.working_dir) / "configs" / f"config_masked_{masking_config.jobname_prefix}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(masking_config.to_dict(), f)
        
        logger.info(f"Created masking config at: {config_path}")
        logger.info(f"Using model parameters from: {masking_config.get_setup_path() / 'params'}")
        logger.info(f"Created MSA directory at: {msa_dir}")
        logger.info(f"Created PDB directory at: {pdb_dir}")
        
        # Submit job
        job_id = self.submit_job(str(config_path), masking_config.jobname)
        if job_id:
            logger.info(f"Submitted masking job with ID {job_id}")
            job_ids.append(job_id)
        
        return job_ids

    def submit_control_job(self) -> Optional[int]:
        """Submit control job"""
        if not self.experiment_config.run_control:
            return None
        
        # Create control config
        control_config = self.experiment_config.copy()
        control_config.jobname = f"{control_config.jobname_prefix}_vanilla_control"
        control_config.parent_path = str(self.working_dir)  # Set parent_path to working directory
        
        # Create necessary directories
        input_dir = Path(self.working_dir) / "in"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the existing config path structure
        config_path = Path(self.working_dir) / "configs" / "config_control.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict and ensure setup_path is set
        config_dict = control_config.to_dict()
        if 'setup_path' not in config_dict or not config_dict['setup_path']:
            config_dict['setup_path'] = str(control_config.get_setup_path())
        
        # Save control config
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        
        logger.info(f"Created control config at: {config_path}")
        logger.info(f"Using model parameters from: {control_config.get_setup_path() / 'params'}")
        
        # Submit job
        job_id = self.submit_job(str(config_path), control_config.jobname)
        if job_id:
            logger.info(f"Submitted control job with ID {job_id}")
            return job_id
        return None

    def submit_masking_jobs(self) -> List[int]:
        """Submit jobs based on masking strategy"""
        if self.experiment_config.masking_strategy == MaskingStrategy.NONE:
            # For no masking, just submit a single job with default settings
            config = self.experiment_config.to_dict()
            config["pipeline_type"] = "default"  # Use default pipeline for no masking
            
            config_path = self.working_dir / "configs" / "config_no_masking.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            job_id = self.submit_job(
                str(config_path),
                f"{self.experiment_config.jobname_prefix}_no_masking"
            )
            return [job_id] if job_id else []
            
        elif self.experiment_config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE:
            return self.submit_iterative_masking_jobs()
        elif self.experiment_config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE_MASK_MUTATE:
            return self.submit_iterative_mask_mutate_jobs()
        elif self.experiment_config.masking_strategy == MaskingStrategy.MUTATE_AND_MASK:
            # For MUTATE_AND_MASK, we submit a single job with the mutations and masking
            config = self.experiment_config.to_dict()
            config["pipeline_type"] = "mutate_and_mask"  # Ensure correct pipeline type
            
            config_path = self.working_dir / "configs" / "config_mutate_and_mask.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            job_id = self.submit_job(
                str(config_path),
                f"{self.experiment_config.jobname_prefix}_mutate_and_mask"
            )
            return [job_id] if job_id else []
        
        elif self.experiment_config.masking_strategy in [MaskingStrategy.MASK_POSITIONS, MaskingStrategy.UNMASK_POSITIONS]:
            return self.submit_position_masking_jobs()
        else:
            raise ValueError(f"Unsupported masking strategy: {self.experiment_config.masking_strategy}")

    def monitor_jobs(self, check_interval: int = 60) -> bool:
        """Monitor jobs until completion"""
        if not self.has_slurm:
            # For local runs, jobs are already complete when they return
            return True
            
        all_completed = False
        
        while not all_completed:
            all_completed = True
            for job_id, job in self.jobs.items():
                status = job.update_status()
                
                if status == JobStatus.FAILED:
                    self.failed_jobs.append(job_id)
                    logger.error(f"Job {job.name} (ID: {job_id}) failed!")
                
                if status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    all_completed = False
            
            if not all_completed:
                time.sleep(check_interval)
        
        return len(self.failed_jobs) == 0

    def get_failed_jobs_info(self) -> List[Dict]:
        """Get detailed information about failed jobs"""
        failed_jobs_info = []
        for job_id in self.failed_jobs:
            job = self.jobs[job_id]
            
            # Read error file if it exists
            error_content = ""
            if os.path.exists(job.error_file):
                with open(job.error_file, 'r') as f:
                    error_content = f.read()
            
            failed_jobs_info.append({
                "job_id": job_id,
                "name": job.name,
                "error_file": job.error_file,
                "error_content": error_content
            })
        
        return failed_jobs_info

    def resubmit_failed_jobs(self) -> List[int]:
        """Resubmit failed jobs"""
        new_job_ids = []
        for job_id in self.failed_jobs:
            old_job = self.jobs[job_id]
            
            # Extract position from job name
            pos = int(old_job.name.split('_')[-1])
            
            # Create new config
            config = self.experiment_config.to_dict()
            config["cols"] = [pos]
            
            config_path = self.working_dir / "configs" / f"config_pos_{pos}_retry.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            new_job_id = self.submit_job(
                str(config_path),
                f"{old_job.name}_retry"
            )
            
            if new_job_id:
                new_job_ids.append(new_job_id)
        
        return new_job_ids 

    def _check_slurm_available(self) -> bool:
        """Check if SLURM is available in the system"""
        return (
            shutil.which('sbatch') is not None and 
            shutil.which('squeue') is not None and
            shutil.which('sacct') is not None
        )

    def run_experiment(self) -> Tuple[bool, Optional[List[Dict]]]:
        """
        Run the complete experiment workflow
        
        Returns:
            Tuple[bool, Optional[List[Dict]]]: (success, failed_jobs_info)
        """
        if self.has_slurm:
            # Submit control job if needed
            if self.experiment_config.run_control or self.experiment_config.run_only_control:
                control_job_id = self.submit_control_job()
                if control_job_id is None:
                    logger.error("Failed to submit control job")
                    return False, None
            
            # Submit masking jobs if not running only control
            if not self.experiment_config.run_only_control:
                job_ids = self.submit_masking_jobs()
                if not job_ids:
                    logger.error("Failed to submit masking jobs")
                    return False, None
            
            # Monitor all jobs
            success = self.monitor_jobs()
            failed_jobs_info = self.get_failed_jobs_info() if not success else None
            
            return success, failed_jobs_info
        else:
            # For local runs, execute directly
            try:
                job_id = self.submit_job(
                    str(self.working_dir / "config.yaml"),
                    self.experiment_config.jobname_prefix
                )
                return job_id is not None, None
            except Exception as e:
                logger.error(f"Error running experiment locally: {e}")
                return False, [{"error": str(e)}]