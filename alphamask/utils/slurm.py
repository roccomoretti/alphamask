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

from .types import ExperimentConfig
from .params import load_defaults

# Get logger for this module
logger = logging.getLogger("alphamask.utils.slurm")

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
    setup_commands: List[str] = None

    def __post_init__(self):
        """Set default setup base path and validate GPU configuration"""
        if self.setup_path is None:
            self.setup_path = os.path.expanduser("~/alphamask_setup")
        
        if self.setup_commands is None:
            self.setup_commands = []
        
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
        
        if not self.schema_path:
            raise ValueError("Schema path must be provided in SlurmJobConfig")
        schema_path = Path(self.schema_path)
        if not schema_path.exists():
            raise ValueError(f"Schema file not found: {schema_path}")
        if not schema_path.is_file():
            raise ValueError(f"Schema path is not a file: {schema_path}")
        self.schema_path = str(schema_path.resolve())

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
        working_dir: str,
        job_name: str
    ):
        self.experiment_config = experiment_config
        self.slurm_config = slurm_config
        self.working_dir = Path(working_dir)
        self.job_name = job_name
        self.jobs: Dict[int, SlurmJob] = {}
        self.failed_jobs: List[int] = []
        self.has_slurm = self._check_slurm_available()
        
        # Set setup base path in experiment config
        self.experiment_config.setup_path = self.slurm_config.setup_path
        
        # Store schema path
        schema_path = Path(self.slurm_config.schema_path)
        if not schema_path.is_absolute():
            schema_path = schema_path.resolve()
        self.schema_path = schema_path
        
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
        logger.debug(f"Created directory structure:")
        logger.debug(f"Working directory: {self.working_dir}")
        logger.debug(f"Config directory: {self.config_dir}")
        logger.debug(f"Script directory: {self.script_dir}")
        logger.debug(f"Log directory: {self.log_dir}")

    def create_job_script(self, config_path: str, job_name: str) -> str:
        """Create a SLURM job script for the given configuration.

        Args:
            config_path (str): Path to the configuration file
            job_name (str): Name of the job

        Returns:
            str: Path to the created job script
        """
        try:
            # Handle MSA if needed
            if self.experiment_config.msa_method == "mmseqs2":
                self._handle_msa()
            
            # Create script with existing config
            script_path = self._generate_script(config_path, job_name)
            return script_path
            
        except Exception as e:
            logger.error(f"Error creating job script: {str(e)}")
            raise

    def _handle_msa(self):
        """Handle MSA generation and sharing for apriori experiments"""
        if "apriori" not in str(self.working_dir):
            return
        
        # Get the protein base directory (e.g., /work/.../my_experiments/her2)
        protein_dir = Path(self.working_dir).parent.parent
        
        # Create shared MSA directory at protein level
        shared_msa_dir = protein_dir / "in" / "msa"
        shared_msa_dir.mkdir(parents=True, exist_ok=True)
        
        # Define MSA and flag file paths
        msa_file = shared_msa_dir / "msa.a3m"
        flag_file = shared_msa_dir / ".msa_in_progress"
        
        if msa_file.exists():
            logger.debug(f"MSA file exists at {msa_file}, using it directly.")
            self.experiment_config.msa_method = "custom_a3m"
            self.experiment_config.custom_a3m_path = str(msa_file)
            return
        
        # Wait for MSA if another job is generating it
        if flag_file.exists():
            self._wait_for_msa(msa_file, flag_file)
        else:
            self._generate_msa(msa_file, flag_file, shared_msa_dir)

    def _wait_for_msa(self, msa_file: Path, flag_file: Path):
        """Wait for MSA generation to complete"""
        logger.debug("Waiting for MSA generation to complete...")
        wait_start = time.time()
        timeout = 3600  # 1 hour timeout
        
        while flag_file.exists():
            if msa_file.exists():
                try:
                    flag_file.unlink()
                    logger.debug("MSA found and flag file removed.")
                    break
                except Exception as e:
                    logger.warning(f"Could not remove flag file: {e}")
            
            if time.time() - wait_start > timeout:
                raise TimeoutError("MSA generation timed out")
            
            time.sleep(10)
        
        if not msa_file.exists():
            raise FileNotFoundError("MSA file not found after waiting")
        
        logger.debug("MSA generation completed.")
        self.experiment_config.msa_method = "custom_a3m"
        self.experiment_config.custom_a3m_path = str(msa_file)

    def _generate_msa(self, msa_file: Path, flag_file: Path, shared_msa_dir: Path):
        """Generate MSA for the experiment"""
        flag_file.touch()
        logger.debug(f"Created MSA in-progress flag at {flag_file}")
        
        # Initialize PrepInputs for MSA generation
        from alphamask.core.msa import PrepInputs
        prep_inputs = PrepInputs(
            sequence=self.experiment_config.sequence,
            jobname="msa",  # Use generic name to avoid job-specific directories
            msa_method="mmseqs2",
            setup_path=self.slurm_config.setup_path,
            parent_path=shared_msa_dir,  # Use shared MSA dir directly
            overwrite=False
        )
        
        # Process sequence and generate MSA
        prep_inputs.process_sequence()
        msa_jobname = prep_inputs.jobname
        logger.debug(f"Using MSA jobname: {msa_jobname}")
        
        prep_inputs.get_msa()
        
        # Find and move the generated MSA
        generated_msa = shared_msa_dir / msa_jobname / "in" / "msa.a3m"
        if generated_msa.exists():
            shutil.copy2(generated_msa, msa_file)
            logger.debug(f"Copied MSA from {generated_msa} to {msa_file}")
            shutil.rmtree(generated_msa.parent.parent, ignore_errors=True)
        else:
            logger.error(f"Generated MSA not found at expected path: {generated_msa}")
            raise FileNotFoundError(f"MSA file not found at {generated_msa}")
        
        # Update experiment config
        self.experiment_config.msa_method = "custom_a3m"
        self.experiment_config.custom_a3m_path = str(msa_file)

    def _generate_script(self, config_path: str, job_name: str) -> str:
        """Generate the actual SLURM script"""
        # Convert all paths to absolute paths
        config_path = str(Path(config_path).resolve())
        container_path = str(Path(self.slurm_config.container_path).resolve())
        working_dir = str(Path(self.working_dir).resolve())
        
        # Create script path
        script_path_obj = Path(self.script_dir) / f"{job_name}.sh"
        script_path_abs = str(script_path_obj.resolve())
        
        # Copy schema file to local directory
        schema_dir = Path(self.working_dir) / "schema"
        schema_dir.mkdir(parents=True, exist_ok=True)
        local_schema_path = schema_dir / "schema_validation.json"
        
        # Ensure source schema exists and is readable
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        if not os.access(self.schema_path, os.R_OK):
            raise PermissionError(f"Cannot read schema file: {self.schema_path}")
        
        try:
            shutil.copy2(self.schema_path, local_schema_path)
            logger.debug(f"Copied schema from {self.schema_path} to {local_schema_path}")
        except (shutil.Error, IOError) as e:
            raise RuntimeError(f"Failed to copy schema file: {e}")

        # Create script content
        script_content = self._get_script_content(
            job_name=job_name,
            working_dir=working_dir,
            config_path=config_path,
            local_schema_path=str(local_schema_path),
            container_path=container_path
        )
        
        # Write script to file
        with open(script_path_abs, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        Path(script_path_abs).chmod(0o755)
        
        logger.debug(f"Created job script at: {script_path_abs}")
        return script_path_abs

    def _get_script_content(self, job_name: str, working_dir: str, config_path: str, 
                           local_schema_path: str, container_path: str) -> str:
        """Generate the content of the SLURM script"""
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={str(Path(self.log_dir) / f"{job_name}.out")}
#SBATCH --error={str(Path(self.log_dir) / f"{job_name}.err")}
#SBATCH --time=24:00:00
#SBATCH --mem=300000
#SBATCH --cpus-per-task=1
#SBATCH --partition={self.slurm_config.partition}
#SBATCH --gres=gpu:{self.slurm_config.gpu_type}:1

# Load required modules and initialize conda
module load Anaconda3
eval "$(conda shell.bash hook)"
source ~/.bashrc

# Check if conda environment exists
if ! conda env list | grep -q "alphamask"; then
    echo "Error: conda environment 'alphamask' not found"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Environment setup
{chr(10).join(self.slurm_config.setup_commands)}

# Add conda environment to PATH
CONDA_BASE=$(conda info --base)
CONDA_ENV_PATH="/home/sc.uni-leipzig.de/$USER/.conda/envs/alphamask"
export PATH="$CONDA_ENV_PATH/bin:$PATH"

# Print environment info
echo "Python path:"
which python
echo "Conda info:"
conda info
echo "PATH:"
echo $PATH
echo "Conda environment path:"
echo $CONDA_ENV_PATH

# Print paths for debugging
echo "Working directory: {working_dir}"
echo "Config path: {config_path}"
echo "Schema path: {local_schema_path}"

cd {working_dir}

# Debug commands to verify paths and permissions
echo "Listing working directory contents:"
ls -la
echo "Listing input directory contents:"
ls -la in/ || echo "Input directory is empty"
echo "Checking MSA file:"
ls -la in/msa.a3m 2>/dev/null || echo "MSA file not found (this is normal if not using custom MSA)"
echo "Checking parent directories:"
ls -la ..
echo "Current directory structure:"
pwd
find . -type f -name "msa.a3m" || echo "No MSA files found"

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
if [ "{self.experiment_config.msa_method}" = "custom_a3m" ] && [ -n "{getattr(self.experiment_config, 'custom_a3m_path', '')}" ]; then
    echo "Waiting for MSA file to be fully written: {getattr(self.experiment_config, 'custom_a3m_path', '')}"
    if ! wait_for_file "{getattr(self.experiment_config, 'custom_a3m_path', '')}"; then
        echo "Error: Failed to get MSA file"
        exit 1
    fi
    echo "MSA file is ready"
    
    # Remove the in-progress flag if we're using a shared MSA
    if [ -f "{getattr(self.experiment_config, 'custom_a3m_path', '')}" ]; then
        flag_file="{Path(getattr(self.experiment_config, 'custom_a3m_path', '')).parent / '.msa_in_progress'}"
        if [ -f "$flag_file" ]; then
            rm -f "$flag_file"
            echo "Removed MSA in-progress flag"
        fi
    fi
fi

# Check if conda environment exists before binding
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Error: Conda environment directory not found at $CONDA_ENV_PATH"
    echo "Current conda environments:"
    conda env list
    exit 1
fi

# Run the command using singularity
echo "Running command: {self.slurm_config.script_path} --config {config_path} --schema {local_schema_path} --pipeline {self.experiment_config.pipeline_type or 'default'}"

# Export the conda environment path inside the container
echo "Using container's built-in environment..."

singularity exec --nv \\
    -B /work:/work \\
    -B {working_dir}:{working_dir} \\
    -B /home/sc.uni-leipzig.de/$USER/github/alphamask:/opt/alphamask \\
    {container_path} \\
    ~/.conda/envs/alphamask/bin/alphamask predict-job \\
    --config {config_path} \\
    --schema {local_schema_path} \\
    --pipeline {self.experiment_config.pipeline_type or "default"}
"""

    def submit_job(self, config_path: str, job_name: str) -> Optional[int]:
        """Submit a job either to SLURM or run it directly"""
        try:
            # Log input parameters
            logger.debug(f"Submitting job with config_path: {config_path}, job_name: {job_name}")
            logger.debug(f"Working directory: {self.working_dir}")
            
            # Use the config path that was passed in
            config_path = Path(config_path)
            
            # Create the job script
            script_path = self.create_job_script(str(config_path), job_name)
            logger.debug(f"Created job script at: {script_path}")

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
            logger.debug(f"sbatch command path: {sbatch_path}")
            
            # Ensure script directory exists
            script_dir = Path(script_path).parent
            logger.debug(f"Ensuring script directory exists: {script_dir}")
            script_dir.mkdir(parents=True, exist_ok=True)
            
            # Log file existence and permissions
            logger.debug(f"Script path exists: {Path(script_path).exists()}")
            if Path(script_path).exists():
                logger.debug(f"Script permissions: {oct(Path(script_path).stat().st_mode)}")
            
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
            
            logger.debug(f"Current directory before cd: {original_dir}")
            os.chdir(str(self.working_dir))
            logger.debug(f"Changed to working directory: {os.getcwd()}")
            
            # Log environment PATH
            logger.debug(f"PATH environment variable: {os.environ.get('PATH', 'Not found')}")
            
            # Submit job
            cmd = f"sbatch {script_path}"
            logger.debug(f"Submitting SLURM job with command: {cmd}")
            logger.debug(f"Full script path: {os.path.abspath(script_path)}")
            
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
                logger.debug(f"Successfully submitted job {job_id}")
                
                # Log SLURM partition status
                # try:
                #     sinfo_result = subprocess.run(["sinfo"], capture_output=True, text=True)
                #     logger.debug("Current SLURM partition status:")
                #     logger.debug(sinfo_result.stdout)
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
            logger.debug(f"Running job locally")
            
            # Change to working directory
            original_dir = os.getcwd()
            os.chdir(str(self.working_dir))
            
            # Run the script
            cmd = f"bash {script_path}"
            logger.debug(f"Running command: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            # Change back to original directory
            os.chdir(original_dir)
            
            # Check result
            if result.returncode == 0:
                logger.debug("Successfully completed local job")
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
        
        # Get MSA from WT directory if it exists
        wt_msa_path = self.working_dir.parent / "WT" / "in" / "msa" / "msa.a3m"
        if wt_msa_path.exists():
            logger.debug(f"Using WT MSA from: {wt_msa_path}")
            self.experiment_config.msa_method = "custom_a3m"
            self.experiment_config.custom_a3m_path = str(wt_msa_path)
        
        for pos in range(1, sequence_length + 1):
            # Create config for this position
            config = self.experiment_config.to_dict()
            config["cols"] = [pos]  # Mask one position at a time
            config["positions"] = [pos]  # Also set positions for consistency
            
            # Set masking configuration
            config["masking_mode"] = "list"
            config["mask_msa"] = True
            config["mask_deletion_matrix"] = True
            config["mask_token"] = "X"
            config["pipeline_type"] = "masking"
            
            # Set other parameters
            config["num_seeds"] = self.defaults.get('num_seeds', 2)
            config["num_recycles"] = self.defaults.get('num_recycles', 2)
            
            # Create unique job name for this position
            job_name = f"{self.experiment_config.jobname_prefix}_pos_{pos}"
            
            # Save config
            config_path = configs_dir / f"config_pos_{pos}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Submit job
            job_id = self.submit_job(str(config_path), job_name)
            if job_id:
                job_ids.append(job_id)
                logger.debug(f"Submitted masking job for position {pos} with ID {job_id}")
            else:
                logger.error(f"Failed to submit job for position {pos}")
        
        return job_ids

    def submit_wt_job(self) -> Optional[int]:
        """Submit wild-type prediction job"""
        try:
            # Create config directory
            configs_dir = self.working_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create WT config
            config = self.experiment_config.to_dict()
            config["pipeline_type"] = "default"
            config["jobname_prefix"] = "WT"
            
            # Save config
            config_path = configs_dir / "config_wt.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Submit job
            job_id = self.submit_job(str(config_path), "WT")
            if job_id:
                logger.debug(f"Submitted WT job with ID {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting WT job: {str(e)}")
            return None

    def submit_mutation_job(self, mutations: List[str]) -> Optional[int]:
        """Submit job for a specific mutation or set of mutations"""
        try:
            # Create config directory
            configs_dir = self.working_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            
            # Get WT MSA path
            wt_msa_path = self.working_dir.parent / "WT" / "in" / "msa" / "msa.a3m"
            
            # Create mutation config
            config = self.experiment_config.to_dict()
            config["pipeline_type"] = "mutate"
            config["mutations"] = mutations
            config["jobname_prefix"] = "_".join(mutations)
            
            # Use WT MSA if available
            if wt_msa_path.exists():
                config["msa_method"] = "custom_a3m"
                config["custom_a3m_path"] = str(wt_msa_path)
            
            # Save config
            config_path = configs_dir / f"config_{'_'.join(mutations)}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Submit job
            job_name = f"mutation_{'_'.join(mutations)}"
            job_id = self.submit_job(str(config_path), job_name)
            if job_id:
                logger.debug(f"Submitted mutation job {job_name} with ID {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error submitting mutation job: {str(e)}")
            return None

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
        # pdb_dir = Path(self.working_dir) / "out" / "pdbs"
        msa_dir.mkdir(parents=True, exist_ok=True)
        # pdb_dir.mkdir(parents=True, exist_ok=True)
        
        # Save masking config
        config_path = Path(self.working_dir) / "configs" / f"config_masked_{masking_config.jobname_prefix}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(masking_config.to_dict(), f)
        
        logger.debug(f"Created masking config at: {config_path}")
        logger.debug(f"Using model parameters from: {masking_config.get_setup_path() / 'params'}")
        logger.debug(f"Created MSA directory at: {msa_dir}")
        # logger.debug(f"Created PDB directory at: {pdb_dir}")
        
        # Submit job
        job_id = self.submit_job(str(config_path), masking_config.jobname)
        if job_id:
            logger.debug(f"Submitted masking job with ID {job_id}")
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
        
        logger.debug(f"Created control config at: {config_path}")
        logger.debug(f"Using model parameters from: {control_config.get_setup_path() / 'params'}")
        
        # Submit job
        job_id = self.submit_job(str(config_path), control_config.jobname)
        if job_id:
            logger.debug(f"Submitted control job with ID {job_id}")
            return job_id
        return None

    def submit_masking_jobs(self) -> List[int]:
        """Submit jobs based on pipeline type"""
        # For no masking, just submit a single job with default settings
        if self.experiment_config.pipeline_type == "default":
            config = self.experiment_config.to_dict()
            config["pipeline_type"] = "default"  # Use default pipeline for no masking
            config["schema_path"] = str(self.schema_path)  # Add schema path to config
            
            config_path = self.working_dir / "configs" / "config_no_masking.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            job_id = self.submit_job(
                str(config_path),
                f"{self.experiment_config.jobname_prefix}_no_masking"
            )
            return [job_id] if job_id else []
            
        elif self.experiment_config.pipeline_type == "masking":
            return self.submit_iterative_masking_jobs()
        elif self.experiment_config.pipeline_type == "mutate":
            return self.submit_iterative_mask_mutate_jobs()
        elif self.experiment_config.pipeline_type == "mutate_and_mask":
            # For mutate_and_mask, we submit a single job with the mutations and masking
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
        else:
            raise ValueError(f"Unsupported pipeline type: {self.experiment_config.pipeline_type}")

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

    def run_experiment(self) -> Tuple[bool, List[str]]:
        """Run the experiment based on configuration type"""
        try:
            # Handle MSA if needed
            if self.experiment_config.msa_method == "mmseqs2":
                self._handle_msa()
            
            # Save the config with any MSA updates
            configs_dir = Path(self.working_dir) / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            config_path = configs_dir / f"config_{self.job_name}.yaml"
            
            # Save config with all updates
            self.experiment_config.save(config_path)
            logger.debug(f"Saved config with all settings at: {config_path}")
            
            # Create and submit job
            script_path = self.create_job_script(str(config_path), self.job_name)
            
            if self.has_slurm:
                job_id = self._submit_slurm_job(script_path)
            else:
                job_id = self._run_job_locally(script_path)
            
            if not job_id:
                return False, [self.job_name]
            
            return True, []
            
        except Exception as e:
            logger.error(f"Error running experiment: {str(e)}")
            return False, ["Error: " + str(e)]