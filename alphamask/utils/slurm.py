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

logger = setup_logger()

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
    gpu_type: str = "a30"
    gpu_count: int = 1
    partition: str = "paula"
    email: Optional[str] = None
    container_path: str = ""
    script_path: str = ""
    schema_path: str = ""
    setup_base_path: Optional[str] = None

    def __post_init__(self):
        """Set default setup base path if not provided"""
        if self.setup_base_path is None:
            self.setup_base_path = os.path.expanduser("~/alphamask_setup")

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
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.jobs: Dict[int, SlurmJob] = {}
        self.failed_jobs: List[int] = []
        self.has_slurm = self._check_slurm_available()
        
        # Set setup base path in experiment config
        self.experiment_config.setup_base_path = self.slurm_config.setup_base_path
        
        if not self.has_slurm:
            logger.warning("SLURM not detected - jobs will run sequentially")
        
        # Create necessary directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for job management"""
        (self.working_dir / "configs").mkdir(exist_ok=True)
        (self.working_dir / "scripts").mkdir(exist_ok=True)
        (self.working_dir / "logs").mkdir(exist_ok=True)

    def create_job_script(self, config_path: str, job_name: str) -> str:
        """Create a SLURM job script"""
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={self.working_dir}/logs/{job_name}.out
#SBATCH --error={self.working_dir}/logs/{job_name}.err
#SBATCH --time={self.slurm_config.time}
#SBATCH --mem={self.slurm_config.memory}
#SBATCH --cpus-per-task={self.slurm_config.cpus_per_task}
#SBATCH --gres=gpu:{self.slurm_config.gpu_type}:{self.slurm_config.gpu_count}
#SBATCH --partition={self.slurm_config.partition}
"""

        if self.slurm_config.email:
            script_content += f"""#SBATCH --mail-user={self.slurm_config.email}
#SBATCH --mail-type=FAIL,END
"""

        script_content += f"""
singularity exec --nv --cleanenv {self.slurm_config.container_path} \\
    python {self.slurm_config.script_path} \\
    --yaml_file {config_path} \\
    --json_schema {self.slurm_config.schema_path}
"""

        script_path = self.working_dir / "scripts" / f"{job_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)

    def submit_job(self, config_path: str, job_name: str) -> Optional[int]:
        """Submit a job either to SLURM or run it directly"""
        if self.has_slurm:
            return self._submit_slurm_job(config_path, job_name)
        else:
            return self._run_job_locally(config_path, job_name)

    def _submit_slurm_job(self, config_path: str, job_name: str) -> Optional[int]:
        """Submit job to SLURM"""
        script_path = self.create_job_script(config_path, job_name)
        
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                check=True
            )
            job_id = int(result.stdout.strip().split()[-1])
            logger.info(f"Submitted SLURM job {job_name} with ID {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SLURM job {job_name}: {e}")
            return None

    def _run_job_locally(self, config_path: str, job_name: str) -> Optional[int]:
        """Run job locally using singularity"""
        try:
            logger.info(f"Running job {job_name} locally")
            
            # Convert all paths to absolute paths
            workspace_dir = Path.cwd().resolve()
            script_path = workspace_dir / self.slurm_config.script_path
            config_path = workspace_dir / config_path
            schema_path = workspace_dir / self.slurm_config.schema_path
            working_dir = workspace_dir / self.working_dir
            
            # Create config directory and file
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.experiment_config.to_dict(), f)
            
            # Create bind paths for singularity
            bind_paths = [
                f"{workspace_dir}:{workspace_dir}",
                f"{working_dir}:{working_dir}"
            ]
            
            cmd = [
                "singularity", "exec",
                "--nv",
                "--cleanenv",
                *[f"--bind={path}" for path in bind_paths],
                self.slurm_config.container_path,
                "python", str(script_path),
                "--yaml_file", str(config_path),
                "--json_schema", str(schema_path)
            ]
            
            # Create output and error files
            out_file = working_dir / "logs" / f"{job_name}.out"
            err_file = working_dir / "logs" / f"{job_name}.err"
            out_file.parent.mkdir(exist_ok=True)
            
            logger.info(f"Command: {' '.join(cmd)}")
            logger.info(f"Working directory: {working_dir}")
            logger.info(f"Script path: {script_path}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Schema path: {schema_path}")
            logger.info(f"Output file: {out_file}")
            logger.info(f"Error file: {err_file}")
            
            # Run the command and capture output
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # Don't raise exception, we'll handle it
            )
            
            # Save output and error to files
            with open(out_file, 'w') as f:
                f.write(process.stdout)
            with open(err_file, 'w') as f:
                f.write(process.stderr)
            
            if process.returncode != 0:
                logger.error(f"Command failed with return code {process.returncode}")
                logger.error("Command output:")
                logger.error(process.stdout)
                logger.error("Command error:")
                logger.error(process.stderr)
                raise subprocess.CalledProcessError(
                    process.returncode, cmd, 
                    output=process.stdout, 
                    stderr=process.stderr
                )
            
            logger.info(f"Completed job {job_name}")
            return 0  # Return 0 as a pseudo job ID for local runs
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run job {job_name} locally")
            logger.error(f"Command output: {e.output}")
            logger.error(f"Command error: {e.stderr}")
            return None

    def submit_iterative_masking_jobs(self) -> List[int]:
        """Submit jobs for iterative masking experiments"""
        job_ids = []
        sequence_length = len(self.experiment_config.sequence)
        
        for pos in range(1, sequence_length + 1):
            # Create config for this position
            config = self.experiment_config.to_dict()
            config["cols"] = [pos]
            
            config_path = self.working_dir / "configs" / f"config_pos_{pos}.yaml"
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
        """Submit jobs for position-based masking experiments"""
        if not self.experiment_config.positions:
            raise ValueError("Positions must be provided for position-based masking")
        
        job_ids = []
        
        # Create config
        config = self.experiment_config.to_dict()
        
        # Get positions based on strategy
        if self.experiment_config.masking_strategy == MaskingStrategy.MASK_POSITIONS:
            config["cols"] = self.experiment_config.positions
            strategy_name = "masked"
        else:  # UNMASK_POSITIONS
            all_positions = set(range(1, len(self.experiment_config.sequence) + 1))
            config["cols"] = list(all_positions - set(self.experiment_config.positions))
            strategy_name = "unmasked"
        
        # Create config file
        positions_str = "_".join(map(str, config["cols"]))
        config_path = self.working_dir / "configs" / f"config_{strategy_name}_{positions_str}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Submit job
        job_id = self.submit_job(
            str(config_path),
            f"{self.experiment_config.jobname_prefix}_{strategy_name}"
        )
        
        if job_id:
            job_ids.append(job_id)
        
        return job_ids

    def submit_control_job(self) -> Optional[int]:
        """Submit control job (vanilla AF2)"""
        if not self.experiment_config.run_control:
            return None
            
        # Create config for control run
        config = self.experiment_config.to_dict()
        config["cols"] = []
        
        # Create config file
        config_path = self.working_dir / "configs" / "config_control.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Submit job
        return self.submit_job(
            str(config_path),
            f"{self.experiment_config.jobname_prefix}_control"
        )

    def submit_masking_jobs(self) -> List[int]:
        """Submit jobs based on masking strategy"""
        if self.experiment_config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE:
            return self.submit_iterative_masking_jobs()
        elif self.experiment_config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE_MASK_MUTATE:
            return self.submit_iterative_mask_mutate_jobs()
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