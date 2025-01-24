"""Job submission command implementation."""

import logging
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...utils.slurm import SlurmJobConfig
from ...experiments.runner import run_experiments
from ...utils.partition import PartitionManager
from ...core.model import CompressionConfig
from .utils import show_summary, console

logger = logging.getLogger(__name__)

def run(args):
    """Submit protein experiment jobs to the SLURM queue."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Submitting jobs...", total=None)
        try:
            # Create compression config
            compression_config = CompressionConfig(
                compress_format=args.compress,
                compression_level=args.compression_level,
                store_uncompressed_pdbs=args.store_uncompressed,
                store_best_pdb=True  # Always store best PDB
            )
            
            # Create SLURM configuration
            slurm_config = SlurmJobConfig(
                container_path=args.container,
                script_path="alphamask predict-job",
                schema_path=args.schema,
                partition=None if args.force_local else args.partitions[0],
                gpu_type=None if args.force_local else args.gpu_types[0],
                setup_commands=["conda activate alphamask"],
                time=args.time,
                memory=args.memory,
                cpus_per_task=args.cpus_per_task,
                bind_work=args.bind_work,
                alphamask_bin_path=args.alphamask_bin_path,
                alphamask_mount_path=args.alphamask_mount_path,
                env_manager=args.env_manager,
                env_module=None if args.env_module.lower() == 'none' else args.env_module,
                env_name=args.env_name,
                env_base_path=args.env_base_path,
                env_setup_script=args.env_setup_script
            )
            
            # Initialize partition manager
            partition_manager = PartitionManager()
            
            # Get base directory
            base_dir = Path(args.path) if hasattr(args, 'path') else Path("/work/nw99ixuq-alphamask/my_experiments")
            
            # Run experiments
            success = run_experiments(
                config_path=args.config,
                slurm_config=slurm_config,
                base_dir=base_dir,
                protein_ids=args.proteins,
                compression_config=compression_config
            )
            
            progress.update(task, completed=True)
            
            show_summary(
                success=success,
                title="Job Submission Complete",
                details={
                    "Config File": args.config,
                    "Command": "alphamask predict-job",
                    "Container": args.container,
                    "Base Directory": str(base_dir),
                    "Partitions": ", ".join(args.partitions),
                    "GPU Types": ", ".join(args.gpu_types),
                    "Proteins": ", ".join(args.proteins) if args.proteins else "All",
                    "Compression": args.compress,
                    "Compression Level": args.compression_level,
                    "Store Uncompressed": "Yes" if args.store_uncompressed else "No"
                }
            )
            
            if not success:
                raise RuntimeError("Some job submissions failed")
                
        except Exception as e:
            logger.error(f"Job submission failed: {str(e)}")
            show_summary(
                success=False,
                title="Job Submission Failed",
                details={
                    "Error": str(e),
                    "Config File": args.config
                }
            )
            raise