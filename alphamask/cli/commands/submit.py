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
            
            # Check if running in Colab or similar environment where container isn't needed
            is_colab = hasattr(args, 'is_colab') and args.is_colab
            
            # Create SLURM configuration
            slurm_config = SlurmJobConfig(
                container_path=None if is_colab else args.container,
                script_path="alphamask predict-job",
                schema_path=args.schema if hasattr(args, 'schema') and args.schema else None,
                partition=None if args.force_local or is_colab else args.partitions[0] if hasattr(args, 'partitions') and args.partitions else None,
                gpu_type=None if args.force_local or is_colab else args.gpu_types[0] if hasattr(args, 'gpu_types') and args.gpu_types else None,
                setup_commands=["conda activate alphamask"],
                time=args.time if hasattr(args, 'time') else None,
                memory=args.memory if hasattr(args, 'memory') else None,
                cpus_per_task=args.cpus_per_task if hasattr(args, 'cpus_per_task') else None,
                bind_work=args.bind_work if hasattr(args, 'bind_work') else False,
                alphamask_bin_path=args.alphamask_bin_path if hasattr(args, 'alphamask_bin_path') else None,
                alphamask_mount_path=args.alphamask_mount_path if hasattr(args, 'alphamask_mount_path') else None,
                env_manager=args.env_manager if hasattr(args, 'env_manager') else 'conda',
                env_name=args.env_name if hasattr(args, 'env_name') else None,
                env_base_path=args.env_base_path if hasattr(args, 'env_base_path') else None
                )
            
            # Initialize partition manager
            partition_manager = PartitionManager()
            
            # Get base directory
            base_dir = Path(args.path) if hasattr(args, 'path') else Path("~/alphamask_experiments/test_experiment")
            
            # Run experiments
            success = run_experiments(
                config_path=args.config,
                slurm_config=slurm_config,
                base_dir=base_dir,
                protein_ids=args.proteins if hasattr(args, 'proteins') else None,
                compression_config=compression_config,
                is_colab=is_colab
            )
            
            progress.update(task, completed=True)
            
            details = {
                "Config File": args.config,
                "Command": "alphamask predict-job",
                "Base Directory": str(base_dir),
                "Compression": args.compress,
                "Compression Level": args.compression_level,
                "Store Uncompressed": "Yes" if args.store_uncompressed else "No"
            }
            
            # Add SLURM-specific details when not in Colab
            if not is_colab:
                details.update({
                    "Container": args.container if hasattr(args, 'container') else "None",
                    "Partitions": ", ".join(args.partitions) if hasattr(args, 'partitions') and args.partitions else "None",
                    "GPU Types": ", ".join(args.gpu_types) if hasattr(args, 'gpu_types') and args.gpu_types else "None",
                    "Proteins": ", ".join(args.proteins) if hasattr(args, 'proteins') and args.proteins else "All",
                })
            
            show_summary(
                success=success,
                title="Job Submission Complete",
                details=details
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