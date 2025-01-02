"""Command handlers for AlphaMask CLI"""

import logging
import traceback
from pathlib import Path
import os
import textwrap
import yaml
from typing import Dict
from datetime import datetime, timedelta

# Configure logging first
logger = logging.getLogger("alphamask.cli.commands")

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Group

from ..experiments.setup import ExperimentSetup
from ..utils.slurm import SlurmJobConfig
from ..experiments.runner import run_experiments
from ..utils.config import load_config, validate_config
from ..core.pipeline import DefaultPipeline, MaskingPipeline, MutatePipeline, MutateAndMaskingPipeline

from .utils.status import (
    get_protein_jobs,
    get_completed_jobs_by_protein,
    create_stats_table,
    create_jobs_table,
    get_slurm_jobs,
    get_job_counts,
    parse_config_for_total_jobs,
    get_completed_jobs_count,
    create_status_layout,
)

console = Console()

HELP_TEXTS = {
    "setup": """
        # AlphaMask Setup Command

        Sets up the experiment directory structure and resources.

        ## Usage
        ```bash
        alphamask setup [options]
        ```

        ## Options
        - `--config`: Path to protein configuration file (default: config/proteins.yaml)
        - `--path`: Base path for experiment setup (default: .)
        - `--setup-path`: Base path for setup files (default: ~/alphamask_setup)
        - `--force`: Force setup even if directories exist

        ## Example
        ```bash
        alphamask setup --config my_config.yaml --path /path/to/experiments
        ```
    """,
    "run": """
        # AlphaMask Run Command

        Runs protein experiments based on configuration.

        ## Usage
        ```bash
        alphamask run [options]
        ```

        ## Options
        - `--config`: Path to protein configuration file
        - `--proteins`: Specific proteins to run (optional)
        - `--container`: Path to Singularity container
        - `--script`: Path to prediction script
        - `--schema`: Path to JSON schema
        - `--partition`: SLURM partition (default: clara)
        - `--gpu-type`: GPU type to request (default: rtx2080ti)
        - `--force-local`: Force local execution

        ## Example
        ```bash
        alphamask run --config config.yaml --container container.sif --script predict.py --schema schema.json
        ```
    """,
    "config": """
        # AlphaMask Configuration Guide

        Configuration files use YAML format with the following structure:

        ```yaml
        proteins:
          protein_id:
            sequence: "PROTEIN_SEQUENCE"
            iterative_masking:
              enabled: true
              mutations: [["I89S"]]
            apriori_masking:
              enabled: true
              experiments:
                - name: "experiment_name"
                  positions: [89]
                  mutations: ["I89S"]
                  conditions:
                    - {mask: false, mutate: false}
                    - {mask: true, mutate: false}
                    - {mask: false, mutate: true}
                    - {mask: true, mutate: true}
            frustra_masking:
              enabled: true
              top_positions: 10

        global_settings:
          mask_token: "X"
          output_dir: "./results"
          logging:
            enabled: true
            level: "INFO"
            file: "masking_analysis.log"
        ```

        See migration guide for more details.
        """
}

def show_summary(success: bool, title: str, details: dict):
    """Show a summary of the operation"""
    panel_style = "green" if success else "red"
    status = "[green]SUCCESS[/green]" if success else "[red]FAILED[/red]"
    
    content = [
        title,
        f"Status: {status}",
        "",
        "Details:",
    ]
    
    for key, value in details.items():
        content.append(f"  {key}: {value}")
    
    panel_content = "\n".join(content)
    
    console.print("\n")
    console.print(Panel(
        panel_content,
        style=panel_style,
        title="AlphaMask Operation Summary"
    ))

def setup_cmd(args):
    """Set up experiment directories and resources.

    This command initializes the directory structure and resources needed for running
    AlphaMask experiments. It creates the necessary directories, copies configuration
    files, and prepares the environment.

    Args:
        args: Namespace object from argparse containing:
            - config (str): Path to protein configuration file
            - path (str): Base path for experiment setup
            - setup_path (str): Base path for setup files
            - force (bool): Whether to force setup even if directories exist
            - debug (bool): Enable debug logging
            - quiet (bool): Disable logging output
            - log_file (str, optional): Path to log file

    Raises:
        Exception: If setup fails for any reason (details in error message)
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Setting up experiment directories...", total=None)
            
            setup = ExperimentSetup(
                config_path=args.config,
                setup_path=os.path.expanduser(args.setup_path),
                base_dir=args.path,
                force=args.force
            )
            setup.setup()
            
            progress.update(task, completed=True)
            
            show_summary(
                success=True,
                title="Setup Complete",
                details={
                    "Config File": args.config,
                    "Setup Path": args.setup_path,
                    "Base Directory": args.path,
                    "Force Mode": "Yes" if args.force else "No"
                }
            )
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        show_summary(
            success=False,
            title="Setup Failed",
            details={
                "Error": str(e),
                "Config File": args.config,
                "Traceback": traceback.format_exc()
            }
        )
        raise

def predict_job_cmd(args):
    """Run predictions within a SLURM job context.

    This command is specifically designed to be executed within a SLURM job.
    It handles the conda environment activation and runs the prediction pipeline
    in the job context. This should not be called directly by users, but rather
    is called by the job scheduler.

    Args:
        args: Namespace object from argparse containing:
            - config (str): Path to configuration file
            - schema (str): Path to JSON schema file
            - pipeline (str): Pipeline type to use
            - conda_env (str): Name of conda environment
            - debug (bool): Enable debug logging
            - quiet (bool): Disable logging output
            - log_file (str, optional): Path to log file

    Raises:
        ValueError: If an invalid pipeline type is specified
        RuntimeError: If the pipeline execution fails
        Exception: For other errors during execution
    """
    try:
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Running job prediction...",
                total=None
            )
            
            # Load and validate configuration
            config = load_config(args.config)
            validate_config(config, args.schema)
            
            # Get appropriate pipeline class
            pipeline_map = {
                "default": DefaultPipeline,  # Basic prediction without masking
                "vanilla": DefaultPipeline,  # Alias for default
                "masking": MaskingPipeline,  # Masking operations
                "mutate": MutatePipeline,  # Mutation operations
                "mutate_and_mask": MutateAndMaskingPipeline  # Combined mutation and masking
            }
            
            if args.pipeline not in pipeline_map:
                raise ValueError(f"Unknown pipeline type: {args.pipeline}. Valid choices are: {list(pipeline_map.keys())}")
            
            pipeline_class = pipeline_map[args.pipeline]
            
            # Initialize and run pipeline directly
            logger.info(f"Running prediction pipeline: {args.pipeline}")
            logger.info(f"Config: {yaml.dump(config, default_flow_style=False, indent=4)}")
            
            pipeline = pipeline_class(
                params=config,
                use_parent_dir=True
            )
            result = pipeline.run()
            
            if not result.success:
                raise RuntimeError(f"Pipeline failed: {result.error}")
            
            progress.update(task, completed=True)
            console.print("[green]Job prediction completed successfully")
            
    except Exception as e:
        console.print(f"[red]Error running job prediction: {str(e)}")
        if args.debug:
            console.print_exception()
        raise

def submit_jobs_cmd(args):
    """Submit protein experiment jobs to the SLURM queue.

    This command prepares and submits protein experiment jobs to the SLURM queue based on
    the configuration file. It handles job configuration, submission to SLURM, and provides
    a summary of the submission status. Can submit jobs for all proteins or specific ones.

    Args:
        args: Namespace object from argparse containing:
            - config (str): Path to protein configuration file
            - path (str, optional): Base path for experiments
            - proteins (List[str], optional): Specific proteins to run
            - container (str): Path to Singularity container
            - schema (str): Path to JSON schema
            - partition (str): SLURM partition
            - gpu_type (str): GPU type to request
            - force_local (bool): Force local execution
            - compress (str): Compression format ("h5", "npz", or "both")
            - compression_level (int): Compression level (1-9)
            - store_uncompressed (bool): Whether to store uncompressed PDBs
            - debug (bool): Enable debug logging
            - quiet (bool): Disable logging output
            - log_file (str, optional): Path to log file

    Raises:
        RuntimeError: If any job submissions fail
        Exception: For other errors during submission process
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Submitting jobs...", total=None)
        try:
            # Create compression config
            from ..core.model import CompressionConfig
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
                partition=None if args.force_local else args.partition,
                gpu_type=None if args.force_local else args.gpu_type,
                setup_commands=["conda activate alphamask"]
            )
            
            # Get base directory
            base_dir = Path(args.path) if hasattr(args, 'path') else Path("/work/nw99ixuq-alphamask/my_experiments")
            
            # Add compression config to experiment parameters
            config = load_config(args.config)
            if 'global_settings' not in config:
                config['global_settings'] = {}
            config['global_settings']['compression'] = {
                'format': args.compress,
                'level': args.compression_level,
                'store_uncompressed': args.store_uncompressed
            }
            
            # Run experiments with compression config
            success = run_experiments(
                config_path=args.config,
                slurm_config=slurm_config,
                base_dir=base_dir,
                protein_ids=args.proteins,
                compression_config=compression_config  # Pass compression config
            )
            
            progress.update(task, completed=True)
            
            show_summary(
                success=success,
                title="Job Submission Complete",
                details={
                    "Config File": args.config,
                    "Container": args.container,
                    "Command": "alphamask predict-job",
                    "Base Directory": str(base_dir),
                    "Partition": args.partition if not args.force_local else "Local",
                    "GPU Type": args.gpu_type if not args.force_local else "Local",
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

def predict_cmd(args):
    """Run direct predictions without using experiments or SLURM.

    This command executes a single prediction pipeline directly, without the overhead
    of experiment management or SLURM job scheduling. It's suitable for development,
    testing, and simple predictions that don't require distributed execution.

    The command supports different pipeline types:
    - default/vanilla: Basic prediction without masking
    - masking: Prediction with masking operations
    - mutate: Prediction with mutations
    - mutate_and_mask: Prediction with both mutations and masking

    Args:
        args: Namespace object from argparse containing:
            - config (str): Path to configuration file
            - schema (str): Path to JSON schema file
            - pipeline (str): Pipeline type to use
            - debug (bool): Enable debug logging
            - quiet (bool): Disable logging output
            - log_file (str, optional): Path to log file

    Raises:
        ValueError: If an invalid pipeline type is specified
        RuntimeError: If the pipeline execution fails
        Exception: For other errors during execution
    """
    try:
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Running prediction...",
                total=None
            )
            
            # Load and validate configuration
            config = load_config(args.config)
            validate_config(config, args.schema)
            
            # Get appropriate pipeline class
            pipeline_map = {
                "default": DefaultPipeline,  # Basic prediction without masking
                "masking": MaskingPipeline,  # Masking operations
                "mutate": MutatePipeline,  # Mutation operations
                "mutate_and_mask": MutateAndMaskingPipeline  # Combined mutation and masking
            }
            
            if args.pipeline not in pipeline_map:
                raise ValueError(f"Unknown pipeline type: {args.pipeline}. Valid choices are: {list(pipeline_map.keys())}")
            
            pipeline_class = pipeline_map[args.pipeline]
            
            # Initialize and run pipeline directly
            logger.info(f"Running prediction pipeline: {args.pipeline}")
            logger.info(f"Config: {yaml.dump(config, default_flow_style=False, indent=4)}")
            
            pipeline = pipeline_class(params=config)
            result = pipeline.run()
            
            if not result.success:
                raise RuntimeError(f"Pipeline failed: {result.error}")
            
            progress.update(task, completed=True)
            console.print("[green]Prediction completed successfully")
            
    except Exception as e:
        console.print(f"[red]Error running prediction: {str(e)}")
        if args.debug:
            console.print_exception()
        raise

def help_cmd(args):
    """Display help information for AlphaMask commands.

    This command shows detailed help information about AlphaMask commands
    and configuration. It can display general help or specific help for
    a particular topic.

    Args:
        args: Namespace object from argparse containing:
            - topic (str, optional): Specific topic to get help on
                Choices: ["setup", "run", "config"]
            - debug (bool): Enable debug logging
            - quiet (bool): Disable logging output
            - log_file (str, optional): Path to log file
    """
    if args.topic:
        if args.topic in HELP_TEXTS:
            console.print(Markdown(textwrap.dedent(HELP_TEXTS[args.topic])))
        else:
            console.print(f"No help available for topic: {args.topic}")
    else:
        console.print(Panel.fit(
            "AlphaMask: Protein Masking Analysis Tool\n\n"
            "Available commands:\n"
            "  setup  - Set up experiment directories\n"
            "  run    - Run protein experiments\n"
            "  help   - Show this help message\n\n"
            "For detailed help on a command:\n"
            "  alphamask help [command]\n\n"
            "For configuration help:\n"
            "  alphamask help config",
            title="AlphaMask Help"
        ))

def extract_pdbs_cmd(args):
    """Extract PDBs from compressed storage."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting PDBs...", total=None)
            
            # Load and parse YAML config
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            # Get proteins to process
            proteins = args.proteins if args.proteins else list(config.get("proteins", {}).keys())
            
            if not proteins:
                raise ValueError("No proteins found in config file")
            
            from alphamask.analysis.compressed import extract_pdbs_from_experiments
            
            success = extract_pdbs_from_experiments(
                config=config,
                protein_ids=proteins,
                models=args.models,
                seeds=args.seeds,
                recycles=args.recycles,
                best_only=args.best_only
            )
            
            progress.update(task, completed=True)
            
            show_summary(
                success=success,
                title="PDB Extraction Complete",
                details={
                    "Config File": args.config,
                    "Proteins": ", ".join(proteins),
                    "Models": ", ".join(args.models) if args.models else "All",
                    "Seeds": ", ".join(args.seeds) if args.seeds else "All",
                    "Recycles": ", ".join(args.recycles) if args.recycles else "All",
                    "Best Only": "Yes" if args.best_only else "No"
                }
            )
            
    except Exception as e:
        logger.error(f"PDB extraction failed: {str(e)}")
        show_summary(
            success=False,
            title="PDB Extraction Failed",
            details={
                "Error": str(e),
                "Config File": args.config
            }
        )
        raise

def status_cmd(args):
    """Show status of running AlphaMask experiments."""
    try:
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Table
        from rich.console import Group
        from datetime import datetime, timedelta
        import os
        from pathlib import Path

        # Initial setup
        layout = create_status_layout()
        total_proteins = parse_config_for_total_jobs(args.config)

        # Create progress bar
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True
        )
        
        total_task = progress.add_task(
            "Total Progress", 
            total=total_proteins,
            completed=0
        )

        if args.refresh > 0:
            with Live(layout, refresh_per_second=1/args.refresh, screen=True):
                while True:
                    # Get job counts
                    jobs = get_slurm_jobs()
                    running_jobs, pending_jobs, slurm_completed, failed_jobs = get_job_counts(jobs)
                    completed_jobs = get_completed_jobs_count(args.path)
                    
                    # Get per-protein progress
                    protein_jobs = get_protein_jobs(args.config)
                    completed_by_protein = get_completed_jobs_by_protein(args.path, args.config)
                    
                    protein_progress = {
                        protein_id: (completed_by_protein.get(protein_id, 0), total)
                        for protein_id, total in protein_jobs.items()
                    }
                    
                    # Update components
                    progress.update(total_task, completed=completed_jobs, total=total_proteins)
                    
                    stats_table = create_stats_table(
                        total_proteins, 
                        completed_jobs, 
                        running_jobs, 
                        pending_jobs, 
                        failed_jobs,
                        protein_progress=protein_progress,
                        jobs=jobs,
                        args=args
                    )
                    
                    # Update header
                    layout["header"].update(Panel(
                        f"[bold blue]AlphaMask Job Status[/bold blue]\n"
                        f"Path: {args.path}",
                        style="blue"
                    ))
                    
                    # Update left panel with new stats table
                    left_panel = Panel(
                        Group(
                            progress,
                            "\n",
                            stats_table
                        ),
                        title="Progress Overview"
                    )
                    layout["left"].update(left_panel)
                    
                    # Update right panel with new jobs table
                    active_jobs_table = create_jobs_table(jobs)
                    layout["right"].update(Panel(active_jobs_table))
                    
                    # Update footer
                    layout["footer"].update(Panel(""))
        else:
            # Single update mode
            jobs = get_slurm_jobs()
            running_jobs, pending_jobs, slurm_completed, failed_jobs = get_job_counts(jobs)
            completed_jobs = get_completed_jobs_count(args.path)
            
            # Update all components once
            layout["header"].update(Panel(
                f"[bold blue]AlphaMask Job Status[/bold blue]\n"
                f"Path: {args.path}",
                style="blue"
            ))
            
            progress.update(total_task, completed=completed_jobs, total=total_proteins)
            
            stats_table = create_stats_table(
                total_proteins, 
                completed_jobs, 
                running_jobs, 
                pending_jobs, 
                failed_jobs,
                protein_progress=protein_progress,
                jobs=jobs,
                args=args
            )
            left_panel = Panel(
                Group(
                    progress,
                    "\n",
                    stats_table
                ),
                title="Progress Overview"
            )
            layout["left"].update(left_panel)
            
            active_jobs_table = create_jobs_table(jobs)
            layout["right"].update(Panel(active_jobs_table))
            
            layout["footer"].update(Panel(""))
            
            console.print(layout)

    except KeyboardInterrupt:
        return
    except Exception as e:
        logger.error(f"Status command failed: {str(e)}")
        show_summary(
            success=False,
            title="Status Check Failed",
            details={
                "Error": str(e),
                "Config": args.config,
                "Path": args.path
            }
        )
        raise