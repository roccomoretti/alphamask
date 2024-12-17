"""Command handlers for AlphaMask CLI"""

import logging
from pathlib import Path
import os
import textwrap

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..experiments.setup import ExperimentSetup
from ..utils.slurm import SlurmJobConfig
from ..experiments.runner import run_experiments
from ..utils.config import load_config, validate_config
from ..experiments.predict import run_prediction_pipeline

logger = logging.getLogger(__name__)
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
    """Handle setup command"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Setting up experiment directories...", total=None)
        try:
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
            logger.error(f"Setup failed: {str(e)}")
            show_summary(
                success=False,
                title="Setup Failed",
                details={
                    "Error": str(e),
                    "Config File": args.config
                }
            )
            raise

def run_cmd(args):
    """Handle run command"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running experiments...", total=None)
        try:
            # Convert script path to absolute path relative to current directory
            script_path = Path(args.script)
            if not script_path.is_absolute():
                script_path = Path.cwd() / script_path
            
            # Create SLURM configuration
            slurm_config = SlurmJobConfig(
                container_path=args.container,
                script_path=str(script_path),
                schema_path=args.schema,
                partition=None if args.force_local else args.partition,
                gpu_type=None if args.force_local else args.gpu_type
            )
            
            # Get base directory from the path where experiments were set up
            base_dir = Path(args.path) if hasattr(args, 'path') else Path("/work/nw99ixuq-alphamask/my_experiments")
            
            # Run experiments
            success = run_experiments(
                config_path=args.config,
                slurm_config=slurm_config,
                base_dir=base_dir,
                protein_ids=args.proteins
            )
            
            progress.update(task, completed=True)
            
            show_summary(
                success=success,
                title="Experiment Run Complete",
                details={
                    "Config File": args.config,
                    "Container": args.container,
                    "Script": str(script_path),
                    "Base Directory": str(base_dir),
                    "Partition": args.partition if not args.force_local else "Local",
                    "GPU Type": args.gpu_type if not args.force_local else "Local",
                    "Proteins": ", ".join(args.proteins) if args.proteins else "All"
                }
            )
            
            if not success:
                raise RuntimeError("Some experiments failed")
                
        except Exception as e:
            logger.error(f"Run failed: {str(e)}")
            show_summary(
                success=False,
                title="Run Failed",
                details={
                    "Error": str(e),
                    "Config File": args.config
                }
            )
            raise

def predict_cmd(args):
    """Handle predict command"""
    try:
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Running prediction...",
                total=None
            )
            
            # Load and validate configuration
            config = load_config(args.config)
            validate_config(config, args.schema)
            
            # Run prediction pipeline
            run_prediction_pipeline(config, pipeline_type=args.pipeline)
            
            progress.update(task, completed=True)
            
        console.print("[green]Prediction completed successfully")
            
    except Exception as e:
        console.print(f"[red]Error running prediction: {str(e)}")
        if args.debug:
            console.print_exception()
        raise

def help_cmd(args):
    """Handle help command"""
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