import argparse
import sys
from pathlib import Path
import logging
import traceback
from typing import Optional

# Configure logging first, before any other imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Get logger for this module
logger = logging.getLogger("alphamask.cli.main")
logger.setLevel(logging.DEBUG)

from .commands import (
    setup_cmd,
    run,
    help_cmd,
    predict_job_cmd,
    analyze_cmd,
    resubmit_incomplete
)
from .utils.status import status_cmd
from .utils.extract_pdbs import extract_pdbs_cmd

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        description="AlphaMask: Protein Masking Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parent_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable all logging output"
    )
    parent_parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Help command
    help_parser = subparsers.add_parser(
        "help",
        help="Show detailed help for AlphaMask commands",
        parents=[parent_parser]
    )
    help_parser.add_argument(
        "topic",
        nargs="?",
        choices=["setup", "run", "config"],
        help="Specific topic to get help on"
    )
    
    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Set up experiment directories",
        parents=[parent_parser]
    )
    setup_parser.add_argument(
        "--config",
        type=str,
        default="config/proteins.yaml",
        help="Path to protein configuration file"
    )
    setup_parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Base path for experiment setup"
    )
    setup_parser.add_argument(
        "--setup-path",
        type=str,
        default="~/alphamask_setup",
        help="Base path for setup files"
    )
    setup_parser.add_argument(
        "--force",
        action="store_true",
        help="Force setup even if directories exist"
    )
    
    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run protein experiments",
        parents=[parent_parser]
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default="config/proteins.yaml",
        help="Path to protein configuration file"
    )
    run_parser.add_argument(
        "--path",
        type=str,
        help="Base path for experiments (should match setup path)"
    )
    run_parser.add_argument(
        "--proteins",
        type=str,
        nargs="*",
        help="Specific proteins to run (default: all)"
    )
    run_parser.add_argument(
        "--container",
        type=str,
        required=True,
        help="Path to Singularity container"
    )
    run_parser.add_argument(
        "--schema",
        type=str,
        required=True,
        help="Path to JSON schema"
    )
    run_parser.add_argument(
        "--partitions",
        type=str,
        nargs="+",
        default=["clara"],
        help="SLURM partitions to use (e.g., clara paula)"
    )
    run_parser.add_argument(
        "--gpu-types",
        type=str,
        nargs="+",
        default=["rtx2080ti"],
        help="GPU types to use (e.g., rtx2080ti v100 a30)"
    )
    run_parser.add_argument(
        "--force-local",
        action="store_true",
        help="Force local execution"
    )
    run_parser.add_argument(
        "--time",
        type=str,
        default="02:00:00",
        help="Wall time limit for SLURM jobs (HH:MM:SS)"
    )
    run_parser.add_argument(
        "--memory",
        type=str,
        default="10000",
        help="Memory limit for SLURM jobs (MB)"
    )
    run_parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="Number of CPUs per task"
    )
    run_parser.add_argument(
        "--bind-work",
        action="store_true",
        help="Bind /work:/work in Singularity container"
    )
    run_parser.add_argument(
        "--alphamask-bin-path",
        type=str,
        default="~/.conda/envs/alphamask/bin/alphamask",
        help="Path to alphamask binary"
    )
    run_parser.add_argument(
        "--alphamask-mount-path",
        type=str,
        default="$HOME/github/alphamask:/opt/alphamask",
        help="Mount path for alphamask in container"
    )
    run_parser.add_argument(
        "--compress",
        choices=["h5", "npz", "both"],
        default="both",
        help="Compression format for predictions"
    )
    run_parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="Compression level (1-9)"
    )
    run_parser.add_argument(
        "--store-uncompressed",
        action="store_true",
        help="Store uncompressed PDBs alongside compressed data"
    )
    
    # Add environment management options
    env_group = run_parser.add_argument_group('Environment Management')
    env_group.add_argument(
        "--env-manager",
        choices=["conda", "mamba", "micromamba"],
        default="conda",
        help="Package manager to use (conda, mamba, micromamba)"
    )
    env_group.add_argument(
        "--env-name",
        type=str,
        default="alphamask",
        help="Environment name"
    )
    env_group.add_argument(
        "--env-base-path",
        type=str,
        help="Base path for environments (default: ~/.conda)"
    )
    env_group.add_argument(
        "--env-setup-script",
        type=str,
        help="Path to environment setup script (e.g., ~/.bashrc)"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Run direct predictions without experiments or SLURM",
        parents=[parent_parser]
    )
    predict_parser.add_argument(
        "--config",
        required=True,
        help="Path to config file"
    )
    predict_parser.add_argument(
        "--schema",
        required=True,
        help="Path to schema file"
    )
    predict_parser.add_argument(
        "--pipeline",
        choices=["default", "vanilla", "masking", "mutate", "mutate_and_mask"],
        default="default",
        help="Pipeline type ('vanilla' is an alias for 'default')"
    )
    
    # Predict Job command (for SLURM execution)
    predict_job_parser = subparsers.add_parser(
        "predict-job",
        help="Run predictions within SLURM job context",
        parents=[parent_parser]
    )
    predict_job_parser.add_argument(
        "--config",
        required=True,
        help="Path to config file"
    )
    predict_job_parser.add_argument(
        "--schema",
        required=True,
        help="Path to schema file"
    )
    predict_job_parser.add_argument(
        "--pipeline",
        choices=["default", "vanilla", "masking", "mutate", "mutate_and_mask"],
        default="default",
        help="Pipeline type ('vanilla' is an alias for 'default')"
    )
    predict_job_parser.add_argument(
        "--conda-env",
        type=str,
        default="alphamask",
        help="Conda environment name containing AlphaMask"
    )
    
    # Extract PDBs command
    extract_pdbs_parser = subparsers.add_parser(
        "extract-pdbs",
        help="Extract PDBs from compressed storage",
        parents=[parent_parser]
    )
    extract_pdbs_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to protein configuration file"
    )
    extract_pdbs_parser.add_argument(
        "--proteins",
        type=str,
        nargs="*",
        help="Specific proteins to extract (default: all)"
    )
    extract_pdbs_parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="Specific models to extract (e.g., model_1, model_2)"
    )
    extract_pdbs_parser.add_argument(
        "--seeds",
        type=str,
        nargs="*",
        help="Specific seeds to extract"
    )
    extract_pdbs_parser.add_argument(
        "--recycles",
        type=str,
        nargs="*",
        help="Specific recycle iterations to extract"
    )
    extract_pdbs_parser.add_argument(
        "--best-only",
        action="store_true",
        help="Extract only the best prediction"
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of running experiments",
        parents=[parent_parser]
    )
    status_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to protein configuration file"
    )
    status_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Base path for experiments"
    )
    status_parser.add_argument(
        "--refresh",
        type=float,
        default=5.0,
        help="Refresh interval in seconds (0 for single update)"
    )
    
    # Analysis command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze RMSD distributions for completed experiments",
        parents=[parent_parser]
    )
    analyze_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to protein configuration file"
    )
    analyze_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Base path for experiments"
    )
    analyze_parser.add_argument(
        "--proteins",
        type=str,
        nargs="*",
        help="Specific proteins to analyze (default: all)"
    )
    analyze_parser.add_argument(
        "--regions",
        type=str,
        nargs="*",
        help="Specific regions to analyze (default: all regions defined in config)"
    )
    analyze_parser.add_argument(
        "--experiment-types",
        type=str,
        nargs="+",
        choices=["apriori", "iterative"],
        default=["apriori", "iterative"],
        help="Types of experiments to analyze (default: all)"
    )
    analyze_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes"
    )
    analyze_parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only analyze new results"
    )
    analyze_parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        help="Plot output format"
    )
    analyze_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    analyze_parser.add_argument(
        "--force",
        action="store_true", 
        help="Force reanalysis of existing results"
    )
    analyze_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing RMSD files and plots"
    )
    
    # Visualization options
    vis_group = analyze_parser.add_argument_group('Visualization Options')
    vis_group.add_argument(
        "--recycle",
        action="store_true",
        help="Generate recycle analysis visualizations"
    )
    vis_group.add_argument(
        "--plot-types",
        nargs="+",
        choices=["rmsd", "summary", "comparison", "all"],
        default=["all"],
        help="Types of plots to generate (rmsd=individual landscapes, summary=collages, comparison=model comparisons)"
    )
    vis_group.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="Specific models to include in visualizations (default: all)"
    )
    vis_group.add_argument(
        "--recycles",
        type=str,
        nargs="*",
        help="Specific recycle iterations to include (default: all)"
    )
    vis_group.add_argument(
        "--positions",
        type=int,
        nargs="*",
        help="Specific positions to analyze (default: all)"
    )
    vis_group.add_argument(
        "--per-position-plots",
        action="store_true",
        default=False,
        help="Generate per-position plots for iterative experiments (default: False)"
    )
    vis_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output plots"
    )
    vis_group.add_argument(
        "--style",
        choices=["default", "publication", "presentation"],
        default="default",
        help="Plot style preset"
    )
    
    # Output organization
    output_group = analyze_parser.add_argument_group('Output Organization')
    output_group.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for plots (default: analysis/plots)"
    )
    output_group.add_argument(
        "--flat",
        action="store_true",
        help="Use flat directory structure instead of hierarchical"
    )
    output_group.add_argument(
        "--prefix",
        type=str,
        help="Prefix for output files"
    )
    
    # Add our new "resubmit-incomplete" command
    resubmit_incomplete_parser = subparsers.add_parser(
        "resubmit-incomplete",
        help="Resubmit incomplete jobs for iterative or apriori experiments",
        parents=[parent_parser]
    )
    resubmit_incomplete_parser.add_argument(
        "--path",
        type=str,
        default="/work/nw99ixuq-alphamask/my_experiments",
        help="Base path for experiments"
    )
    resubmit_incomplete_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be resubmitted without actually submitting"
    )
    resubmit_incomplete_parser.set_defaults(func=resubmit_incomplete)
    
    return parser

def setup_logging(debug: bool = False, log_file: Optional[str] = None, quiet: bool = False) -> None:
    """Configure logging based on command line arguments."""
    # Set log level
    log_level = logging.DEBUG if debug else logging.INFO
    if quiet:
        log_level = logging.WARNING
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure console handler with appropriate format
    console_handler = logging.StreamHandler()
    if debug:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
    
    # Set specific loggers to appropriate levels
    logging.getLogger('jax').setLevel(logging.WARNING)
    logging.getLogger('jax._src').setLevel(logging.WARNING)
    logging.getLogger('absl').setLevel(logging.WARNING)
    
    # Ensure our modules are set to debug level when debug is True
    if debug:
        logging.getLogger('alphamask').setLevel(logging.DEBUG)
        logging.getLogger('alphamask.utils').setLevel(logging.DEBUG)
        logging.getLogger('alphamask.analysis').setLevel(logging.DEBUG)
        # set alphamask.utils.compression to warning
        logging.getLogger('alphamask.utils.compression').setLevel(logging.WARNING)
        
def main():
    """Main entry point for the CLI"""
    parser = create_parser()
    
    try:
        args = parser.parse_args()
        
        # If no command is provided, show help and exit
        if not args.command:
            parser.print_help()
            return 1
        
        # Setup logging based on command line arguments
        setup_logging(
            debug=getattr(args, 'debug', False),
            log_file=getattr(args, 'log_file', None),
            quiet=getattr(args, 'quiet', False)
        )
        
        logger.debug(f"Parsed arguments: {args}")
        logger.debug(f"Running {args.command} command")
        
        # Handle visualization options for analyze command
        if args.command == "analyze":
            # Set up visualization config
            if hasattr(args, 'plot_types'):
                if "all" in args.plot_types:
                    args.plot_types = ["rmsd", "summary", "comparison"]
                
                # Validate plot type combinations
                if args.no_plots and args.plot_types:
                    logger.warning("--no-plots specified with --plot-types, plots will be skipped")
                
                # Set up output directory
                if args.output_dir:
                    args.output_dir = Path(args.output_dir)
                else:
                    args.output_dir = Path(args.path) / "analysis" / "plots"
                
                if not args.flat:
                    # Create hierarchical structure
                    for plot_type in args.plot_types:
                        (args.output_dir / plot_type).mkdir(parents=True, exist_ok=True)
                else:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                
                # Handle style presets
                if args.style == "publication":
                    import matplotlib.pyplot as plt
                    plt.style.use(['science', 'ieee'])
                    plt.rcParams.update({
                        'figure.dpi': args.dpi,
                        'savefig.dpi': args.dpi,
                        'font.size': 8,
                        'axes.labelsize': 8,
                        'axes.titlesize': 10,
                        'xtick.labelsize': 6,
                        'ytick.labelsize': 6,
                        'legend.fontsize': 6,
                        'figure.titlesize': 12
                    })
                elif args.style == "presentation":
                    import matplotlib.pyplot as plt
                    plt.style.use('seaborn-talk')
                    plt.rcParams.update({
                        'figure.dpi': args.dpi,
                        'savefig.dpi': args.dpi,
                        'font.size': 12,
                        'axes.labelsize': 12,
                        'axes.titlesize': 14,
                        'xtick.labelsize': 10,
                        'ytick.labelsize': 10,
                        'legend.fontsize': 10,
                        'figure.titlesize': 16
                    })
            
            analyze_cmd(args)
            
        elif args.command == "setup":
            setup_cmd(args)
        elif args.command == "run":
            run(args)
        elif args.command == "help":
            help_cmd(args)
        elif args.command == "predict":
            logger.debug("Running predict command, this needs to be refactored")
            raise NotImplementedError("Predict command needs refactoring")
        elif args.command == "predict-job":
            predict_job_cmd(args)
        elif args.command == "extract-pdbs":
            extract_pdbs_cmd(args)
        elif args.command == "status":
            status_cmd(args)
        elif args.command == "resubmit-incomplete":
            # Warn the user that this is work in progress
            logger.warning("This command is work in progress and may not work as expected for your use case.")
            resubmit_incomplete(args)
        else:
            parser.print_help()
            return 1
            
        return 0
        
    except Exception as e:
        # Set up basic error logging without args
        setup_logging(debug=False, quiet=False)
        logger.error(f"Command failed: {str(e)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 