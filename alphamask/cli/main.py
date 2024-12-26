import argparse
import sys
from pathlib import Path
import logging
import traceback

# Configure logging first, before any other imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Get logger for this module
logger = logging.getLogger("alphamask.cli.main")
logger.setLevel(logging.DEBUG)

from .commands import setup_cmd, submit_jobs_cmd, help_cmd, predict_cmd, predict_job_cmd

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
        "--partition",
        type=str,
        default="clara",
        help="SLURM partition"
    )
    run_parser.add_argument(
        "--gpu-type",
        type=str,
        default="rtx2080ti",
        help="GPU type to request"
    )
    run_parser.add_argument(
        "--force-local",
        action="store_true",
        help="Force local execution"
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
    
    return parser

def setup_logging(debug: bool, log_file: str = None, quiet: bool = False):
    """Configure logging"""
    if quiet:
        # Disable all logging by setting root logger to CRITICAL+1
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        return
    
    log_level = logging.DEBUG if debug else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add and configure new handlers
    for handler in handlers:
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(handler)

def main():
    """Main entry point for the CLI"""
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        # Setup logging based on command line arguments
        setup_logging(
            debug=args.debug,
            log_file=args.log_file if hasattr(args, 'log_file') else None,
            quiet=args.quiet if hasattr(args, 'quiet') else False
        )
        
        logger.debug(f"Parsed arguments: {args}")
        
        if args.command == "setup":
            logger.debug("Running setup command")
            setup_cmd(args)
        elif args.command == "run":
            logger.debug("Running run command")
            submit_jobs_cmd(args)
        elif args.command == "help":
            help_cmd(args)
        elif args.command == "predict":
            logger.debug("Running predict command")
            predict_cmd(args)
        elif args.command == "predict-job":
            logger.debug("Running predict-job command")
            predict_job_cmd(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        if args.debug:
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main()) 