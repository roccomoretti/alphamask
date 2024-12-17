import argparse
import sys
import logging
from pathlib import Path

from .commands import setup_cmd, run_cmd, help_cmd

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
        "--script",
        type=str,
        required=True,
        help="Path to prediction script"
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
        help="Run predictions",
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
        choices=["default", "masking", "mutate", "mutate_and_mask"],
        default="default",
        help="Pipeline type"
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
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug, args.log_file, args.quiet)
    
    try:
        if args.command == "setup":
            setup_cmd(args)
        elif args.command == "run":
            run_cmd(args)
        elif args.command == "help":
            help_cmd(args)
        elif args.command == "predict":
            from alphamask.experiments.config import process_configuration
            from alphamask.experiments.predict import run_prediction_pipeline
            
            # Process configuration
            config = process_configuration(args.config, args.schema)
            
            # Run prediction pipeline
            run_prediction_pipeline(config, pipeline_type=args.pipeline)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        if not args.quiet:
            logging.error(f"Command failed: {str(e)}")
            if args.debug:
                logging.exception("Detailed error trace:")
        sys.exit(1)

if __name__ == "__main__":
    main() 