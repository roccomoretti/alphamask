"""Setup command implementation."""

import logging
import traceback
import os
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...experiments.setup import ExperimentSetup
from .utils import show_summary, console

logger = logging.getLogger(__name__)

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

