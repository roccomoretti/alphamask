"""Prediction command implementations."""

import logging
from rich.console import Console
from rich.progress import Progress
from ...core.pipeline import DefaultPipeline, MaskingPipeline, MutatePipeline, MutateAndMaskingPipeline
from ...utils.config import load_config, validate_config

import yaml

logger = logging.getLogger(__name__)

console = Console()

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
