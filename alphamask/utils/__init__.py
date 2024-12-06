from typing import Tuple, Optional, Callable, Any
from .types import MaskingStrategy, ExperimentConfig, ExperimentPaths
from .experiment import MaskingExperiment
from .logging import setup_logger
from .params import create_common_params

__all__ = [
    "MaskingStrategy",
    "ExperimentConfig",
    "MaskingExperiment",
    "run_masking_experiment",
]

def run_masking_experiment(
    sequence: str,
    jobname_prefix: str,
    parent_path: str,
    masking_strategy: str,
    positions_str: str = "",
    num_recycles: int = 12,
    num_seeds: int = 12,
    msa_method: str = "custom_a3m",
    custom_a3m_path: str = "",
    mutations: str = "",
    run_control: bool = True,
    run_only_control: bool = False,
    unified_memory: bool = False,
    callback_fn: Optional[Callable[[Any, Optional[str]], None]] = None,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Convenience function to run masking experiments.
    
    Args:
        sequence: Input protein sequence
        jobname_prefix: Prefix for job names
        parent_path: Path to save results
        masking_strategy: Strategy for masking
        positions_str: Comma-separated positions to mask
        num_recycles: Number of recycles
        num_seeds: Number of seeds
        msa_method: Method for MSA generation
        custom_a3m_path: Path to custom MSA file
        mutations: Comma-separated mutations
        run_control: Whether to run control
        run_only_control: Whether to only run control
        unified_memory: Whether to use unified memory
        callback_fn: Optional callback function for visualization/analysis
        
    Returns:
        Tuple[Optional[str], Optional[Any]]: Tuple of (jobname, pipeline instance)
    """
    # Import here to avoid circular imports
    from alphamask.core.pipeline import DefaultPipeline, MaskingPipeline, MutateAndMaskingPipeline

    # Parse positions if provided
    positions = (
        [int(x.strip()) for x in positions_str.split(",")]
        if positions_str.strip()
        else None
    )

    # Parse mutations if provided
    mutations_list = (
        [x.strip() for x in mutations.split(",")] if mutations.strip() else None
    )

    config = ExperimentConfig(
        sequence=sequence,
        jobname_prefix=jobname_prefix,
        parent_path=parent_path,
        masking_strategy=MaskingStrategy(masking_strategy),
        positions=positions,
        num_recycles=num_recycles,
        num_seeds=num_seeds,
        msa_method=msa_method,
        custom_a3m_path=custom_a3m_path,
        mutations=mutations_list,
        run_control=run_control,
        run_only_control=run_only_control,
        unified_memory=unified_memory,
        callback_fn=callback_fn,
    )

    experiment = MaskingExperiment(config)
    jobname, pipeline = experiment.run()
    
    # Call callback if provided and pipeline exists
    if callback_fn is not None and pipeline is not None:
        mutation_str = None
        if config.mutations:
            mutation_str = config.mutations[0]
        callback_fn(pipeline, mutation_str)
    
    return jobname, pipeline
