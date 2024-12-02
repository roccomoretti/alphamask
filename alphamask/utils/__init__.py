from typing import Tuple, Optional
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
) -> Tuple[Optional[str], Optional[str]]:
    """Convenience function to run masking experiments."""
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
    )

    experiment = MaskingExperiment(config)
    return experiment.run()
