import os
import gc
import jax
import shutil
from datetime import datetime
from google.colab import files
from typing import Tuple, List, Optional

from .types import MaskingStrategy, ExperimentConfig, ExperimentPaths
from .logging import setup_logger
from .params import create_common_params

logger = setup_logger()


class MaskingExperiment:
    def __init__(self, config: ExperimentConfig):
        from alphamask.core.pipeline import DefaultPipeline, MaskingPipeline, MutateAndMaskingPipeline
        self.DefaultPipeline = DefaultPipeline
        self.MaskingPipeline = MaskingPipeline
        self.MutateAndMaskingPipeline = MutateAndMaskingPipeline
        
        self.config = config
        self.paths = ExperimentPaths()
        self.common_params = create_common_params(config)
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        if self.config.masking_strategy != MaskingStrategy.ITERATIVE_SINGLE:
            if not self.config.positions:
                raise ValueError(
                    "Positions must be provided for non-iterative strategies"
                )

            invalid_positions = [
                p
                for p in self.config.positions
                if p < 1 or p > len(self.config.sequence)
            ]
            if invalid_positions:
                raise ValueError(f"Invalid positions: {invalid_positions}")

    def _clear_memory(self):
        """Clear memory after each run."""
        logger.info("Clearing memory")
        backend = jax.lib.xla_bridge.get_backend()
        for buf in backend.live_buffers():
            buf.delete()
        gc_result = gc.collect()
        logger.info(f"Garbage collection: {gc_result} objects collected")

    def _get_masked_positions(self, position: Optional[int] = None) -> List[int]:
        """Get positions to mask based on strategy."""
        all_positions = set(range(1, len(self.config.sequence) + 1))

        if self.config.masking_strategy == MaskingStrategy.MASK_POSITIONS:
            return self.config.positions
        elif self.config.masking_strategy == MaskingStrategy.UNMASK_POSITIONS:
            return list(all_positions - set(self.config.positions))
        else:  # ITERATIVE_SINGLE
            return [position] if position is not None else []

    def _get_positions_to_iterate(self) -> List[int]:
        """Get the list of positions to iterate over."""
        if self.config.positions:
            return sorted(self.config.positions)
        return list(range(1, len(self.config.sequence) + 1))

    def _run_control(self) -> Optional[str]:
        """Run vanilla AF2 as control."""
        if not self.config.run_control:
            return None

        logger.info("Starting vanilla AlphaFold2 prediction (Control)")
        vanilla_params = self.common_params.copy()
        vanilla_params["jobname"] = f"{self.config.jobname_prefix}_vanilla"
        vanilla_params["cols"] = []
        vanilla_pipeline = self.DefaultPipeline(params=vanilla_params)
        self.paths.vanilla_jobname = vanilla_pipeline.run()

        # Store the control path
        self.paths.vanilla_path = os.path.join(
            self.config.parent_path, self.paths.vanilla_jobname, "out", "pdbs"
        )

        logger.info("Vanilla AlphaFold2 prediction completed")
        logger.info(f"Vanilla results stored in: {self.paths.vanilla_path}")
        self._clear_memory()
        return self.paths.vanilla_jobname

    def _run_masked(self, masked_positions: List[int], jobname: str) -> str:
        """Run a single masked prediction."""
        logger.info(f"Running masked prediction for {len(masked_positions)} positions")
        masked_params = self.common_params.copy()
        masked_params["jobname"] = jobname
        masked_params["cols"] = masked_positions
        masked_pipeline = self.MaskingPipeline(params=masked_params)
        self.paths.mask_jobname = masked_pipeline.run()
        self._clear_memory()

        # Store paths
        self.paths.mask_path = os.path.join(
            self.config.parent_path, self.paths.mask_jobname, "out", "pdbs"
        )

        # Create zip archive
        output_filename = f"{self.paths.mask_jobname}.zip"
        shutil.make_archive(
            self.paths.mask_jobname,
            "zip",
            f"{self.config.parent_path}/{self.paths.mask_jobname}",
        )

        # Download the zip file if in Colab
        try:
            files.download(output_filename)
        except NameError:
            logger.info("Not running in Colab, skipping file download")

        return self.paths.mask_jobname

    def _run_mask_mutate(self, masked_positions: List[int], jobname: str) -> str:
        """Run a single masked and mutate prediction."""
        logger.info(f"Running masked prediction for {len(masked_positions)} positions")
        masked_params = self.common_params.copy()
        masked_params["jobname"] = jobname
        masked_params["cols"] = masked_positions
        masked_pipeline = self.MutateAndMaskingPipeline(params=masked_params)
        self.paths.mask_mutate_jobname = masked_pipeline.run()
        self._clear_memory()

        # Store paths
        self.paths.mask_mutate_path = os.path.join(
            self.config.parent_path, self.paths.mask_mutate_jobname, "out", "pdbs"
        )

        # Create zip archive
        output_filename = f"{self.paths.mask_mutate_jobname}.zip"
        shutil.make_archive(
            self.paths.mask_mutate_jobname,
            "zip",
            f"{self.config.parent_path}/{self.paths.mask_mutate_jobname}",
        )

        # Download the zip file if in Colab
        try:
            files.download(output_filename)
        except NameError:
            logger.info("Not running in Colab, skipping file download")

        return self.paths.mask_mutate_jobname

    def _run_masking_experiments(self):
        """Run masking experiments based on strategy."""
        if self.config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE:
            positions_to_process = self._get_positions_to_iterate()
            logger.info(
                f"Will process {len(positions_to_process)} positions iteratively"
            )

            for position in positions_to_process:
                logger.info(f"Processing position {position}")
                masked_positions = self._get_masked_positions(position)
                self._run_masked(
                    masked_positions, f"{self.config.jobname_prefix}_pos{position}"
                )

        elif (
            self.config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE_MASK_MUTATE
        ):
            positions_to_process = self._get_positions_to_iterate()
            logger.info(
                f"Will mask and mutate {len(positions_to_process)} positions iteratively"
            )

            for position in positions_to_process:
                logger.info(f"Processing position {position}")
                masked_positions = self._get_masked_positions(position)
                mutations_str = (
                    "_".join(self.config.mutations)
                    if self.config.mutations
                    else str(position)
                )
                self._run_mask_mutate(
                    masked_positions, f"{self.config.jobname_prefix}_pos{mutations_str}"
                )
        else:
            # Run once with all specified positions
            masked_positions = self._get_masked_positions()
            strategy_name = (
                "masked"
                if self.config.masking_strategy == MaskingStrategy.MASK_POSITIONS
                else "unmasked"
            )
            self._run_masked(
                masked_positions, f"{self.config.jobname_prefix}_{strategy_name}"
            )

    def _get_result_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """Return appropriate paths based on experiment type."""
        if self.config.run_control and self.config.run_only_control:
            return self.paths.vanilla_jobname, self.paths.vanilla_path
        elif self.config.run_control:
            return self.paths.vanilla_jobname, None
        elif self.config.run_only_control:
            return None, self.paths.vanilla_path
        elif self.config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE:
            return self.paths.mask_jobname, self.paths.mask_path
        elif (
            self.config.masking_strategy == MaskingStrategy.ITERATIVE_SINGLE_MASK_MUTATE
        ):
            return self.paths.mask_mutate_jobname, self.paths.mask_mutate_path
        else:
            return self.paths.mask_jobname, self.paths.mask_path

    def run(self) -> Tuple[Optional[str], Optional[str]]:
        """Run the masking experiment."""
        logger.info(
            f"Starting masking experiment with strategy: {self.config.masking_strategy.value}"
        )

        start_time = datetime.now()

        if self.config.run_control or self.config.run_only_control:
            self._run_control()

        if not self.config.run_only_control:
            self._run_masking_experiments()

        end_time = datetime.now()
        logger.info(f"Experiment completed in {end_time - start_time}")

        return self._get_result_paths()
