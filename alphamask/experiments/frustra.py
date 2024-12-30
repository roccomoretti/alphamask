from pathlib import Path
import logging
from typing import Optional, List

from ..utils.slurm import SlurmJobManager, SlurmJobConfig
from ..utils.types import ExperimentConfig
from ..utils.params import load_defaults

from .base import BaseExperiment, Control, ExperimentError
from .types import ProteinConfig, Condition

logger = logging.getLogger(__name__)

class FrustraExperiment(BaseExperiment):
    """Implementation of Frustra-guided masking experiments"""
    
    def __init__(
        self,
        name: str,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None
    ):
        super().__init__(name, protein_config, slurm_config, working_dir)
        
        if not protein_config.frustra_masking.enabled:
            raise ExperimentError("Frustra masking is not enabled in configuration")
            
        self.top_positions = protein_config.frustra_masking.top_positions
        
        # Create analysis directories
        self.analysis_dir = self.working_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.working_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """Validate Frustra masking configuration"""
        if self.top_positions < 1:
            raise ExperimentError("Number of top positions must be positive")
            
        if self.top_positions > len(self.protein_config.sequence):
            raise ExperimentError(
                f"Number of top positions ({self.top_positions}) exceeds "
                f"sequence length ({len(self.protein_config.sequence)})"
            )
    
    def setup(self) -> None:
        """Set up Frustra masking experiment"""
        # Create controls directory
        controls_dir = self.working_dir / "controls"
        controls_dir.mkdir(parents=True, exist_ok=True)
        
        # Add vanilla control
        self.controls.append(Control(
            name="vanilla",
            protein_config=self.protein_config,
            slurm_config=self.slurm_config,
            conditions=[Condition(mask=False, mutate=False)],
            working_dir=controls_dir
        ))
        
        # Add basic masking control
        self.controls.append(Control(
            name="masking_only",
            protein_config=self.protein_config,
            slurm_config=self.slurm_config,
            conditions=[Condition(mask=True, mutate=False)],
            working_dir=controls_dir
        ))
    
    def _run_frustra_analysis(self) -> List[int]:
        """
        Run Frustra analysis to identify positions of interest.
        Returns list of positions sorted by frustration score.
        """
        try:
            # TODO: Implement Frustra analysis integration
            # This should:
            # 1. Run Frustra analysis on the protein
            # 2. Calculate frustration scores
            # 3. Sort positions by score
            # 4. Return top N positions
            
            # For now, return dummy positions for testing
            return list(range(1, min(self.top_positions + 1, len(self.protein_config.sequence) + 1)))
            
        except Exception as e:
            logger.error(f"Failed to run Frustra analysis: {str(e)}")
            raise ExperimentError(f"Frustra analysis failed: {str(e)}")
    
    def _create_config(self) -> ExperimentConfig:
        """Create configuration for Frustra masking experiment"""
        # Run Frustra analysis to get positions if needed
        positions = self._run_frustra_analysis()
        
        return ExperimentConfig(
            sequence=self.protein_config.sequence,
            jobname_prefix=f"{self.name}",
            parent_path=str(self.working_dir),
            positions=positions,  # Use positions from Frustra analysis
            num_recycles=self.defaults.get('num_recycles', 2),
            num_seeds=self.defaults.get('num_seeds', 2),
            setup_path=str(self.slurm_config.setup_path),
            pipeline_type="frustra"
        )
    
    def run(self) -> bool:
        """Run Frustra masking experiment"""
        self.validate()
        self.setup()
        
        # Run controls first
        for control in self.controls:
            if not control.run():
                return False
        
        try:
            # Run Frustra analysis
            logger.info("Running Frustra analysis...")
            positions = self._run_frustra_analysis()
            
            # Save positions for reference
            positions_file = self.analysis_dir / "top_positions.txt"
            with open(positions_file, "w") as f:
                f.write("\n".join(map(str, positions)))
            logger.info(f"Saved top positions to {positions_file}")
            
            # Create and run masking experiment
            config = self._create_config()
            
            job_manager = SlurmJobManager(
                experiment_config=config,
                slurm_config=self.slurm_config,
                working_dir=str(self.working_dir)
            )
            
            success, failed_jobs = job_manager.run_experiment()
            if not success:
                logger.error("Failed to run Frustra masking experiment")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in Frustra experiment: {str(e)}")
            return False 