from pathlib import Path
import logging
from typing import Optional, List, Dict, Any

from ..utils.slurm import SlurmJobManager, SlurmJobConfig
from ..utils.types import ExperimentConfig, MaskingStrategy
from ..utils.params import load_defaults

from .base import BaseExperiment, Control, ExperimentError
from .types import ProteinConfig, Condition

logger = logging.getLogger(__name__)

class IterativeExperiment(BaseExperiment):
    """
    Implementation of iterative masking experiments.
    
    Workflow:
    1. Generate MSA for WT sequence
    2. Run single position masking on WT
    3. For each mutation set:
        - Apply mutations to target sequence
        - Use WT MSA with mutated target
        - Run single position masking
    """
    
    def __init__(
        self,
        protein_config: ProteinConfig,
        slurm_config: Optional[SlurmJobConfig] = None,
        working_dir: Optional[Path] = None
    ):
        super().__init__("iterative_masking", protein_config, slurm_config, working_dir)
        
        if not protein_config.iterative_masking.enabled:
            raise ExperimentError("Iterative masking is not enabled in configuration")
            
        self.mutations = protein_config.iterative_masking.mutations
        self.mask_token = protein_config.iterative_masking.mask_token
        
        # Store MSA and results
        self.wt_msa: Optional[str] = None
        self.wt_results: Dict[int, bool] = {}  # position -> success
        self.mutation_results: Dict[str, Dict[int, bool]] = {}  # mutation_set -> position -> success
        
        # Create directories
        self.wt_dir = self.working_dir / "wt"
        self.mutations_dir = self.working_dir / "mutations"
        self.wt_dir.mkdir(parents=True, exist_ok=True)
        self.mutations_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> None:
        """Validate iterative masking configuration"""
        if not self.mutations:
            raise ExperimentError("No mutations specified for iterative masking")
            
        # Validate MSA method
        if not hasattr(self.protein_config, 'msa_method'):
            raise ExperimentError("MSA method not specified in configuration")
            
        # Validate mutations against sequence
        for mutation_set in self.mutations:
            for mutation in mutation_set:
                if not self._validate_mutation(mutation):
                    raise ExperimentError(f"Invalid mutation {mutation}")
    
    def _validate_mutation(self, mutation: str) -> bool:
        """Validate a single mutation against the protein sequence"""
        if len(mutation) < 3:
            return False
            
        orig_aa = mutation[0]
        new_aa = mutation[-1]
        pos = int(mutation[1:-1])
        
        if pos < 1 or pos > len(self.protein_config.sequence):
            return False
            
        if self.protein_config.sequence[pos-1] != orig_aa:
            return False
            
        return True
    
    def _generate_msa(self, sequence: str) -> str:
        """Generate MSA for a sequence."""
        # Create MSA directory if it doesn't exist
        msa_dir = self.working_dir / "in" / "msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        
        # Write sequence to MSA file in A3M format
        msa_path = msa_dir / "msa.a3m"
        with open(msa_path, "w") as f:
            f.write(f">target\n{sequence}\n")
        
        return str(msa_path)
    
    def _apply_mutations(self, sequence: str, mutations: List[str]) -> str:
        """Apply mutations to a sequence"""
        sequence_list = list(sequence)
        for mutation in mutations:
            pos = int(mutation[1:-1]) - 1  # Convert to 0-based index
            new_aa = mutation[-1]
            sequence_list[pos] = new_aa
        return "".join(sequence_list)
    
    def _create_wt_config(self, position: int) -> ExperimentConfig:
        """Create configuration for WT masking at a position"""
        return ExperimentConfig(
            sequence=self.protein_config.sequence,
            jobname_prefix=f"{self.name}_wt_pos{position}",
            parent_path=str(self.wt_dir / f"pos_{position}"),
            masking_strategy=MaskingStrategy.ITERATIVE_SINGLE,
            positions=[position],
            mask_token=self.mask_token,
            num_recycles=self.defaults.get('num_recycles', 2),
            num_seeds=self.defaults.get('num_seeds', 2),
            setup_path=str(self.slurm_config.setup_path),
            pipeline_type="mask"
        )
    
    def _create_mutation_config(
        self,
        position: int,
        mutated_sequence: str,
        mutation_set: List[str]
    ) -> ExperimentConfig:
        """Create configuration for mutation analysis at a position"""
        mutation_name = "_".join(mutation_set)
        return ExperimentConfig(
            sequence=mutated_sequence,
            msa=self.wt_msa,
            jobname_prefix=f"{self.name}_{mutation_name}_pos{position}",
            parent_path=str(self.mutations_dir / mutation_name / f"pos_{position}"),
            masking_strategy=MaskingStrategy.ITERATIVE_SINGLE,
            positions=[position],
            mask_token=self.mask_token,
            num_recycles=self.defaults.get('num_recycles', 2),
            num_seeds=self.defaults.get('num_seeds', 2),
            setup_path=str(self.slurm_config.setup_path),
            pipeline_type="mutate_and_mask",
            mutations=mutation_set,
            use_wt_msa=True
        )
    
    def setup(self) -> None:
        """Set up iterative masking experiment"""
        # Add vanilla control
        self.controls.append(Control(
            "vanilla",
            self.protein_config,
            self.slurm_config,
            [Condition(mask=False, mutate=False)]
        ))
        
        # Add mutation controls for each mutation set
        for i, mutation_set in enumerate(self.mutations):
            self.controls.append(Control(
                f"mutation_set_{i}",
                self.protein_config,
                self.slurm_config,
                [Condition(mask=False, mutate=True)]
            ))
    
    def _run_wt_analysis(self) -> bool:
        """
        Run single position masking on WT sequence.
        
        Workflow:
        1. Generate MSA for WT sequence
        2. Run single position masking for each position
        """
        try:
            logger.info("Starting WT sequence analysis")
            
            # Generate MSA for WT sequence
            logger.info("Generating MSA for WT sequence")
            self.wt_msa = self._generate_msa(self.protein_config.sequence)
            
            # Run single position masking
            sequence_length = len(self.protein_config.sequence)
            for position in range(1, sequence_length + 1):
                logger.info(f"Running WT masking for position {position}")
                
                config = self._create_wt_config(position)
                job_manager = SlurmJobManager(
                    experiment_config=config,
                    slurm_config=self.slurm_config,
                    working_dir=str(self.wt_dir / f"pos_{position}")
                )
                
                success, failed_jobs = job_manager.run_experiment()
                self.wt_results[position] = success
                
                if not success:
                    logger.error(f"Failed to run WT masking for position {position}")
                    return False
            
            logger.info("Completed WT sequence analysis")
            return True
            
        except Exception as e:
            logger.error(f"Error in WT analysis: {str(e)}")
            return False
    
    def _run_mutation_analyses(self) -> bool:
        """
        Run analyses for each mutation set.
        
        Workflow:
        1. For each mutation set:
            - Apply mutations to target sequence
            - Use WT MSA with mutated target
            - Run single position masking
        """
        try:
            logger.info("Starting mutation analyses")
            
            for mutation_set in self.mutations:
                mutation_name = "_".join(mutation_set)
                logger.info(f"Processing mutation set: {mutation_name}")
                
                # Apply mutations to target sequence
                mutated_sequence = self._apply_mutations(
                    self.protein_config.sequence,
                    mutation_set
                )
                
                # Initialize results tracking for this mutation set
                self.mutation_results[mutation_name] = {}
                
                # Run single position masking
                sequence_length = len(mutated_sequence)
                for position in range(1, sequence_length + 1):
                    logger.info(f"Running masking for position {position} with mutations {mutation_name}")
                    
                    config = self._create_mutation_config(
                        position,
                        mutated_sequence,
                        mutation_set
                    )
                    
                    job_manager = SlurmJobManager(
                        experiment_config=config,
                        slurm_config=self.slurm_config,
                        working_dir=str(self.mutations_dir / mutation_name / f"pos_{position}")
                    )
                    
                    success, failed_jobs = job_manager.run_experiment()
                    self.mutation_results[mutation_name][position] = success
                    
                    if not success:
                        logger.error(
                            f"Failed to run masking for position {position} "
                            f"with mutations {mutation_name}"
                        )
                        return False
            
            logger.info("Completed mutation analyses")
            return True
            
        except Exception as e:
            logger.error(f"Error in mutation analyses: {str(e)}")
            return False
    
    def run(self) -> bool:
        """
        Run complete iterative masking experiment.
        
        Workflow:
        1. Run controls
        2. Run WT analysis
        3. Run mutation analyses
        """
        try:
            self.validate()
            self.setup()
            
            # Run controls first
            logger.info("Running controls")
            for control in self.controls:
                if not control.run():
                    logger.error("Failed to run controls")
                    return False
            
            # Run WT analysis
            if not self._run_wt_analysis():
                return False
            
            # Run mutation analyses
            if not self._run_mutation_analyses():
                return False
            
            logger.info("Completed iterative masking experiment successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in iterative experiment: {str(e)}")
            return False 
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of experiment results"""
        return {
            "wt_analysis": {
                "total_positions": len(self.wt_results),
                "successful_positions": sum(1 for success in self.wt_results.values() if success),
                "failed_positions": sum(1 for success in self.wt_results.values() if not success),
                "results": self.wt_results
            },
            "mutation_analyses": {
                mutation_set: {
                    "total_positions": len(results),
                    "successful_positions": sum(1 for success in results.values() if success),
                    "failed_positions": sum(1 for success in results.values() if not success),
                    "results": results
                }
                for mutation_set, results in self.mutation_results.items()
            }
        } 