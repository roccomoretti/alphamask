import os
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pickle

from .setup import SetupAlphaFoldColabDesign
from .model import PrepModel, RunAlphaFold
from .msa import PrepInputs, MSAUtils
from ..utils.types import AlphaFoldResult
from ..utils.params import load_defaults

class DefaultPipeline:
    """Base pipeline for running AlphaFold predictions."""
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize required parameters
        self.required_parameters = [
            "sequence",
            "jobname_prefix",
            "parent_path",
            "setup_path"
        ]
        
        # Initialize parameter types
        self.param_types = {
            "sequence": str,
            "jobname_prefix": str,
            "parent_path": str,
            "setup_path": str,
            "model_type": str,
            "rank_by": str,
            "debug": bool,
            "use_initial_guess": bool,
            "num_msa": int,
            "num_extra_msa": int,
            "use_cluster_profile": bool,
            "copies": int,
            "use_mlm": bool,
            "rm_template_seq": bool,
            "model": str,
            "num_recycles": int,
            "recycle_early_stop_tolerance": float,
            "select_best_across_recycles": bool,
            "use_dropout": bool,
            "seed": int,
            "num_seeds": int,
            "show_images": bool,
            "masking_mode": str,
            "mask_msa": bool,
            "mask_deletion_matrix": bool,
            "cols": list,
            "cols_range": list,
            "mask_identity": str,
            "mutations": list,
            "unified_memory": bool,
            "msa_method": str,
            "custom_a3m_path": str,
            "pair_mode": str,
            "do_not_filter": bool,
            "cov": int,
            "id": int,
            "qid": int,
            "template_mode": str,
            "pdb": str,
            "chain": str,
            "propagate_to_copies": bool,
            "do_not_align": bool,
            "overwrite": bool,
            "show_figures": bool,
            "pipeline_type": str
        }
        
        # Load defaults first
        self.params = load_defaults()
        
        # Then update with provided params
        if yaml_file:
            with open(yaml_file, 'r') as f:
                file_params = yaml.safe_load(f)
                self.params.update(file_params)
        if params:
            self.params.update(params)
            
        # Initialize attributes from params
        for key, value in self.params.items():
            setattr(self, key, value)
            
        # Convert paths to Path objects
        self._convert_paths()
        
        # Initialize storage for MSAs
        self.stored_msas = {
            'original': None,
            'mutated': None,
            'masked': None,
            'deletion_matrix': None
        }
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate required parameters are present and of correct type."""
        # Check required parameters are present
        for param in self.required_parameters:
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"Required parameter {param} is missing")
                
        # Ensure parent_path is set and exists
        if not hasattr(self, 'parent_path') or not self.parent_path:
            raise ValueError("parent_path must be provided")
            
        # Convert parent_path to Path object if it's a string
        if isinstance(self.parent_path, str):
            self.parent_path = Path(self.parent_path)
            
        # Create necessary directories
        self.parent_path.mkdir(parents=True, exist_ok=True)
            
        # Check parameter types
        for param, param_type in self.param_types.items():
            if hasattr(self, param):
                value = getattr(self, param)
                if value is not None and not isinstance(value, param_type):
                    try:
                        # Try to convert the value to the correct type
                        setattr(self, param, param_type(value))
                    except (ValueError, TypeError):
                        raise TypeError(
                            f"Parameter {param} should be of type {param_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                        
    def _convert_paths(self):
        """Convert path strings to Path objects."""
        path_params = ["parent_path", "setup_path"]
        for param in path_params:
            if hasattr(self, param):
                value = getattr(self, param)
                if value and not isinstance(value, Path):
                    self.logger.debug(f"Converting {param} from {type(value)} to Path: {value}")
                    setattr(self, param, Path(str(value)))
                    self.logger.debug(f"Converted {param} to: {getattr(self, param)}")
                    
    def _save_config(self, jobname: str, parent_path: str) -> None:
        """Save pipeline configuration."""
        config = {k: v for k, v in self.params.items()}
        config_path = Path(parent_path) / jobname / f"{jobname}_config.yaml"
        
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump(config, f)
                
    def _prepare_inputs(self) -> 'PrepInputs':
        """Prepare inputs for the pipeline."""
        self.logger.info("Preparing pipeline inputs")
        
        # Use jobname if provided, otherwise use jobname_prefix
        jobname = self.jobname if hasattr(self, 'jobname') and self.jobname else self.jobname_prefix
        self.logger.debug(f"Using jobname: {jobname}")
        
        # Ensure setup_path is set
        if not hasattr(self, 'setup_path') or not self.setup_path:
            raise ValueError("setup_path must be provided")
            
        # Initialize PrepInputs with all parameters
        prep_inputs = PrepInputs(
            sequence=self.sequence,
            jobname=jobname,
            copies=self.copies,
            msa_method=self.msa_method,
            custom_a3m_path=self.custom_a3m_path,
            pair_mode=self.pair_mode,
            cov=self.cov,
            id=self.id,
            qid=self.qid,
            do_not_filter=self.do_not_filter,
            template_mode=self.template_mode,
            pdb=self.pdb,
            chain=self.chain,
            rm_template_seq=self.rm_template_seq,
            propagate_to_copies=self.propagate_to_copies,
            do_not_align=self.do_not_align,
            setup_path=self.setup_path,
            parent_path=self.parent_path,
            overwrite=self.overwrite,
            show_figures=self.show_figures,
        )
        
        # Process inputs
        prep_inputs.process_sequence()
        prep_inputs.get_msa()
        prep_inputs.use_templates()
        
        # Store original MSA
        self.stored_msas['original'] = prep_inputs.msa.copy()
        self.stored_msas['deletion_matrix'] = prep_inputs.deletion_matrix.copy()
        
        return prep_inputs
        
    def _init_alphafold(self, prep_inputs: 'PrepInputs') -> 'RunAlphaFold':
        """Initialize AlphaFold with prepared inputs."""
        self.logger.info("Initializing AlphaFold model")
        
        # Prepare model with original MSA
        prep_model = PrepModel(
            model_type=self.model_type,
            rank_by=self.rank_by,
            debug=self.debug,
            use_initial_guess=self.use_initial_guess,
            num_msa=self.num_msa,
            num_extra_msa=self.num_extra_msa,
            use_cluster_profile=self.use_cluster_profile,
            u_lengths=prep_inputs.u_lengths,
            copies=self.copies,
            use_templates=prep_inputs.has_templates,
            batches=prep_inputs.batches,
            msa=prep_inputs.msa,
            deletion_matrix=prep_inputs.deletion_matrix,
            u_sub_lengths=prep_inputs.u_sub_lengths,
            u_cyclic=prep_inputs.u_cyclic,
            setup_path=self.setup_path,
            use_mlm=self.use_mlm,
            rm_template_seq=self.rm_template_seq,
        )
        
        prep_model.model_options()
        prep_model.initialize_model()
        prep_model.prep_inputs()
        
        if prep_inputs.has_templates:
            prep_model.set_templates()
            
        prep_model.set_msa()
        prep_model.set_chainbreaks()
        prep_model.set_cyclic_constraints()
        
        # Initialize RunAlphaFold
        run_alphafold = RunAlphaFold(
            jobname=prep_inputs.jobname,
            model=self.model,
            num_recycles=self.num_recycles,
            recycle_early_stop_tolerance=self.recycle_early_stop_tolerance,
            select_best_across_recycles=self.select_best_across_recycles,
            use_mlm=self.use_mlm,
            use_dropout=self.use_dropout,
            seed=self.seed,
            num_seeds=self.num_seeds,
            show_images=self.show_images,
            use_initial_guess=self.use_initial_guess,
            af=prep_model.af,
            copies=prep_inputs.input_opts["copies"],
            print_key=prep_model.print_key,
            rank_by=prep_model.rank_by,
            Ls=prep_inputs.Ls,
            parent_path=self.parent_path,
            masking_mode=self.masking_mode,
            mask_msa=self.mask_msa,
            mask_deletion_matrix=self.mask_deletion_matrix,
            cols=self.cols,
            cols_range=self.cols_range,
            mask_identity=self.mask_identity,
            mutations=self.mutations,
        )
        
        return run_alphafold
        
    def run(self) -> Union[str, AlphaFoldResult]:
        """Run the pipeline with error handling."""
        try:
            self.logger.info("Initializing AlphaMask Setup")
            predictor = SetupAlphaFoldColabDesign(
            self.unified_memory, self.parent_path, self.setup_path
            )
            predictor.setup()

            self.logger.info("Preparing pipeline inputs")
            prep_inputs = self._prepare_inputs()
            
            self.logger.info("Initializing AlphaFold")
            run_alphafold = self._init_alphafold(prep_inputs)
            
            self.logger.info("Saving configuration")
            self._save_config(
                jobname=prep_inputs.jobname,
                parent_path=prep_inputs.parent_path
            )
            
            # Run prediction and handle results
            result = run_alphafold.run()
            if not isinstance(result, AlphaFoldResult):
                self.logger.warning("Received legacy return type from AlphaFold run")
                return result
                
            if not result.success:
                self.logger.error("AlphaFold prediction failed: %s", result.error)
                
            return result
            
        except Exception as e:
            self.logger.error("Pipeline execution failed", exc_info=True)
            return AlphaFoldResult(
                success=False,
                error=f"Pipeline error: {str(e)}",
                data=None
            )
            
    def get_stored_msas(self) -> Dict:
        """Return stored MSAs for inspection"""
        return self.stored_msas


class MaskingPipeline(DefaultPipeline):
    """Pipeline for running AlphaFold predictions with masking."""
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        super().__init__(params, yaml_file)
        
        # Ensure setup_path is set
        if not hasattr(self, 'setup_path') or not self.setup_path:
            raise ValueError("setup_path must be provided for masking pipeline")
        
    def _apply_masking(self, prep_inputs: 'PrepInputs') -> Tuple[np.ndarray, np.ndarray]:
        """Apply masking to MSA and deletion matrix based on configuration."""
        self.logger.info("Applying masking operations")
        msa_utils = MSAUtils()
        
        msa = prep_inputs.msa.copy()
        deletion_matrix = prep_inputs.deletion_matrix.copy()
        
        if self.masking_mode == "list":
            self.logger.info(f"Using list mode masking with columns: {self.cols}")
            if self.mask_msa:
                self.logger.info(f"Masking MSA columns with identity: {self.mask_identity}")
                msa = msa_utils.mask_columns_list_msa(
                    arr=msa,
                    cols=self.cols,
                    mask_identity=self.mask_identity
                )
            if self.mask_deletion_matrix:
                self.logger.info("Masking deletion matrix columns")
                deletion_matrix = msa_utils.mask_columns_list_deletion_matrix(
                    arr=deletion_matrix,
                    cols=self.cols
                )
                
        elif self.masking_mode == "ranges":
            self.logger.info(f"Using ranges mode masking with ranges: {self.cols_range}")
            if self.mask_msa:
                self.logger.info(f"Masking MSA ranges with identity: {self.mask_identity}")
                msa = msa_utils.mask_columns_ranges_msa(
                    arr=msa,
                    cols_range=self.cols_range,
                    mask_identity=self.mask_identity
                )
            if self.mask_deletion_matrix:
                self.logger.info("Masking deletion matrix ranges")
                deletion_matrix = msa_utils.mask_columns_ranges_deletion_matrix(
                    arr=deletion_matrix,
                    cols_range=self.cols_range
                )
                
        elif self.masking_mode == "random":
            raise NotImplementedError("Random masking is not implemented yet")
            
        return msa, deletion_matrix
        
    def _prepare_inputs(self) -> 'PrepInputs':
        """Prepare inputs with masking applied."""
        self.logger.info("Preparing pipeline inputs")
        
        # Get base inputs from parent class
        prep_inputs = super()._prepare_inputs()
        
        # Apply masking operations
        masked_msa, masked_deletion_matrix = self._apply_masking(prep_inputs)
        
        # Store masked MSA
        self.stored_msas['masked'] = masked_msa.copy()
        
        # Update prep_inputs with masked data
        prep_inputs.msa = masked_msa
        prep_inputs.deletion_matrix = masked_deletion_matrix
        
        return prep_inputs


class MutatePipeline(MaskingPipeline):
    """Pipeline for running AlphaFold predictions with mutations."""
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        super().__init__(params, yaml_file)
        
    def _apply_mutations(self, prep_inputs: 'PrepInputs') -> np.ndarray:
        """Apply mutations to MSA."""
        self.logger.info("Applying mutations")
        msa_utils = MSAUtils()
        
        # Store original MSA
        self.stored_msas['original'] = prep_inputs.msa.copy()
        
        # Apply mutations
        self.logger.info(f"Applying mutations: {self.mutations}")
        mutated_msa = msa_utils.mutate_first_sequence(
            arr=prep_inputs.msa,
            mutations=self.mutations
        )
        self.stored_msas['mutated'] = mutated_msa.copy()
        
        return mutated_msa
        
    def _prepare_inputs(self) -> 'PrepInputs':
        """Prepare inputs with mutations applied."""
        self.logger.info("Preparing pipeline inputs")
        
        # Get base inputs from parent class
        prep_inputs = super()._prepare_inputs()
        
        # Apply mutations
        mutated_msa = self._apply_mutations(prep_inputs)
        
        # Update prep_inputs with mutated data
        prep_inputs.msa = mutated_msa
        
        return prep_inputs


class MutateAndMaskingPipeline(MaskingPipeline):
    """Pipeline for running AlphaFold predictions with both mutations and masking."""
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        super().__init__(params, yaml_file)
        
    def _apply_mutations_and_masking(self, prep_inputs: 'PrepInputs') -> Tuple[np.ndarray, np.ndarray]:
        """Apply both mutations and masking to MSA."""
        self.logger.info("Applying mutations and masking")
        msa_utils = MSAUtils()
        
        # Store original MSA
        self.stored_msas['original'] = prep_inputs.msa.copy()
        
        # First apply mutations
        self.logger.info(f"Applying mutations: {self.mutations}")
        mutated_msa = msa_utils.mutate_first_sequence(
            arr=prep_inputs.msa,
            mutations=self.mutations
        )
        self.stored_msas['mutated'] = mutated_msa.copy()
        
        # Then apply masking to the mutated MSA
        self.logger.info("Applying masking to mutated MSA")
        prep_inputs.msa = mutated_msa  # Update MSA before masking
        masked_msa, masked_deletion_matrix = self._apply_masking(prep_inputs)
        self.stored_msas['masked'] = masked_msa.copy()
        
        return masked_msa, masked_deletion_matrix
        
    def _prepare_inputs(self) -> 'PrepInputs':
        """Prepare inputs with both mutations and masking applied."""
        self.logger.info("Preparing pipeline inputs")
        
        # Get base inputs from parent class
        prep_inputs = super()._prepare_inputs()
        
        # Apply both mutations and masking
        masked_msa, masked_deletion_matrix = self._apply_mutations_and_masking(prep_inputs)
        
        # Update prep_inputs with mutated and masked data
        prep_inputs.msa = masked_msa
        prep_inputs.deletion_matrix = masked_deletion_matrix
        
        return prep_inputs