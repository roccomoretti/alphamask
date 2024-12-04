import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .setup import SetupAlphaFoldColabDesign
from .model import PrepModel, RunAlphaFold
from .msa import PrepInputs, MSAUtils

class DefaultPipeline:
    """
    Base pipeline for AlphaFold predictions.
    
    Attributes:
        params (Dict): Pipeline parameters
        required_parameters (List[str]): Required parameter names
        param_types (Dict): Parameter types
        param_ranges (Dict): Valid parameter ranges
    """
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        if yaml_file is not None:
            with open(yaml_file, "r") as f:
                params = yaml.safe_load(f)

        self.params = params or {}
        self._setup_parameter_definitions()
        self._validate_parameters()
        
        # Add storage for MSAs in base class
        self.stored_msas = {
            'original': None,
            'mutated': None,
            'masked': None,
            'deletion_matrix': None,
            'masked_deletion_matrix': None
        }

    def get_stored_msas(self) -> Dict:
        """Return stored MSAs for inspection"""
        return self.stored_msas

    def _setup_parameter_definitions(self) -> None:
        """Set up parameter definitions and constraints."""
        self.required_parameters = [
            "unified_memory", "parentPath", "setupPath", "sequence",
            "jobname", "copies", "msa_method", "custom_a3m_path",
            "pair_mode", "cov", "id", "qid", "do_not_filter",
            "template_mode", "pdb", "chain", "rm_template_seq",
            "propagate_to_copies", "do_not_align", "model_type",
            "rank_by", "debug", "use_initial_guess", "num_msa",
            "num_extra_msa", "use_cluster_profile", "model",
            "num_recycles", "recycle_early_stop_tolerance",
            "select_best_across_recycles", "use_mlm", "use_dropout",
            "seed", "num_seeds", "show_images", "masking_mode",
            "mask_msa", "mask_deletion_matrix", "cols", "cols_range",
            "mask_identity", "mutations"
        ]
        
        self.param_types = {
            "unified_memory": bool,
            "parentPath": str,
            "setupPath": str,
            "sequence": str,
            "jobname": str,
            "copies": int,
            "msa_method": str,
            "custom_a3m_path": str,
            "pair_mode": str,
            "cov": int,
            "id": int,
            "qid": int,
            "do_not_filter": bool,
            "template_mode": str,
            "pdb": str,
            "chain": str,
            "rm_template_seq": bool,
            "propagate_to_copies": bool,
            "do_not_align": bool,
            "model_type": str,
            "rank_by": str,
            "debug": bool,
            "use_initial_guess": bool,
            "num_msa": int,
            "num_extra_msa": int,
            "use_cluster_profile": bool,
            "model": str,
            "num_recycles": int,
            "recycle_early_stop_tolerance": float,
            "select_best_across_recycles": bool,
            "use_mlm": bool,
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
        }
        
        self.param_ranges = {
            "copies": list(range(1, 13)),
            "msa_method": ["mmseqs2", "single_sequence", "custom_fas", "custom_a3m", "custom_sto"],
            "pair_mode": ["unpaired_paired", "paired", "unpaired"],
            "cov": [0, 25, 50, 75, 90, 99],
            "id": [90, 100],
            "qid": [0, 10, 15, 20, 30],
            "template_mode": ["none", "mmseqs2", "custom"],
            "model_type": ["monomer (ptm)", "pseudo_multimer (v3)", "multimer (v3)", "auto"],
            "rank_by": ["auto", "plddt", "ptm"],
            "num_msa": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "num_extra_msa": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            "model": ["1", "2", "3", "4", "5", "all"],
            "recycle_early_stop_tolerance": [0.0, 0.5, 1.0],
        }

    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        for param in self.required_parameters:
            if param not in self.params:
                raise ValueError(f"{param} is a required parameter")

            if not isinstance(self.params[param], self.param_types[param]):
                raise ValueError(
                    f"Parameter {param} should be of type {self.param_types[param]}"
                )

            if (
                param in self.param_ranges
                and self.params[param] not in self.param_ranges[param]
            ):
                raise ValueError(
                    f"Parameter {param} should be one of {self.param_ranges[param]}"
                )

    def __getattr__(self, attr: str) -> Any:
        """Get parameter value."""
        if attr in self.params:
            return self.params[attr]
        raise AttributeError(f"Attribute {attr} not found")

    def _save_config(self, jobname: str, parentPath: str) -> None:
        """Save pipeline configuration."""
        config = {k: v for k, v in self.params.items()}
        config_path = Path(parentPath) / jobname / f"{jobname}_config.yaml"
        
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump(config, f)

    def run(self) -> str:
        """Run the pipeline."""
        # Setup
        predictor = SetupAlphaFoldColabDesign(
            self.unified_memory, self.parentPath, self.setupPath
        )
        predictor.setup()

        # Prepare inputs
        prep_inputs = PrepInputs(
            sequence=self.sequence,
            jobname=self.jobname,
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
            setupPath=self.setupPath,
            parentPath=self.parentPath,
            overwrite=self.overwrite,
            show_figures=self.show_figures,
        )
        
        prep_inputs.filter_options()
        prep_inputs.process_sequence()
        prep_inputs.get_msa()
        prep_inputs.use_templates()

        # Store original MSA
        self.stored_msas['original'] = prep_inputs.msa.copy()
        self.stored_msas['deletion_matrix'] = prep_inputs.deletion_matrix.copy()

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
            setupPath=self.setupPath,
            use_mlm=self.use_mlm,
            rm_template_seq=self.rm_template_seq,
        )
        
        prep_model.model_options()
        prep_model.initialize_model()
        prep_model.prep_inputs()
        prep_model.set_templates()
        prep_model.set_msa()
        prep_model.set_chainbreaks()
        prep_model.set_cyclic_constraints()

        # Run prediction
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
            parentPath=self.parentPath,
            masking_mode=self.masking_mode,
            mask_msa=self.mask_msa,
            mask_deletion_matrix=self.mask_deletion_matrix,
            cols=self.cols,
            cols_range=self.cols_range,
            mask_identity=self.mask_identity,
            mutations=self.mutations,
        )

        self._save_config(
            jobname=prep_inputs.jobname,
            parentPath=prep_inputs.parentPath
        )
        
        return run_alphafold.run() 

class MaskingPipeline(DefaultPipeline):
    """
    Pipeline for running AlphaFold predictions with masking.
    Extends DefaultPipeline with masking-specific functionality.
    """
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        super().__init__(params, yaml_file)
        self.required_parameters.extend([
            "masking_mode",
            "mask_msa",
            "mask_deletion_matrix",
            "cols",
            "cols_range",
            "mask_identity",
        ])
        self.param_types.update({
            "masking_mode": str,
            "mask_msa": bool,
            "mask_deletion_matrix": bool,
            "cols": list,
            "cols_range": list,
            "mask_identity": str,
        })
        self._validate_parameters()

    def _save_config(self, jobname: str, parentPath: str) -> None:
        """Save masking pipeline configuration."""
        config = {k: v for k, v in self.params.items()}
        config_path = Path(parentPath) / jobname / f"{jobname}_config.yaml"
        
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump(config, f)

    def run(self) -> str:
        """Run the masking pipeline."""
        print("Masking pipeline is being used")
        msa_utils = MSAUtils()
        
        # Setup
        predictor = SetupAlphaFoldColabDesign(
            self.unified_memory, self.parentPath, self.setupPath
        )
        predictor.setup()

        # Prepare inputs
        prep_inputs = PrepInputs(
            sequence=self.sequence,
            jobname=self.jobname,
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
            setupPath=self.setupPath,
            parentPath=self.parentPath,
            overwrite=self.overwrite,
            show_figures=self.show_figures,
        )
        
        prep_inputs.filter_options()
        prep_inputs.process_sequence()
        prep_inputs.get_msa()
        prep_inputs.use_templates()

        # Store original MSA
        self.stored_msas['original'] = prep_inputs.msa.copy()
        self.stored_msas['deletion_matrix'] = prep_inputs.deletion_matrix.copy()

        # Apply masking
        if self.masking_mode == "list":
            if self.mask_msa:
                print("Masking mode is set to list. Using cols list for masking.")
                print(f"The following columns will be masked in the msa: {self.cols}")
                msa_masked = msa_utils.mask_columns_list_msa(
                    arr=prep_inputs.msa,
                    cols=self.cols,
                    mask_identity=self.mask_identity,
                )
                self.stored_msas['masked'] = msa_masked.copy()
                
            if self.mask_deletion_matrix:
                print(f"The following columns will be masked in the deletion matrix: {self.cols}")
                deletion_matrix_masked = msa_utils.mask_columns_list_deletion_matrix(
                    arr=prep_inputs.deletion_matrix,
                    cols=self.cols
                )
                self.stored_msas['masked_deletion_matrix'] = deletion_matrix_masked.copy()
        elif self.masking_mode == "ranges":
            raise NotImplementedError("The masking_mode 'range' is not fully implemented yet.")
        elif self.masking_mode == "random":
            raise NotImplementedError("Random masking is not implemented yet")

        # Prepare model with masked MSA
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
            msa=msa_masked,
            deletion_matrix=deletion_matrix_masked,
            u_sub_lengths=prep_inputs.u_sub_lengths,
            u_cyclic=prep_inputs.u_cyclic,
            setupPath=self.setupPath,
            use_mlm=self.use_mlm,
            rm_template_seq=self.rm_template_seq,
        )
        
        prep_model.model_options()
        prep_model.initialize_model()
        prep_model.prep_inputs()
        prep_model.set_templates()
        prep_model.set_msa()
        prep_model.set_chainbreaks()
        prep_model.set_cyclic_constraints()

        # Run prediction
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
            parentPath=self.parentPath,
            masking_mode=self.masking_mode,
            mask_msa=self.mask_msa,
            mask_deletion_matrix=self.mask_deletion_matrix,
            cols=self.cols,
            cols_range=self.cols_range,
            mask_identity=self.mask_identity,
            mutations=self.mutations,
        )

        self._save_config(
            jobname=prep_inputs.jobname,
            parentPath=prep_inputs.parentPath
        )
        
        return run_alphafold.run() 

class MutatePipeline(MaskingPipeline):
    """
    Pipeline for running AlphaFold predictions with mutations.
    Extends MaskingPipeline with mutation-specific functionality.
    """
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        super().__init__(params, yaml_file)
        self.required_parameters.extend(["mutations"])
        self.param_types["mutations"] = list
        self._validate_parameters()

    def _save_config(self, jobname: str, parentPath: str) -> None:
        """Save mutation pipeline configuration."""
        config = {k: v for k, v in self.params.items()}
        config_path = Path(parentPath) / jobname / f"{jobname}_config.yaml"
        
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump(config, f)

    def run(self) -> str:
        """Run the mutation pipeline."""
        print("Mutate only pipeline is being used")
        msa_utils = MSAUtils()
        
        # Setup
        predictor = SetupAlphaFoldColabDesign(
            self.unified_memory, self.parentPath, self.setupPath
        )
        predictor.setup()

        # Prepare inputs
        prep_inputs = PrepInputs(
            sequence=self.sequence,
            jobname=self.jobname,
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
            setupPath=self.setupPath,
            parentPath=self.parentPath,
            overwrite=self.overwrite,
            show_figures=self.show_figures,
        )
        
        prep_inputs.filter_options()
        prep_inputs.process_sequence()
        prep_inputs.get_msa()
        prep_inputs.use_templates()

        # Store original MSA
        self.stored_msas['original'] = prep_inputs.msa.copy()
        self.stored_msas['deletion_matrix'] = prep_inputs.deletion_matrix.copy()

        # Apply mutations
        print(f"The following mutations will be performed: {self.mutations}")
        mutated_msa = msa_utils.mutate_first_sequence(prep_inputs.msa, self.mutations)
        self.stored_msas['mutated'] = mutated_msa.copy()
        
        # Prepare model with mutated MSA
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
            msa=mutated_msa,
            deletion_matrix=prep_inputs.deletion_matrix,
            u_sub_lengths=prep_inputs.u_sub_lengths,
            u_cyclic=prep_inputs.u_cyclic,
            setupPath=self.setupPath,
            use_mlm=self.use_mlm,
            rm_template_seq=self.rm_template_seq,
        )
        
        prep_model.model_options()
        prep_model.initialize_model()
        prep_model.prep_inputs()
        prep_model.set_templates()
        prep_model.set_msa()
        prep_model.set_chainbreaks()
        prep_model.set_cyclic_constraints()

        # Run prediction
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
            parentPath=self.parentPath,
            masking_mode=self.masking_mode,
            mask_msa=self.mask_msa,
            mask_deletion_matrix=self.mask_deletion_matrix,
            cols=self.cols,
            cols_range=self.cols_range,
            mask_identity=self.mask_identity,
            mutations=self.mutations,
        )

        self._save_config(
            jobname=prep_inputs.jobname,
            parentPath=prep_inputs.parentPath
        )
        
        return run_alphafold.run()


class MutateAndMaskingPipeline(MaskingPipeline):
    """
    Pipeline for running AlphaFold predictions with both mutations and masking.
    Extends MaskingPipeline with combined mutation and masking functionality.
    """
    
    def __init__(self, params: Optional[Dict] = None, yaml_file: Optional[str] = None):
        super().__init__(params, yaml_file)
        self.required_parameters.extend(["mutations"])
        self.param_types["mutations"] = list
        self._validate_parameters()
        # Add storage for MSAs
        self.stored_msas = {
            'original': None,
            'mutated': None,
            'masked': None,
            'deletion_matrix': None,
            'masked_deletion_matrix': None
        }

    def get_stored_msas(self) -> Dict:
        """Return stored MSAs for inspection"""
        return self.stored_msas

    def _save_config(self, jobname: str, parentPath: str, mutations: List[str]) -> None:
        """Save mutation and masking pipeline configuration."""
        config = {k: v for k, v in self.params.items()}
        mutations_str = "_".join(mutations)
        config_path = Path(parentPath) / jobname / f"{jobname}_{mutations_str}_mask_mut_config.yaml"
        
        if not config_path.exists():
            with open(config_path, "w") as f:
                yaml.dump(config, f)

    def run(self) -> str:
        """Run the mutation and masking pipeline."""
        print("Mutate and Mask pipeline is being used")
        msa_utils = MSAUtils()
        
        # Setup
        predictor = SetupAlphaFoldColabDesign(
            self.unified_memory, self.parentPath, self.setupPath
        )
        predictor.setup()

        # Prepare inputs
        prep_inputs = PrepInputs(
            sequence=self.sequence,
            jobname=self.jobname,
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
            setupPath=self.setupPath,
            parentPath=self.parentPath,
            overwrite=self.overwrite,
            show_figures=self.show_figures,
        )
        
        prep_inputs.filter_options()
        prep_inputs.process_sequence()
        prep_inputs.get_msa()
        prep_inputs.use_templates()

        # Store original MSA
        self.stored_msas['original'] = prep_inputs.msa.copy()
        self.stored_msas['deletion_matrix'] = prep_inputs.deletion_matrix.copy()

        # Apply mutations and masking
        print(f"The following mutations will be performed: {self.mutations}")
        mutated_msa = msa_utils.mutate_first_sequence(prep_inputs.msa, self.mutations)
        self.stored_msas['mutated'] = mutated_msa.copy()
        
        print("The MSA will be masked in the mutated positions")
        masked_msa = msa_utils.mask_mutated_positions(
            mutated_msa, self.mutations, self.mask_identity
        )
        self.stored_msas['masked'] = masked_msa.copy()
        
        print("The deletion matrix will be masked in the mutated positions")
        masked_deletion_matrix = msa_utils.mask_mutated_positions_deletion_matrix(
            prep_inputs.deletion_matrix, self.mutations
        )
        self.stored_msas['masked_deletion_matrix'] = masked_deletion_matrix.copy()

        # Prepare model with mutated and masked MSA
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
            msa=masked_msa,
            deletion_matrix=masked_deletion_matrix,
            u_sub_lengths=prep_inputs.u_sub_lengths,
            u_cyclic=prep_inputs.u_cyclic,
            setupPath=self.setupPath,
            use_mlm=self.use_mlm,
            rm_template_seq=self.rm_template_seq,
        )
        
        prep_model.model_options()
        prep_model.initialize_model()
        prep_model.prep_inputs()
        prep_model.set_templates()
        prep_model.set_msa()
        prep_model.set_chainbreaks()
        prep_model.set_cyclic_constraints()

        # Run prediction
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
            parentPath=self.parentPath,
            masking_mode=self.masking_mode,
            mask_msa=self.mask_msa,
            mask_deletion_matrix=self.mask_deletion_matrix,
            cols=self.cols,
            cols_range=self.cols_range,
            mask_identity=self.mask_identity,
            mutations=self.mutations,
        )

        self._save_config(
            jobname=prep_inputs.jobname,
            parentPath=prep_inputs.parentPath,
            mutations=self.mutations
        )
        
        return run_alphafold.run()