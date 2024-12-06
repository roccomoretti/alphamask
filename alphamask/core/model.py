import os
import gc
import jax
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from colabdesign import mk_af_model, clear_mem
from colabdesign.af.contrib.cyclic import add_cyclic_offset
from colabdesign.shared.protein import _np_rmsd, _np_kabsch
from colabdesign.shared.plot import plot_pseudo_3D, pymol_cmap

class PrepModel:
    """
    Prepares the AlphaFold model for prediction.
    
    Attributes:
        model_type (str): Type of model to use
        rank_by (str): Metric to rank predictions by
        debug (bool): Whether to run in debug mode
        use_initial_guess (bool): Whether to use initial guess
        num_msa (int): Number of MSA sequences
        num_extra_msa (int): Number of extra MSA sequences
        use_cluster_profile (bool): Whether to use cluster profile
        u_lengths (List[int]): List of sequence lengths
        copies (int): Number of copies
        use_templates (bool): Whether to use templates
        batches (List): List of template batches
        msa (np.ndarray): MSA array
        deletion_matrix (np.ndarray): Deletion matrix
        u_sub_lengths (List[List[int]]): List of subsequence lengths
        u_cyclic (List[bool]): List of cyclic flags
        setupPath (str): Path to setup directory
        use_mlm (bool): Whether to use MLM
        rm_template_seq (bool): Whether to remove template sequence
    """
    
    def __init__(
        self,
        model_type: str,
        rank_by: str,
        debug: bool,
        use_initial_guess: bool,
        num_msa: int,
        num_extra_msa: int,
        use_cluster_profile: bool,
        u_lengths: List[int],
        copies: int,
        use_templates: bool,
        batches: List,
        msa: np.ndarray,
        deletion_matrix: np.ndarray,
        u_sub_lengths: List[List[int]],
        u_cyclic: List[bool],
        setupPath: str,
        use_mlm: bool,
        rm_template_seq: bool,
    ):
        self.model_type = model_type
        self.rank_by = rank_by
        self.debug = debug
        self.use_initial_guess = use_initial_guess
        self.num_msa = num_msa
        self.num_extra_msa = num_extra_msa
        self.use_cluster_profile = use_cluster_profile
        self.u_lengths = u_lengths
        self.copies = copies
        self.use_templates = use_templates
        self.batches = batches
        self.msa = msa
        self.deletion_matrix = deletion_matrix
        self.u_sub_lengths = u_sub_lengths
        self.u_cyclic = u_cyclic
        self.setupPath = Path(setupPath)
        self.use_mlm = use_mlm
        self.rm_template_seq = rm_template_seq
        self.rm_sidechain = self.rm_sequence = self.rm_template_seq
        self.rm_interchain = False
        self.af = None
        self.model_opts = None

    def model_options(self) -> None:
        """Set model options based on configuration."""
        if self.model_type == "monomer (ptm)":
            use_multimer = False
            pseudo_multimer = False
        elif self.model_type == "multimer (v3)":
            use_multimer = True
            pseudo_multimer = False
        elif self.model_type == "pseudo_multimer (v3)":
            use_multimer = True
            pseudo_multimer = True
        elif len(self.u_lengths) > 1 or self.copies > 1:
            use_multimer = True
            pseudo_multimer = False
        else:
            use_multimer = False
            pseudo_multimer = False

        if self.rank_by == "auto":
            self.rank_by = "multi" if (len(self.u_lengths) > 1 or self.copies > 1) else "plddt"

        self.model_opts = {
            "num_msa": self.num_msa,
            "num_extra_msa": self.num_extra_msa,
            "num_templates": len(self.batches),
            "use_cluster_profile": self.use_cluster_profile,
            "use_multimer": use_multimer,
            "pseudo_multimer": pseudo_multimer,
            "use_templates": self.use_templates,
            "use_batch_as_template": False,
            "use_dgram": True,
            "protocol": "hallucination",
            "best_metric": self.rank_by,
            "optimize_seq": False,
            "debug": self.debug,
            "clear_prev": False,
        }

    def initialize_model(self) -> None:
        """Initialize the AlphaFold model."""
        if hasattr(self, 'af') and self.af is not None:
            if self.model_opts != getattr(self, '_prev_model_opts', None):
                if (
                    self.model_opts["use_multimer"] == self.af._args["use_multimer"]
                    and self.model_opts["use_templates"] == self.af._args["use_templates"]
                ):
                    old_params = dict(zip(self.af._model_names, self.af._model_params))
                else:
                    print("loading alphafold params")
                    old_params = {}
                    clear_mem()
                self.af = mk_af_model(
                    old_params=old_params, 
                    use_mlm=self.use_mlm, 
                    data_dir=str(self.setupPath),
                    **self.model_opts
                )
                self._prev_model_opts = self.model_opts.copy()
        else:
            print("loading alphafold params")
            self.af = mk_af_model(
                use_mlm=self.use_mlm, 
                data_dir=str(self.setupPath),
                **self.model_opts
            )
            self._prev_model_opts = self.model_opts.copy()

    def prep_inputs(self) -> None:
        """Prepare model inputs."""
        self.af.prep_inputs(self.u_lengths, copies=self.copies, seed=0)
        self.print_key = ["plddt", "ptm"]
        if len(self.af._lengths) > 1:
            self.print_key += ["i_ptm", "multi"]
        self.af.set_opt("con", cutoff=8.0)

    def set_templates(self) -> None:
        """Set up templates if using them."""
        if self.use_templates:
            self.af.set_opt(use_initial_guess=self.use_initial_guess)
            for n, batch in enumerate(self.batches):
                self.af.set_template(batch=batch, n=n)
            self.af.set_opt(
                "template",
                rm_sc=self.rm_sidechain,
                rm_seq=self.rm_sequence,
                rm_ic=self.rm_interchain,
            )

    def set_msa(self) -> None:
        """Set the MSA and deletion matrix."""
        self.af.set_msa(self.msa, self.deletion_matrix)

    def set_chainbreaks(self) -> None:
        """Set chain breaks in the model."""
        L_prev = 0
        for n, l in enumerate(self.u_sub_lengths * self.copies):
            for L_i in l[:-1]:
                self.af._inputs["residue_index"][L_prev + L_i:] += 32
                L_prev += L_i
            L_prev += l[-1]

    def set_cyclic_constraints(self) -> None:
        """Set cyclic constraints if needed."""
        i_cyclic = [n for n, c in enumerate(self.u_cyclic * self.copies) if c]
        if len(i_cyclic) > 0:
            add_cyclic_offset(self.af, i_cyclic) 

class RunAlphaFold:
    """
    Runs AlphaFold predictions and handles results.
    
    Attributes:
        jobname (str): Name of the job
        model (str): Model to use
        num_recycles (int): Number of recycles
        recycle_early_stop_tolerance (float): Early stopping tolerance
        select_best_across_recycles (bool): Whether to select best across recycles
        use_mlm (bool): Whether to use MLM
        use_dropout (bool): Whether to use dropout
        seed (int): Random seed
        num_seeds (int): Number of seeds to use
        show_images (bool): Whether to show images
        use_initial_guess (bool): Whether to use initial guess
        af: AlphaFold model instance
        copies (int): Number of copies
        print_key (List[str]): Metrics to print
        rank_by (str): Metric to rank by
        Ls (List[int]): Sequence lengths
        parentPath (Path): Parent directory path
        masking_mode (str): Mode of masking
        mask_msa (bool): Whether to mask MSA
        mask_deletion_matrix (bool): Whether to mask deletion matrix
        cols (List[int]): Columns to mask
        cols_range (List[Tuple[int, int]]): Ranges of columns to mask
        mask_identity (str): Identity to use for masking
        mutations (List[str]): List of mutations
    """
    
    def __init__(
        self,
        jobname: str,
        model: str,
        num_recycles: int,
        recycle_early_stop_tolerance: float,
        select_best_across_recycles: bool,
        use_mlm: bool,
        use_dropout: bool,
        seed: int,
        num_seeds: int,
        show_images: bool,
        use_initial_guess: bool,
        af: Any,
        copies: int,
        print_key: List[str],
        rank_by: str,
        Ls: List[int],
        parentPath: str,
        masking_mode: str,
        mask_msa: bool,
        mask_deletion_matrix: bool,
        cols: List[int],
        cols_range: List[Tuple[int, int]],
        mask_identity: str,
        mutations: List[str],
    ):
        self.jobname = jobname
        self.model = model
        self.num_recycles = num_recycles
        self.recycle_early_stop_tolerance = recycle_early_stop_tolerance
        self.select_best_across_recycles = select_best_across_recycles
        self.use_mlm = use_mlm
        self.use_dropout = use_dropout
        self.seed = seed
        self.num_seeds = num_seeds
        self.show_images = show_images
        self.use_initial_guess = use_initial_guess
        self.af = af
        self.copies = copies
        self.print_key = print_key
        self.rank_by = rank_by
        self.Ls = Ls
        self.parentPath = Path(parentPath)
        self.masking_mode = masking_mode
        self.mask_msa = mask_msa
        self.mask_deletion_matrix = mask_deletion_matrix
        self.cols = cols
        self.cols_range = cols_range
        self.mask_identity = mask_identity
        self.mutations = mutations
        
    def _get_pdb_filename(self, model: str, recycle: int, seed: int) -> str:
        """Generate PDB filename based on parameters."""
        base = f"{self.jobname}_{model}_r{recycle}_seed_{str(seed).zfill(3)}"
        
        if self.mask_msa:
            if self.masking_mode == "list":
                cols_str = (
                    str(self.cols[0])
                    if len(self.cols) == 1
                    else f"{self.cols[0]}-{self.cols[-1]}" if self.cols else "false"
                )
                if self.mutations:
                    cols_str += "_mut_" + "_".join(self.mutations)
                return f"{base}_mask_{cols_str}_id_{self.mask_identity}.pdb"
        
        return f"{base}.pdb"

    def _save_results(self, info: List, rank: List[int], aux_best: Dict) -> None:
        """Save prediction results."""
        pdb_path = self.parentPath / self.jobname / "out"
        
        # Save best PDB file
        if self.mask_msa:
            if self.masking_mode == "list":
                cols_str = (
                    str(self.cols[0])
                    if len(self.cols) == 1
                    else f"{self.cols[0]}-{self.cols[-1]}" if self.cols else "false"
                )
                base_name = f"{self.jobname}_best_{info[rank[0]][0]}_mask_{cols_str}_id_{self.mask_identity}"
            else:
                raise NotImplementedError("The masking_mode 'range' is not fully implemented yet.")
        else:
            base_name = f"{self.jobname}_best_{info[rank[0]][0]}"

        # Save PDB
        self.af.save_pdb(pdb_path / "pdbs" / f"{base_name}.pdb")

        # Save NPZ files
        np.savez_compressed(
            pdb_path / "npz" / f"{base_name}.npz",
            plddt=aux_best["plddt"].astype(np.float16),
            pae=aux_best["pae"].astype(np.float16),
            tag=np.array(info[rank[0]][0]),
            metrics=np.array(info[rank[0]][1]),
        )

        np.savez_compressed(
            pdb_path / "npz" / f"{self.jobname}_all_{info[rank[0]][0]}.npz",
            plddt=np.array(self.af._tmp["traj"]["plddt"], dtype=np.float16),
            pae=np.array(self.af._tmp["traj"]["pae"], dtype=np.float16),
            tag=np.array([x[0] for x in info]),
            metrics=np.array([x[1] for x in info]),
        )

    def _plot_results(self, aux_best: Dict) -> None:
        """Plot prediction results."""
        try:
            pdb_path = self.parentPath / self.jobname / "out"
            
            # Plot 3D structure
            plt.figure(figsize=(10, 5))
            xyz = aux_best["atom_positions"][:, 1].copy()  # Make a copy of the coordinates
            
            # Get rotation matrix and apply it safely
            try:
                rotation = _np_kabsch(xyz, xyz, return_v=True, use_jax=False)
                if rotation is not None and rotation.size > 0:
                    xyz = np.matmul(xyz, rotation)
            except Exception as e:
                print(f"Warning: Could not apply Kabsch rotation: {e}")
            
            # Chain plot
            ax = plt.subplot(1, 2, 1)
            if len(self.Ls) > 1:
                plt.title("chain")
                c = np.concatenate([[n] * L for n, L in enumerate(self.Ls)])
                plot_pseudo_3D(xyz=xyz, c=c, cmap=pymol_cmap, cmin=0, cmax=39, Ls=self.Ls, ax=ax)
            else:
                plt.title("length")
                plot_pseudo_3D(xyz=xyz, Ls=self.Ls, ax=ax)
            plt.axis(False)
            
            # pLDDT plot
            ax = plt.subplot(1, 2, 2)
            plt.title("plddt")
            plot_pseudo_3D(xyz=xyz, c=aux_best["plddt"], cmin=0.5, cmax=0.9, Ls=self.Ls, ax=ax)
            plt.axis(False)
            
            plt.savefig(pdb_path / "figs/best.pdf", dpi=200, bbox_inches="tight")
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
            # Continue execution even if plotting fails

    def run(self) -> str:
        """Run AlphaFold prediction."""
        run_opts = {
            "seed": self.seed,
            "use_mlm": self.use_mlm,
            "use_dropout": self.use_dropout,
            "num_recycles": self.num_recycles,
            "model": self.model,
            "use_initial_guess": self.use_initial_guess,
            "select_best_across_recycles": self.select_best_across_recycles,
            "recycle_early_stop_tolerance": self.recycle_early_stop_tolerance,
        }

        # Decide which models to use
        models = (
            self.af._model_names if self.model == "all" 
            else [self.af._model_names[int(self.model) - 1]]
        )

        # Set options
        self.af.set_opt("mlm", replace_fraction=0.15 if self.use_mlm else 0.0)

        # Create output directories
        pdb_path = self.parentPath / self.jobname / "out"
        for subdir in ["figs", "pdbs", "npz"]:
            (pdb_path / subdir).mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        info = []
        self.af._tmp = {
            "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []},
            "log": [],
            "best": {},
        }

        # Run prediction
        print("running prediction")
        with open(self.parentPath / self.jobname / "log.txt", "w") as handle:
            seeds = list(range(self.seed, self.seed + self.num_seeds))
            
            for seed in seeds:
                self.af.set_seed(seed)
                
                for model in models:
                    recycle = 0
                    self.af._inputs.pop("prev", None)
                    stop_recycle = False
                    prev_pos = None
                    
                    while recycle < self.num_recycles + 1:
                        # Run prediction and process results
                        self._run_prediction_cycle(
                            handle, model, recycle, seed, prev_pos, info
                        )
                        
                        # Update previous position
                        current_pos = self.af.aux["atom_positions"][:, 1]
                        if recycle > 0:
                            rmsd_tol = _np_rmsd(prev_pos, current_pos, use_jax=False)
                            if rmsd_tol < self.recycle_early_stop_tolerance:
                                stop_recycle = True
                        prev_pos = current_pos
                        
                        if stop_recycle:
                            break
                        recycle += 1

        # Save and plot results
        if not info:
            print("Warning: No predictions were completed successfully")
            return self.jobname
            
        rank = np.argsort([x[2] for x in info])[::-1][:5]
        print(f"best_tag={info[rank[0]][0]} {info[rank[0]][1]}")
        
        aux_best = self.af._tmp["best"]["aux"]
        self._save_results(info, rank, aux_best)
        self._plot_results(aux_best)

        # Cleanup
        print("GC", gc.collect())
        return self.jobname

    def _run_prediction_cycle(
        self, 
        handle: Any, 
        model: str, 
        recycle: int, 
        seed: int, 
        prev_pos: Optional[np.ndarray],
        info: List
    ) -> None:
        """Run a single prediction cycle."""
        try:
            print_str = f"seed={str(seed).zfill(3)} model={model} recycle={recycle}"
            print(print_str)
            
            # Run prediction
            self.af.predict(dropout=self.use_dropout, models=[model], verbose=False)
            self.af._inputs["prev"] = self.af.aux["prev"]

            # Process results
            if len(self.af._lengths) > 1:
                self.af.aux["log"]["multi"] = (
                    0.8 * self.af.aux["log"]["i_ptm"] + 0.2 * self.af.aux["log"]["ptm"]
                )

            # Save PDB
            pdb_path = self.parentPath / self.jobname / "out"
            filename = self._get_pdb_filename(model, recycle, seed)
            self.af.save_current_pdb(pdb_path / "pdbs" / filename)

            # Print metrics
            for k in self.print_key:
                print_str += f" {k}={self.af.aux['log'][k]:.3f}"

            if prev_pos is not None:
                rmsd_tol = _np_rmsd(prev_pos, self.af.aux["atom_positions"][:, 1], use_jax=False)
                print_str += f" rmsd_tol={rmsd_tol:.3f}"

            handle.write(f"{print_str}\n")

            # Save results
            tag = f"{model}_r{recycle}_seed_{str(seed).zfill(3)}"
            if self.select_best_across_recycles:
                info.append([tag, print_str, self.af.aux["log"][self.rank_by]])
                self.af._save_results(
                    save_best=True,
                    best_metric=self.rank_by,
                    metric_higher_better=True,
                    verbose=False,
                )
                self.af._k += 1
            elif recycle == self.num_recycles:
                # If not selecting best across recycles, save only the final recycle
                info.append([tag, print_str, self.af.aux["log"][self.rank_by]])
                self.af._save_results(
                    save_best=True,
                    best_metric=self.rank_by,
                    metric_higher_better=True,
                    verbose=False,
                )
                self.af._k += 1

        except Exception as e:
            print(f"Error in prediction cycle: {e}")
            handle.write(f"Error in cycle: {e}\n")