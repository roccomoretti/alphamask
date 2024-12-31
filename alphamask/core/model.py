import os
import gc
import jax
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
from pathlib import Path
from datetime import datetime
import logging
import pickle
from colabdesign import mk_af_model, clear_mem
from colabdesign.af.contrib.cyclic import add_cyclic_offset
from colabdesign.shared.protein import _np_rmsd, _np_kabsch
from colabdesign.shared.plot import plot_pseudo_3D, pymol_cmap
from ..utils.types import AlphaFoldResult, AlphaFoldTempData, AlphaFoldAuxData
import h5py
from dataclasses import dataclass
from colabdesign.af.alphafold.common import protein

@dataclass
class CompressionConfig:
    """Configuration for compression storage
    
    Attributes:
        compress_format: Format for compressed storage ("h5", "npz", or "both")
        compression_level: Level of compression (1-9)
        store_uncompressed_pdbs: Whether to save individual PDBs in uncompressed format
        store_best_pdb: Whether to save best PDB in uncompressed format (defaults to True)
        
    By default:
        - Saves both H5 and NPZ compressed files
        - Only saves best PDB in uncompressed format
        - All other PDBs are stored in H5 format
    """
    compress_format: Literal["h5", "npz", "both"] = "both"
    compression_level: int = 9
    store_uncompressed_pdbs: bool = False  # Don't save individual uncompressed PDBs by default
    store_best_pdb: bool = True  # Always save best PDB uncompressed by default

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
        setup_path (str): Path to setup directory
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
        setup_path: str,
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
        self.setup_path = Path(setup_path)
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
                    data_dir=str(self.setup_path),
                    **self.model_opts
                )
                self._prev_model_opts = self.model_opts.copy()
        else:
            print("loading alphafold params")
            self.af = mk_af_model(
                use_mlm=self.use_mlm, 
                data_dir=str(self.setup_path),
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
    
    The predictions are saved in a single pickle file with the following structure:
    {
        'seeds': {
            '1': {  # Seed number
                'model_1': {  # Model name
                    'recycle_0': {  # Recycle iteration
                        'backbone_atoms': {  # Backbone atom coordinates
                            'N': array[...],   # Nitrogen coordinates (N x 3)
                            'CA': array[...],  # Alpha carbon coordinates (N x 3)
                            'C': array[...],   # Carbon coordinates (N x 3)
                            'O': array[...],   # Oxygen coordinates (N x 3)
                        },
                        'plddt': array[...],      # pLDDT scores (N)
                        'pae': array[...],        # PAE matrix (N x N)
                        'metrics': {              # Prediction metrics
                            'plddt': float,       # Average pLDDT
                            'ptm': float,         # PTM score
                            'i_ptm': float,       # Interface PTM (if multimer)
                            'multi': float        # Combined score (if multimer)
                        }
                    },
                    'recycle_1': {...},
                    ...
                },
                'model_2': {...},
                ...
            },
            '2': {...},
            ...
        },
        'best': {  # Best prediction across all seeds/models/recycles
            'backbone_atoms': {     # Backbone atom coordinates
                'N': array[...],    # Nitrogen coordinates (N x 3)
                'CA': array[...],   # Alpha carbon coordinates (N x 3)
                'C': array[...],    # Carbon coordinates (N x 3)
                'O': array[...],    # Oxygen coordinates (N x 3)
            },
            'plddt': array[...],      # pLDDT scores (N)
            'pae': array[...],        # PAE matrix (N x N)
            'tag': str,               # Identifier of best prediction
            'metrics': dict,          # Metrics of best prediction
            'sequence_length': int    # Length of the sequence
        }
    }
    
    Note on atom_positions array structure:
    The atom_positions array from AlphaFold has shape (N, 37, 3) where:
    - N is the number of residues
    - 37 is the number of atoms per residue
    - 3 is the XYZ coordinates
    
    The first 4 indices in the second dimension are backbone atoms:
    - 0: N (nitrogen)
    - 1: CA (alpha carbon) 
    - 2: C (carbon)
    - 3: O (oxygen)
    - 4+: Side chain atoms (not stored)
    
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
        parent_path (Path): Parent directory path
        masking_mode (str): Mode of masking
        mask_msa (bool): Whether to mask MSA
        mask_deletion_matrix (bool): Whether to mask deletion matrix
        cols (List[int]): Columns to mask
        cols_range (List[Tuple[int, int]]): Ranges of columns to mask
        mask_identity (str): Identity to use for masking
        mutations (List[str]): List of mutations
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize RunAlphaFold."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize predictions storage
        self.all_predictions = {
            'seeds': {},  # Will store data for each seed
            'best': None  # Will store the best prediction
        }
        
        # Store all initialization parameters
        for key, value in kwargs.items():
            if key == 'parent_path':
                self.logger.debug(f"Converting parent_path from {type(value)} to Path: {value}")
                value = Path(value) if not isinstance(value, Path) else value
            setattr(self, key, value)
            
        # Set default for compressed storage and parent_dir
        self.use_compressed_storage = kwargs.get('use_compressed_storage', False)
        self.use_parent_dir = kwargs.get('use_parent_dir', False)
        
        self.logger.debug(f"Initialized with parent_path: {self.parent_path} (type: {type(self.parent_path)})")
        
        # Initialize models list
        if hasattr(self, 'model') and self.model == "all":
            self.models = self.af._model_names
        else:
            model_idx = int(self.model) - 1
            self.models = [self.af._model_names[model_idx]]
        
        self.logger.debug(f"Initialized models: {self.models}")

        self.compression_config = kwargs.get('compression_config', CompressionConfig())

    def _get_pdb_filename(self, model: str, recycle: int, seed: int) -> str:
        """Generate PDB filename based on parameters."""
        base = f"{self.jobname}_{model}_r{recycle}_seed_{str(seed).zfill(3)}"
        
        if self.mask_msa:
            # Handle masking case (with or without mutations)
            if self.masking_mode == "list":
                cols_str = (
                    str(self.cols[0])
                    if len(self.cols) == 1
                    else f"{self.cols[0]}-{self.cols[-1]}" if self.cols else "false"
                )
                if self.mutations:
                    cols_str += "_mut_" + "_".join(self.mutations)
                return f"{base}_mask_{cols_str}_id_{self.mask_identity}.pdb"
        elif hasattr(self, 'mutations') and self.mutations:
            # Handle mutation-only case
            return f"{base}_mut_" + "_".join(self.mutations) + ".pdb"
        
        return f"{base}.pdb"

    def _extract_backbone_atoms(self, atom_positions: np.ndarray) -> Dict:
        """Extract backbone atoms from atom positions array."""
        if atom_positions.shape[1] < 4:
            raise ValueError(f"Expected at least 4 atoms per residue, got {atom_positions.shape[1]}")
            
        return {
            'N': atom_positions[:, 0].copy(),   # Nitrogen
            'CA': atom_positions[:, 1].copy(),  # Alpha carbon
            'C': atom_positions[:, 2].copy(),   # Carbon
            'O': atom_positions[:, 3].copy(),   # Oxygen
        }

    def save_pdb_fixed(self, filename=None, aux=None) -> str:
        """Fixed version of save_pdb that maintains array dimensions."""
        if aux is None:
            aux = self.af._tmp["best"]["aux"] if ("best" in self.af._tmp and "aux" in self.af._tmp["best"]) else self.af.aux
        aux = aux["all"]
        
        # Collect required fields
        p = {
            "aatype": aux["aatype"],
            "residue_index": aux["residue_index"],
            "atom_positions": aux["atom_positions"],
            "atom_mask": aux["atom_mask"]
        }
        p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][..., None]
        
        # Convert protein object to PDB string directly
        p_str = protein.to_pdb(protein.Protein(**p))
        
        if filename is None:
            return p_str
        else:
            with open(filename, 'w') as f:
                f.write(p_str)
            return p_str

    def _save_compressed_data(self, base_name: str, aux_best: Dict, info: List, rank: List[int]) -> None:
        """Save compressed prediction data."""
        try:
            # Determine output directory based on use_parent_dir flag
            if self.use_parent_dir:
                compressed_dir = self.parent_path / "out" / "compressed"
            else:
                compressed_dir = self.parent_path / self.jobname / "out" / "compressed"
            
            compressed_dir.mkdir(parents=True, exist_ok=True)
            
            # Debug logging
            self.logger.debug(f"all_predictions structure: {self.all_predictions.keys()}")
            
            # Save all CA coordinates in NPZ format
            npz_path = compressed_dir / f"{base_name}_all_ca.npz"
            self.logger.info(f"Saving all CA coordinates to {npz_path}")
            
            # Prepare data for NPZ - collect all CA coordinates
            ca_data = {}
            for seed_id, seed_data in self.all_predictions['seeds'].items():
                for model_name, model_data in seed_data.items():
                    for recycle_id, recycle_data in model_data.items():
                        # Fix recycle ID format - remove 'recycle_' prefix if present
                        clean_recycle_id = recycle_id.replace('recycle_', '')
                        key = f"{model_name}_r{clean_recycle_id}_seed_{seed_id}"
                        
                        if 'atom_positions' in recycle_data:
                            ca_data[f"{key}_ca"] = recycle_data['atom_positions'][:, 1].copy()  # CA atoms
                            ca_data[f"{key}_plddt"] = recycle_data.get('plddt', None)
                            if 'pae' in recycle_data:
                                ca_data[f"{key}_pae"] = recycle_data['pae']
                            self.logger.debug(f"Added CA coordinates for {key}")
                        else:
                            self.logger.warning(f"No atom positions found for {key}")
            
            # Save all CA coordinates
            self.logger.debug(f"Saving {len(ca_data)} entries to NPZ file")
            np.savez_compressed(npz_path, **ca_data)
            
            # Save full atom data and PDBs in H5 format
            if self.compression_config.compress_format in ["h5", "both"]:
                h5_path = compressed_dir / f"{base_name}_all_atoms.h5"
                self.logger.info(f"Saving all atom data to {h5_path}")
                
                with h5py.File(h5_path, 'w') as h5f:
                    # Create groups for different data types
                    pdbs_grp = h5f.create_group('pdbs')
                    data_grp = h5f.create_group('data')  # For storing other arrays
                    
                    for seed_id, seed_data in self.all_predictions['seeds'].items():
                        for model_name, model_data in seed_data.items():
                            for recycle_id, recycle_data in model_data.items():
                                clean_recycle_id = recycle_id.replace('recycle_', '')
                                
                                if 'atom_positions' in recycle_data:
                                    # Ensure arrays are numpy arrays with proper shapes
                                    temp_aux = {
                                        "all": {
                                            "aatype": np.asarray(recycle_data["aatype"]),
                                            "residue_index": np.asarray(recycle_data["residue_index"]),
                                            "atom_positions": np.asarray(recycle_data["atom_positions"]),
                                            "atom_mask": np.asarray(recycle_data["atom_mask"]),
                                            "plddt": np.asarray(recycle_data["plddt"])
                                        }
                                    }
                                    
                                    try:
                                        # Use our fixed save_pdb function instead
                                        pdb_content = self.save_pdb_fixed(filename=None, aux=temp_aux)
                                        self.logger.debug("Successfully generated PDB content")
                                        
                                        pdb_name = self._get_pdb_filename(
                                            model_name, 
                                            int(clean_recycle_id), 
                                            int(seed_id)
                                        )
                                        
                                        # Store PDB content as string dataset
                                        pdb_bytes = pdb_content.encode('utf-8')
                                        pdbs_grp.create_dataset(
                                            pdb_name,
                                            data=np.frombuffer(pdb_bytes, dtype='S1'),
                                            compression=self.compression_config.compression_level
                                        )
                                        
                                        # Store other data arrays in data_grp
                                        # Use the same base name as PDB files (without .pdb extension)
                                        data_prefix = self._get_pdb_filename(
                                            model_name,
                                            int(clean_recycle_id),
                                            int(seed_id)
                                        ).replace('.pdb', '')  # Remove .pdb extension
                                        
                                        for k, v in recycle_data.items():
                                            if isinstance(v, np.ndarray):
                                                data_grp.create_dataset(
                                                    f"{data_prefix}/{k}",
                                                    data=v,
                                                    compression=self.compression_config.compression_level
                                                )
                                        
                                    except Exception as e:
                                        self.logger.error(f"Failed to generate PDB: {str(e)}")
                                        self.logger.error(f"Array shapes that failed:")
                                        for k, v in temp_aux["all"].items():
                                            self.logger.error(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                                        raise
                                else:
                                    self.logger.warning(f"No atom positions found for {model_name}_{clean_recycle_id}_{seed_id}")
                                
            self.logger.debug("Successfully saved compressed data")
            
        except Exception as e:
            self.logger.error(f"Error saving compressed data: {str(e)}")
            raise

    def _save_results(self, info: List, rank: List[int], aux_best: Dict, base_dir: Path) -> None:
        """Save prediction results.
        
        This method handles saving:
        1. Best PDB file (if store_best_pdb=True)
        2. All PDB files in compressed H5 format
        3. CA coordinates in NPZ format
        """
        try:
            if self.use_parent_dir:
                base_dir = self.parent_path
                self.logger.info(f"Saving results directly to parent directory: {base_dir}")
            else:
                base_dir = self.parent_path / self.jobname
                self.logger.info(f"Saving results to job directory: {base_dir}")

            # Debug logging for compression settings
            self.logger.info(f"Compression config: {self.compression_config}")
            self.logger.info(f"Store uncompressed PDBs: {self.compression_config.store_uncompressed_pdbs}")
            self.logger.info(f"Store best PDB: {self.compression_config.store_best_pdb}")

            pdb_path = base_dir / "out"
            
            # Generate base name
            if self.mask_msa:
                # Handle masking case (with or without mutations)
                if self.masking_mode == "list":
                    # Format masked positions
                    if hasattr(self, 'cols') and self.cols:
                        if len(self.cols) == 1:
                            cols_str = str(self.cols[0])
                        else:
                            # Sort positions to ensure consistent naming
                            sorted_cols = sorted(self.cols)
                            # Group consecutive positions
                            ranges = []
                            current_range = [sorted_cols[0]]
                            
                            for pos in sorted_cols[1:]:
                                if pos == current_range[-1] + 1:
                                    current_range.append(pos)
                                else:
                                    ranges.append(current_range)
                                    current_range = [pos]
                            ranges.append(current_range)
                            
                            # Format ranges
                            range_strs = []
                            for r in ranges:
                                if len(r) == 1:
                                    range_strs.append(str(r[0]))
                                else:
                                    range_strs.append(f"{r[0]}-{r[-1]}")
                            cols_str = "_".join(range_strs)
                    else:
                        cols_str = "false"
                        
                    # Add mutations if present
                    if hasattr(self, 'mutations') and self.mutations:
                        cols_str += "_mut_" + "_".join(self.mutations)
                        
                    base_name = f"{self.jobname}_mask_{cols_str}_id_{self.mask_identity}"
                else:
                    raise NotImplementedError("The masking_mode 'range' is not fully implemented yet.")
            elif hasattr(self, 'mutations') and self.mutations:
                # Handle mutation-only case
                base_name = f"{self.jobname}_mut_" + "_".join(self.mutations)
            else:
                base_name = f"{self.jobname}"

            # Create output directories
            pdb_path.mkdir(parents=True, exist_ok=True)
            (pdb_path / "figs").mkdir(parents=True, exist_ok=True)
            (pdb_path / "compressed").mkdir(parents=True, exist_ok=True)
            
            # Only create pdbs directory if we're saving uncompressed files
            if self.compression_config.store_uncompressed_pdbs or self.compression_config.store_best_pdb:
                (pdb_path / "pdbs").mkdir(parents=True, exist_ok=True)

            # Save best PDB if configured
            if self.compression_config.store_best_pdb:
                best_pdb_path = pdb_path / "pdbs" / f"{base_name}_best.pdb"
                self.logger.info(f"Saving best PDB to {best_pdb_path}")
                self.af.save_pdb(best_pdb_path)

            self.logger.info("PDB files will only be stored in compressed H5 format")

            # Save compressed data (includes PDBs in H5 format)
            self._save_compressed_data(base_name, aux_best, info, rank)
            
            # Plot results if available
            try:
                self._plot_results(aux_best, base_dir)
            except Exception as e:
                self.logger.warning(f"Could not create plots: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in _save_results: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)
            raise

    def _plot_results(self, aux_best: Dict, base_dir: Path) -> None:
        """Plot prediction results."""
        try:
            pdb_path = base_dir / "out"
            
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

    def validate_input_files(self) -> Optional[str]:
        """Validate that all required input files exist."""
        try:
            # Use parent_path directly or with jobname based on use_parent_dir
            if self.use_parent_dir:
                input_dir = self.parent_path / "in"
            else:
                input_dir = self.parent_path / self.jobname / "in"
            
            msa_file = input_dir / "msa.a3m"
            filtered_msa = input_dir / "msa.filt.a3m"
            
            self.logger.info(f"Checking input directory: {input_dir}")
            self.logger.info(f"Expected MSA file: {msa_file}")
            self.logger.debug(f"Alternative filtered MSA file: {filtered_msa}")
            
            if not input_dir.exists():
                self.logger.debug(f"Input directory does not exist: {input_dir}")
                # Try to create the input directory
                try:
                    input_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created input directory at {input_dir}")
                except Exception as e:
                    return f"Failed to create input directory at {input_dir}: {str(e)}"

            # If we have a filtered MSA, that's also acceptable
            if filtered_msa.exists():
                self.logger.info(f"Found filtered MSA file at {filtered_msa}")
                return None
                
            if not msa_file.exists():
                self.logger.debug(f"MSA file not found at {msa_file}")
                # Check if MSA exists in parent directories
                parent_msa = None
                current_dir = input_dir.parent
                while current_dir != Path('/'):
                    possible_msa = current_dir / "in" / "msa.a3m"
                    self.logger.debug(f"Checking for MSA in parent directory: {possible_msa}")
                    if possible_msa.exists():
                        parent_msa = possible_msa
                        self.logger.info(f"Found MSA file in parent directory: {parent_msa}")
                        break
                    current_dir = current_dir.parent
                
                if parent_msa:
                    # Copy MSA from parent directory
                    try:
                        import shutil
                        input_dir.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"Copying MSA from {parent_msa} to {msa_file}")
                        shutil.copy2(parent_msa, msa_file)
                        return None
                    except Exception as e:
                        return f"Failed to copy MSA from {parent_msa} to {msa_file}: {str(e)}"
                else:
                    return f"MSA file not found at {msa_file} or in parent directories"
            
            return None
            
        except Exception as e:
            self.logger.error("Error during input validation", exc_info=True)
            return f"Error validating input files: {str(e)}"

    def get_aux_data(self) -> AlphaFoldAuxData:
        """Safely extract auxiliary data from temporary storage."""
        if not hasattr(self.af, '_tmp'):
            raise ValueError("No temporary data found in AlphaFold object")
            
        tmp_data: Dict[str, Any] = self.af._tmp
        if 'best' not in tmp_data:
            raise ValueError("No 'best' prediction found in temporary data")
            
        best_data = tmp_data['best']
        if 'aux' not in best_data:
            self.logger.error("Temporary data structure: %s", tmp_data)
            raise ValueError("No auxiliary data found in best prediction")
            
        return best_data['aux']

    def run(self) -> AlphaFoldResult:
        """Run AlphaFold prediction with error handling."""
        try:
            # Validate input files
            if error_msg := self.validate_input_files():
                self.logger.error(f"Input validation failed: {error_msg}")
                return AlphaFoldResult(success=False, error=error_msg, data=None)

            self.logger.info(f"Starting AlphaFold prediction for {self.jobname}")
            
            # Initialize tracking
            info = []
            self.af._tmp = {
                "traj": {"seq": [], "xyz": [], "plddt": [], "pae": []},
                "log": [],
                "best": {},
            }
            
            # Set options
            self.logger.debug("Setting MLM options")
            self.af.set_opt("mlm", replace_fraction=0.15 if self.use_mlm else 0.0)
            
            # Create output directories based on use_parent_dir
            if self.use_parent_dir:
                base_dir = self.parent_path
                self.logger.info(f"Using parent directory for outputs: {base_dir}")
            else:
                base_dir = self.parent_path / self.jobname
                self.logger.info(f"Using job-specific directory for outputs: {base_dir}")
            
            pdb_path = base_dir / "out" / "pdbs"
            self.logger.info(f"Creating PDB output directory: {pdb_path}")
            pdb_path.mkdir(parents=True, exist_ok=True)
            
            for subdir in ["figs", "npz"]:
                subdir_path = base_dir / "out" / subdir
                self.logger.debug(f"Creating output subdirectory: {subdir_path}")
                subdir_path.mkdir(parents=True, exist_ok=True)
            
            # Run prediction cycles
            self.logger.info("Running prediction cycles")
            with open(base_dir / "log.txt", "w") as handle:
                seeds = list(range(self.seed, self.seed + self.num_seeds))
                self.logger.debug(f"Using seeds: {seeds}")
                
                for seed in seeds:
                    self.logger.info(f"Processing seed {seed}")
                    self.af.set_seed(seed)
                    
                    for model in self.models:
                        self.logger.info(f"Processing model {model}")
                        recycle = 0
                        self.af._inputs.pop("prev", None)
                        stop_recycle = False
                        prev_pos = None
                        
                        while recycle < self.num_recycles + 1:
                            self.logger.info(f"Running recycle {recycle}")
                            # Run prediction cycle with updated base_dir
                            self._run_prediction_cycle(
                                handle, model, recycle, seed, prev_pos, info, base_dir
                            )
                            
                            # Check early stopping
                            current_pos = self.af.aux["atom_positions"][:, 1]
                            if recycle > 0:
                                rmsd_tol = _np_rmsd(prev_pos, current_pos, use_jax=False)
                                if rmsd_tol < self.recycle_early_stop_tolerance:
                                    self.logger.info(f"Early stopping at recycle {recycle} (RMSD: {rmsd_tol:.3f})")
                                    stop_recycle = True
                            prev_pos = current_pos
                            
                            if stop_recycle:
                                break
                            recycle += 1

            # Process results
            if not info:
                error_msg = "No predictions were completed successfully"
                self.logger.error(error_msg)
                return AlphaFoldResult(success=False, error=error_msg, data=None)

            # Get best prediction
            rank = np.argsort([x[2] for x in info])[::-1][:5]
            best_tag = f"best_tag={info[rank[0]][0]} {info[rank[0]][1]}"
            self.logger.info(best_tag)

            # Safely get auxiliary data
            try:
                aux_best = self.get_aux_data()
            except ValueError as e:
                self.logger.error(f"Failed to get auxiliary data: {str(e)}")
                return AlphaFoldResult(success=False, error=str(e), data=None)

            # Save and plot results with base_dir
            self._save_results(info, rank, aux_best, base_dir)
            self._plot_results(aux_best, base_dir)

            # Cleanup
            self.logger.debug("Running garbage collection")
            gc.collect()

            return AlphaFoldResult(
                success=True,
                error=None,
                data={"best": self.af._tmp["best"], "aux": aux_best}
            )

        except Exception as e:
            self.logger.error("Error during prediction", exc_info=True)
            return AlphaFoldResult(
                success=False,
                error=str(e),
                data=None
            )

    def _run_prediction_cycle(
        self, 
        handle: Any, 
        model: str, 
        recycle: int, 
        seed: int, 
        prev_pos: Optional[np.ndarray],
        info: List,
        base_dir: Path
    ) -> None:
        """Run a single prediction cycle and store results."""
        try:
            print_str = f"seed={str(seed).zfill(3)} model={model} recycle={recycle}"
            self.logger.info(f"Starting prediction cycle: {print_str}")
            print(print_str)
            
            # Run prediction
            self.logger.debug("Running AF2 prediction")
            self.af.predict(dropout=self.use_dropout, models=[model], verbose=False)
            self.af._inputs["prev"] = self.af.aux["prev"]

            # Process results
            if len(self.af._lengths) > 1:
                self.af.aux["log"]["multi"] = (
                    0.8 * self.af.aux["log"]["i_ptm"] + 0.2 * self.af.aux["log"]["ptm"]
                )

            # Initialize seed storage if needed
            if str(seed) not in self.all_predictions['seeds']:
                self.all_predictions['seeds'][str(seed)] = {}
            if model not in self.all_predictions['seeds'][str(seed)]:
                self.all_predictions['seeds'][str(seed)][model] = {}
                
            # Extract backbone atom coordinates
            atom_positions = self.af.aux["atom_positions"]
            self.logger.debug(f"Atom positions shape: {atom_positions.shape}")
            
            # Store prediction data by pulling directly from self.af.aux
            # (remove "['all']" references so we don't accidentally store a scalar)
            self.all_predictions['seeds'][str(seed)][model][f'recycle_{recycle}'] = {
                "aatype": self.af.aux["aatype"].copy(),            # shape [N]
                "residue_index": self.af.aux["residue_index"].copy(),  # shape [N]
                "atom_positions": self.af.aux["atom_positions"].copy(),# shape [N, 37, 3]
                "atom_mask": self.af.aux["atom_mask"].copy(),      # shape [N, 37]
                "plddt": self.af.aux["plddt"].copy(),              # shape [N]
                "pae": self.af.aux["pae"].copy() if "pae" in self.af.aux else None,
                "metrics": {k: self.af.aux["log"][k] for k in self.print_key if k in self.af.aux["log"]}
            }
            
            # Only save uncompressed PDB files if configured to do so
            if self.compression_config.store_uncompressed_pdbs:
                pdb_path = base_dir / "out" / "pdbs"
                pdb_file = pdb_path / self._get_pdb_filename(model, recycle, seed)
                self.logger.debug(f"Saving uncompressed PDB file: {pdb_file}")
                
                try:
                    self.af.save_pdb(str(pdb_file))
                    self.logger.debug(f"Successfully saved uncompressed PDB file: {pdb_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save uncompressed PDB file {pdb_file}: {str(e)}")
            
            # Save prediction metrics
            confidence = self.af.aux["plddt"].mean()
            self.logger.info(f"Prediction confidence: {confidence:.2f}")
            
            # Print key metrics
            for k in self.print_key:
                if k in self.af.aux["log"]:
                    v = self.af.aux["log"][k]
                    print(f"{k} {v:.4f}", file=handle)
                    self.logger.info(f"{k}: {v:.4f}")
            
            # Store best prediction
            if self.rank_by in self.af.aux["log"]:
                score = self.af.aux["log"][self.rank_by]
                tag = f"{model}_{recycle}_{seed}"
                info.append([tag, print_str, score])
                
                if (not hasattr(self.af, "_tmp") or 
                    "best" not in self.af._tmp or 
                    self.rank_by not in self.af._tmp["best"] or 
                    score > self.af._tmp["best"][self.rank_by]):
                    self.logger.info(f"New best prediction found: {tag} (score: {score:.4f})")
                    self.af._tmp["best"] = {
                        "aux": self.af.aux.copy(),
                        self.rank_by: score
                    }
                    self.all_predictions['best'] = {
                        'seed': seed,
                        'model': model,
                        'recycle': recycle,
                        'score': score,
                        'tag': tag
                    }
            
            self.logger.debug(f"Completed prediction cycle: {print_str}")
            
        except Exception as e:
            self.logger.error(f"Error in prediction cycle {print_str}", exc_info=True)
            raise
