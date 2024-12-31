"""
Utilities for working with compressed AlphaFold prediction data.
"""
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import h5py
from pathlib import Path
import logging
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PredictionIdentifier:
    """Identifies a specific prediction."""
    seed: str
    model: str
    recycle: str

class CompressedPredictionReader:
    """Reads compressed AlphaFold prediction data."""
    
    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize reader.
        
        Args:
            base_path: Path to directory containing compressed files
        """
        self.base_path = Path(base_path)
        self._ca_data = None
        self._full_data = None
        
    def _load_npz(self, filename: str) -> Dict:
        """Load NPZ file."""
        npz_path = self.base_path / filename
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        return dict(np.load(npz_path, allow_pickle=True))
    
    def _get_h5_file(self, filename: str) -> h5py.File:
        """Get HDF5 file handle."""
        h5_path = self.base_path / filename
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        return h5py.File(h5_path, 'r')
    
    def get_ca_coordinates(self, pred_id: Optional[PredictionIdentifier] = None) -> np.ndarray:
        """
        Get CA coordinates for specified prediction or best prediction.
        
        Args:
            pred_id: Identifier for specific prediction, or None for best
            
        Returns:
            Array of CA coordinates
        """
        if self._ca_data is None:
            self._ca_data = self._load_npz("ca.npz")
            
        if pred_id is None:
            return self._ca_data['best']['ca_positions']
            
        return self._ca_data['seeds'][pred_id.seed][pred_id.model][pred_id.recycle]['ca_positions']
    
    def get_plddt(self, pred_id: Optional[PredictionIdentifier] = None) -> np.ndarray:
        """Get pLDDT scores."""
        if self._ca_data is None:
            self._ca_data = self._load_npz("ca.npz")
            
        if pred_id is None:
            return self._ca_data['best']['plddt']
            
        return self._ca_data['seeds'][pred_id.seed][pred_id.model][pred_id.recycle]['plddt']
    
    def extract_pdb(
        self, 
        pred_id: Optional[PredictionIdentifier] = None,
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Extract PDB file from compressed storage.
        
        Args:
            pred_id: Identifier for specific prediction, or None for best
            output_path: Path to write PDB file, or None to return string
            
        Returns:
            PDB file contents as string if output_path is None
        """
        with self._get_h5_file("full.h5") as h5f:
            if pred_id is None:
                atom_pos = h5f['best']['atom_positions'][:]
                plddt = h5f['best']['plddt'][:]
            else:
                group = h5f[f'seeds/{pred_id.seed}/{pred_id.model}/{pred_id.recycle}']
                atom_pos = group['atom_positions'][:]
                plddt = group['plddt'][:]
                
        # Convert to PDB format
        structure = Structure.Structure('pred')
        model = Model.Model(0)
        chain = Chain.Chain('A')
        
        for res_idx, (res_atoms, res_plddt) in enumerate(zip(atom_pos, plddt)):
            residue = Residue.Residue((' ', res_idx, ' '), 'GLY', '')
            
            for atom_idx, (atom_name, coords) in enumerate(zip(['N', 'CA', 'C', 'O'], res_atoms[:4])):
                atom = Atom.Atom(
                    atom_name,
                    coords,
                    res_plddt,  # B-factor stores pLDDT
                    1.0,        # Occupancy
                    ' ',        # Altloc
                    atom_name,  # Fullname
                    atom_idx,   # Serial number
                    element=atom_name[0]
                )
                residue.add(atom)
                
            chain.add(residue)
            
        model.add(chain)
        structure.add(model)
        
        # Write PDB file or return string
        io = PDBIO()
        io.set_structure(structure)
        if output_path is not None:
            io.save(str(output_path))
            return None
            
        from io import StringIO
        out = StringIO()
        io.save(out)
        return out.getvalue()

def get_prediction_ids(compressed_path: Union[str, Path]) -> List[PredictionIdentifier]:
    """Get list of all available predictions in compressed storage."""
    with h5py.File(Path(compressed_path) / "full.h5", 'r') as h5f:
        ids = []
        for seed in h5f['seeds'].keys():
            for model in h5f[f'seeds/{seed}'].keys():
                for recycle in h5f[f'seeds/{seed}/{model}'].keys():
                    ids.append(PredictionIdentifier(seed, model, recycle))
    return ids 

def extract_pdbs_from_experiments(
    config: Dict,
    protein_ids: List[str],
    models: Optional[List[str]] = None,
    seeds: Optional[List[str]] = None,
    recycles: Optional[List[str]] = None,
    best_only: bool = False
) -> bool:
    """Extract PDBs from compressed storage maintaining experiment hierarchy."""
    try:
        for protein_id in protein_ids:
            if protein_id not in config["proteins"]:
                logger.warning(f"Protein {protein_id} not found in config")
                continue
                
            protein_config = config["proteins"][protein_id]
            
            # Get protein hash from first experiment directory
            base_dir = Path.cwd()  # Use current directory as base
            protein_dirs = list(base_dir.glob(f"{protein_id}_*"))
            if not protein_dirs:
                logger.warning(f"No experiment directory found for {protein_id}")
                continue
            
            protein_dir = protein_dirs[0]  # Use first matching directory
            logger.info(f"Processing protein directory: {protein_dir}")
            
            # Process each experiment type
            experiment_types = {
                "iterative": protein_config.get("iterative_masking", {}).get("enabled", False),
                "apriori": protein_config.get("apriori_masking", {}).get("enabled", False),
                "frustra": protein_config.get("frustra_masking", {}).get("enabled", False)
            }
            
            for exp_type, enabled in experiment_types.items():
                if not enabled:
                    continue
                    
                exp_base_dir = protein_dir / exp_type
                if not exp_base_dir.exists():
                    logger.warning(f"Experiment directory not found: {exp_base_dir}")
                    continue
                
                if exp_type == "apriori":
                    # Handle each experiment in apriori
                    for exp in protein_config["apriori_masking"].get("experiments", []):
                        exp_dir = exp_base_dir / exp["name"]
                        if not exp_dir.exists():
                            continue
                            
                        # Process each condition directory
                        for condition in ["unmasked_unmutated", "masked_unmutated", 
                                        "unmasked_mutated", "masked_mutated"]:
                            condition_dir = exp_dir / condition
                            if not condition_dir.exists():
                                continue
                                
                            compressed_dir = condition_dir / "out" / "compressed"
                            if not compressed_dir.exists():
                                continue
                                
                            logger.info(f"Processing {compressed_dir}")
                            
                            # Extract PDBs from this directory
                            _extract_pdbs_from_directory(
                                compressed_dir=compressed_dir,
                                models=models,
                                seeds=seeds,
                                recycles=recycles,
                                best_only=best_only
                            )
                else:
                    # Handle iterative and frustra experiments
                    compressed_dir = exp_base_dir / "out" / "compressed"
                    if compressed_dir.exists():
                        logger.info(f"Processing {compressed_dir}")
                        _extract_pdbs_from_directory(
                            compressed_dir=compressed_dir,
                            models=models,
                            seeds=seeds,
                            recycles=recycles,
                            best_only=best_only
                        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error extracting PDBs: {str(e)}")
        return False

def _extract_pdbs_from_directory(
    compressed_dir: Path,
    models: Optional[List[str]] = None,
    seeds: Optional[List[str]] = None,
    recycles: Optional[List[str]] = None,
    best_only: bool = False
) -> None:
    """Extract PDBs from a specific compressed directory."""
    # Find H5 files
    h5_files = list(compressed_dir.glob("*_all_atoms.h5"))
    if not h5_files:
        logger.warning(f"No H5 files found in {compressed_dir}")
        return
    
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as h5f:
                # Create output directory
                out_dir = compressed_dir.parent / "pdbs"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract PDBs from 'pdbs' group
                if 'pdbs' not in h5f:
                    logger.warning(f"No 'pdbs' group in {h5_file}")
                    continue
                    
                pdbs_grp = h5f['pdbs']
                logger.debug(f"Found PDB group with keys: {list(pdbs_grp.keys())}")
                
                for pdb_name in pdbs_grp.keys():
                    # Apply filters if specified
                    if models and not any(m in pdb_name for m in models):
                        continue
                    if seeds and not any(f"seed_{s}" in pdb_name for s in seeds):
                        continue
                    if recycles and not any(f"_r{r}_" in pdb_name for r in recycles):
                        continue
                    
                    # If best_only is True, only extract files with "best" in the name
                    if best_only and "best" not in pdb_name:
                        continue
                        
                    try:
                        # Read PDB content
                        dataset = pdbs_grp[pdb_name]
                        if isinstance(dataset, h5py.Dataset):
                            # Read the data
                            data = dataset[:]
                            
                            try:
                                if isinstance(data, np.ndarray):
                                    if data.dtype.kind == 'S':  # String data
                                        pdb_content = b''.join(data).decode('utf-8')
                                    else:  # Numeric data
                                        pdb_content = data.tobytes().decode('utf-8')
                                else:
                                    pdb_content = str(data)
                                    
                                # Ensure the content looks like a PDB file
                                if not (pdb_content.startswith('MODEL') or pdb_content.startswith('ATOM')):
                                    logger.warning(f"Invalid PDB content in {pdb_name}")
                                    logger.debug(f"Content starts with: {pdb_content[:100]}")
                                    continue
                                
                                # Write PDB file (ensure .pdb extension)
                                out_name = f"{pdb_name}.pdb" if not pdb_name.endswith('.pdb') else pdb_name
                                pdb_path = out_dir / out_name
                                
                                with open(pdb_path, 'w') as f:
                                    f.write(pdb_content)
                                    
                                logger.info(f"Extracted {pdb_path}")
                                
                            except Exception as decode_error:
                                logger.error(f"Failed to decode PDB content for {pdb_name}")
                                logger.debug(f"Data type: {type(data)}, shape: {data.shape if isinstance(data, np.ndarray) else 'N/A'}, dtype: {data.dtype if isinstance(data, np.ndarray) else 'N/A'}")
                                logger.debug(f"Error: {str(decode_error)}")
                                continue
                            
                    except Exception as e:
                        logger.error(f"Error extracting {pdb_name}: {str(e)}")
                        continue
                            
        except Exception as e:
            logger.error(f"Error processing {h5_file}: {str(e)}")
            
        logger.info(f"Finished processing {h5_file}") 