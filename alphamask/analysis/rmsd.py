"""
Core RMSD calculation functionality for protein structure analysis.
"""
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence
import numpy as np
import jax.numpy as jnp
import jax
from pathlib import Path
from Bio import PDB
import logging
from dataclasses import dataclass
from ..utils.compression import CompressedPredictionReader, CompressedPrediction
import time
import traceback

logger = logging.getLogger(__name__)

@dataclass
class RMSDConfig:
    """Configuration for RMSD calculations."""
    atom_selection: str = "CA"  # "CA" or "all"
    start_residue: Optional[int] = None
    end_residue: Optional[int] = None
    use_region: bool = False

@dataclass
class RMSDResult:
    """Container for RMSD calculation results."""
    rmsd_ref1: float
    rmsd_ref2: Optional[float] = None
    model_name: Optional[str] = None
    plddt: Optional[float] = None

# Pre-compile JAX functions at module level for vectorized operations
@jax.jit
def _kabsch(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Kabsch algorithm using JAX."""
    u, s, vh = jnp.linalg.svd(a.T @ b, full_matrices=False)
    u = jnp.where(jnp.linalg.det(u @ vh) < 0, u.at[:,-1].set(-u[:,-1]), u)
    return u @ vh

@jax.jit
def _rmsd(true: jnp.ndarray, pred: jnp.ndarray) -> float:
    """Calculate RMSD using JAX."""
    p = true - true.mean(0, keepdims=True)
    q = pred - pred.mean(0, keepdims=True)
    p = p @ _kabsch(p, q)
    return jnp.sqrt(jnp.square(p-q).sum(-1).mean())

# Vectorized RMSD calculation
jnp_rmsd_parallel = jax.jit(jax.vmap(_rmsd, (None, 0)))

class RMSDCalculator:
    """
    Handles RMSD calculations between protein structures.
    
    Attributes:
        config (RMSDConfig): Configuration for RMSD calculations
    """
    
    def __init__(self, config: RMSDConfig):
        self.config = config
        
    def _calculate_rmsd(
        self, 
        ref_coords: np.ndarray,
        model_coords: np.ndarray
    ) -> float:
        """Calculate RMSD between reference and model coordinates."""
        # Create JAX functions locally each time
        @jax.jit
        def _kabsch(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
            u, s, vh = jnp.linalg.svd(a.T @ b, full_matrices=False)
            u = jnp.where(jnp.linalg.det(u @ vh) < 0, u.at[:,-1].set(-u[:,-1]), u)
            return u @ vh

        @jax.jit
        def _rmsd(true: jnp.ndarray, pred: jnp.ndarray) -> float:
            p = true - true.mean(0, keepdims=True)
            q = pred - pred.mean(0, keepdims=True)
            p = p @ _kabsch(p, q)
            return jnp.sqrt(jnp.square(p-q).sum(-1).mean())
            
        return float(_rmsd(
            jnp.array(ref_coords),
            jnp.array(model_coords)
        ))

    def calculate_rmsds(
        self,
        reader: CompressedPredictionReader,
        ref_coords1: np.ndarray,
        ref_coords2: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
        overwrite: bool = False
    ) -> List[RMSDResult]:
        """Calculate RMSDs for all predictions."""
        try:
            start_time = time.time()
            
            # Define output file path
            output_file = None
            if output_dir is not None:
                # Add _ca to filename if using CA atoms
                suffix = "_ca" if self.config.atom_selection.upper() == "CA" else "_all"
                output_file = output_dir / f"{Path(reader.file_path).stem}_rmsd{suffix}.npz"
                
                # Check if results already exist and we're not overwriting
                if output_file.exists() and not overwrite:
                    logger.debug(f"RMSD results already exist for {reader.file_path.name}, loading from file.")
                    # Load and convert npz data to RMSDResult objects
                    data = np.load(output_file)
                    results = []
                    for i in range(len(data['rmsd_ref1'])):
                        results.append(RMSDResult(
                            rmsd_ref1=float(data['rmsd_ref1'][i]),
                            rmsd_ref2=float(data['rmsd_ref2'][i]) if 'rmsd_ref2' in data else None,
                            plddt=float(data['plddt'][i]) if 'plddt' in data else None,
                            model_name=str(data['model_name'][i])
                        ))
                    return results

            # Get predictions
            predictions = reader.get_predictions()
            if not predictions:
                logger.warning(f"No valid predictions found in {reader.file_path}")
                return []

            # Pre-compute reference data
            ref1_centered = ref_coords1 - ref_coords1.mean(axis=0)
            ref2_centered = ref_coords2 - ref_coords2.mean(axis=0) if ref_coords2 is not None else None
            prep_time = time.time()
            
            # Extract and stack all coordinates
            coords_list = []
            valid_indices = []
            for i, pred in enumerate(predictions):
                coords = self.extract_coordinates(pred)
                if coords is not None:
                    coords_list.append(coords)
                    valid_indices.append(i)
            
            if not coords_list:
                return []
            
            # Stack coordinates into a single array
            all_coords = jnp.array(np.stack(coords_list))
            
            # Center all coordinates at once
            all_coords_centered = all_coords - all_coords.mean(axis=1, keepdims=True)
            
            # Calculate all RMSDs in parallel
            rmsd1_values = jnp_rmsd_parallel(ref1_centered, all_coords_centered)
            rmsd2_values = None
            if ref2_centered is not None:
                rmsd2_values = jnp_rmsd_parallel(ref2_centered, all_coords_centered)
            
            # Create results
            results = [None] * len(predictions)
            for idx, orig_idx in enumerate(valid_indices):
                results[orig_idx] = RMSDResult(
                    rmsd_ref1=float(rmsd1_values[idx]),
                    rmsd_ref2=float(rmsd2_values[idx]) if rmsd2_values is not None else None,
                    plddt=float(np.mean(predictions[orig_idx].plddt)) if predictions[orig_idx].plddt is not None else None,
                    model_name=predictions[orig_idx].name
                )
            
            # Remove any unused slots
            results = [r for r in results if r is not None]
            
            # Save results to npz file
            if output_file is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                rmsd_data = {
                    "rmsd_ref1": np.array([r.rmsd_ref1 for r in results]),
                    "rmsd_ref2": np.array([r.rmsd_ref2 for r in results if r.rmsd_ref2 is not None]),
                    "plddt": np.array([r.plddt for r in results]),
                    "model_name": np.array([r.model_name for r in results])
                }
                np.savez(output_file, **rmsd_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate RMSDs: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            return []

    def extract_coordinates(self, input_data: Union[str, Path, CompressedPrediction]) -> Optional[np.ndarray]:
        """Extract coordinates based on atom selection."""
        try:
            if isinstance(input_data, (str, Path)):
                # For reference PDB files, keep using BioPython
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure("protein", str(input_data))
                atoms = []
                
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if self.config.use_region and not (
                                self.config.start_residue <= residue.id[1] <= self.config.end_residue
                            ):
                                continue
                            
                            if self.config.atom_selection.upper() == "CA":
                                ca_atom = residue["CA"]
                                if ca_atom:
                                    atoms.append(ca_atom)
                            else:  # all atoms
                                atoms.extend(residue.get_atoms())
                
                if not atoms:
                    raise ValueError(
                        f"No atoms found with selection '{self.config.atom_selection}' "
                        f"in region {self.config.start_residue}-{self.config.end_residue}"
                    )
                
                return np.array([atom.get_coord() for atom in atoms])
                
            elif isinstance(input_data, CompressedPrediction):
                # For compressed predictions, directly access atom positions array
                atom_positions = input_data.atom_positions  # shape: [n_res, n_atoms, 3]
                
                if self.config.atom_selection.upper() == "CA":
                    # Extract CA coordinates (index 1 in atom dimension)
                    coords = atom_positions[:, 1]  # shape: [n_res, 3]
                else:
                    # Use all atoms
                    coords = atom_positions.reshape(-1, 3)  # shape: [n_res * n_atoms, 3]
                
                # Apply region selection if configured
                if self.config.use_region:
                    start_idx = self.config.start_residue - 1
                    end_idx = self.config.end_residue
                    coords = coords[start_idx:end_idx]
                
                return coords
            
            raise ValueError(f"Unsupported input type: {type(input_data)}")
            
        except Exception as e:
            logger.error(f"Failed to extract coordinates: {str(e)}")
            return None

    def extract_plddt(self, pdb_file: Union[str, Path]) -> float:
        """
        Extract mean pLDDT score from PDB file B-factor column.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Mean pLDDT score
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", str(pdb_file))

        atoms = []
        for atom in structure.get_atoms():
            residue = atom.get_parent()
            if self.config.use_region and not (
                self.config.start_residue <= residue.id[1] <= self.config.end_residue
            ):
                continue
                
            if self.config.atom_selection == "CA":
                if atom.get_name() == "CA":
                    atoms.append(atom)
            elif self.config.atom_selection == "backbone":
                if atom.get_name() in ["N", "CA", "C", "O"]:
                    atoms.append(atom)
            else:
                atoms.append(atom)

        return float(np.mean([atom.get_bfactor() for atom in atoms]))

    def process_models(
        self,
        model_dir: Union[str, Path],
        ref_coords1: np.ndarray,
        ref_coords2: Optional[np.ndarray] = None,
        pattern: str = "*.pdb"
    ) -> List[RMSDResult]:
        """
        Process multiple models and calculate RMSDs.
        
        Args:
            model_dir: Directory containing model PDB files
            ref_coords1: First reference structure coordinates
            ref_coords2: Optional second reference structure coordinates
            pattern: Glob pattern for finding PDB files
            
        Returns:
            List of RMSD results for each model
        """
        model_dir = Path(model_dir)
        results = []
        
        for pdb_file in model_dir.glob(pattern):
            if "best_model" in str(pdb_file):
                continue
                
            try:
                model_coords = self.extract_coordinates(pdb_file)
                rmsd1 = self._calculate_rmsd(ref_coords1, model_coords)
                rmsd2 = None
                if ref_coords2 is not None:
                    rmsd2 = self._calculate_rmsd(ref_coords2, model_coords)
                    
                plddt = self.extract_plddt(pdb_file)
                
                results.append(RMSDResult(
                    rmsd_ref1=rmsd1,
                    rmsd_ref2=rmsd2,
                    model_name=pdb_file.name,
                    plddt=plddt
                ))
                
            except Exception as e:
                logger.warning(f"Failed to process {pdb_file}: {str(e)}")
                
        return results 