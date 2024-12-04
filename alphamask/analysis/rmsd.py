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

logger = logging.getLogger(__name__)

@dataclass
class RMSDConfig:
    """Configuration for RMSD calculations."""
    atom_selection: str = "CA"  # "CA", "backbone", or "all"
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

class RMSDCalculator:
    """
    Handles RMSD calculations between protein structures.
    
    Attributes:
        config (RMSDConfig): Configuration for RMSD calculations
    """
    
    def __init__(self, config: RMSDConfig):
        self.config = config
        self._setup_jax_functions()
        
    def _setup_jax_functions(self) -> None:
        """Initialize JAX functions for RMSD calculation."""
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

        self._rmsd_parallel = jax.jit(jax.vmap(_rmsd, (None,0)))
        self._rmsd = _rmsd

    def extract_coordinates(self, pdb_file: Union[str, Path]) -> np.ndarray:
        """
        Extract atomic coordinates from PDB file based on configuration.
        
        Args:
            pdb_file: Path to PDB file
            
        Returns:
            Array of atomic coordinates
            
        Raises:
            ValueError: If no atoms found in specified region
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
            else:  # all atoms
                atoms.append(atom)

        if not atoms:
            raise ValueError(
                f"No atoms found in region {self.config.start_residue}-{self.config.end_residue}"
            )

        return np.array([atom.get_coord() for atom in atoms])

    def calculate_rmsd(
        self, 
        ref_coords: np.ndarray,
        model_coords: np.ndarray
    ) -> float:
        """
        Calculate RMSD between reference and model coordinates.
        
        Args:
            ref_coords: Reference structure coordinates
            model_coords: Model structure coordinates
            
        Returns:
            RMSD value
        """
        return float(self._rmsd(
            jnp.array(ref_coords),
            jnp.array(model_coords)
        ))

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
                rmsd1 = self.calculate_rmsd(ref_coords1, model_coords)
                rmsd2 = None
                if ref_coords2 is not None:
                    rmsd2 = self.calculate_rmsd(ref_coords2, model_coords)
                    
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