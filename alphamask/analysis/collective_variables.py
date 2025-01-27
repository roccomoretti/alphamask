"""Collective variables calculation functionality."""
from typing import List, Tuple, Optional, Dict, Any, Union, Sequence
import numpy as np
import jax.numpy as jnp
import jax
from dataclasses import dataclass
from ..utils.compression import CompressedPrediction
import logging

logger = logging.getLogger(__name__)

@dataclass
class CVConfig:
    """Configuration for collective variables calculations."""
    calculate_cv1: bool = True  # Dihedral angle
    calculate_cv2: bool = True  # Distances
    # Residue numbers for CV1 (dihedral)
    cv1_residues: Tuple[int, int, int, int] = (151, 152, 153, 154)
    # Residue numbers for CV2 (distances)
    cv2_residues: Tuple[int, int, int] = (41, 58, 156)

@dataclass
class CVResult:
    """Container for collective variables calculation results."""
    cv1_dihedral: Optional[float] = None  # Dihedral angle
    cv2_d1: Optional[float] = None  # Distance K41-E58
    cv2_d2: Optional[float] = None  # Distance R156-E58
    cv2_diff: Optional[float] = None  # Difference cv2_d2 - cv2_d1
    model_name: Optional[str] = None
    plddt: Optional[float] = None  # Add pLDDT field

@jax.jit
def _calculate_dihedral(coords: jnp.ndarray) -> float:
    """Calculate dihedral angle between 4 points using JAX."""
    # Vectors between points
    b1 = coords[1] - coords[0]
    b2 = coords[2] - coords[1]
    b3 = coords[3] - coords[2]
    
    # Normal vectors
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    
    # Normalize vectors
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    
    # Calculate angle
    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2 / jnp.linalg.norm(b2))
    phi = jnp.arctan2(sin_phi, cos_phi)
    
    return phi  # Return radians instead of degrees

@jax.jit
def _calculate_distance(point1: jnp.ndarray, point2: jnp.ndarray) -> float:
    """Calculate Euclidean distance between two points using JAX."""
    return jnp.sqrt(jnp.sum((point1 - point2) ** 2))

class CVCalculator:
    """Handles collective variables calculations for protein structures."""
    
    def __init__(self, config: CVConfig):
        self.config = config
        
    def calculate_cvs(self, prediction: CompressedPrediction) -> CVResult:
        """Calculate collective variables for a prediction."""
        try:
            result = CVResult(model_name=prediction.name)
            
            # Get atom positions
            atom_positions = prediction.atom_positions  # shape: [n_res, n_atoms, 3]
            
            if self.config.calculate_cv1:
                # Get CA coordinates for CV1 residues
                cv1_coords = []
                for res in self.config.cv1_residues:
                    # Adjust residue number to 0-based index
                    res_idx = res - 1
                    # Get CA atom (index 1)
                    ca_coord = atom_positions[res_idx, 1]
                    cv1_coords.append(ca_coord)
                
                # Calculate dihedral
                cv1_coords = jnp.array(cv1_coords)
                result.cv1_dihedral = float(_calculate_dihedral(cv1_coords))
            
            if self.config.calculate_cv2:
                # Get coordinates for CV2 distances
                k41_nz = atom_positions[self.config.cv2_residues[0]-1, 4]  # NZ atom
                e58_cd = atom_positions[self.config.cv2_residues[1]-1, 4]  # CD atom
                r156_ne = atom_positions[self.config.cv2_residues[2]-1, 4]  # NE atom
                
                # Calculate distances
                result.cv2_d1 = float(_calculate_distance(k41_nz, e58_cd))
                result.cv2_d2 = float(_calculate_distance(r156_ne, e58_cd))
                # Calculate difference
                result.cv2_diff = result.cv2_d2 - result.cv2_d1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate CVs: {str(e)}")
            return CVResult(model_name=prediction.name) 