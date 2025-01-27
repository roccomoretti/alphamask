"""Test script for collective variables calculation."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import re
import h5py
from tqdm import tqdm
from cv_test_plots import plot_cv_comparison_scatter, plot_cv_comparison_hist2d, plot_cv_recycle_comparison, create_cv_breakdown, plot_cv_summary_comparison
from typing import List, Dict
# Add parent directory to path to import alphamask
sys.path.append(str(Path(__file__).parent.parent))

#####################
# Use JAX for speed #
#####################
import jax
import jax.numpy as jnp
from jax import vmap
from colabdesign.af.alphafold.common import residue_constants

from alphamask.analysis.collective_variables import CVCalculator, CVConfig, CVResult
from alphamask.utils.compression import CompressedPredictionReader

# Define atom indices as constants for clarity and maintainability
NE_IDX = residue_constants.atom_order['NE']  # NE atom
NZ_IDX = residue_constants.atom_order['NZ']  # NZ atom
CD_IDX = residue_constants.atom_order['CD']  # CD atom

def extract_position(filename: str) -> int:
    """Extract position number from filename."""
    match = re.search(r'pos_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

# -----------------------------
# 1) Define a pure JAX function
#    to compute the CVs for a
#    single structure.
# -----------------------------
def compute_cvs_single_jax(atom_positions: jnp.ndarray,
                          cv1_res: tuple,
                          cv2_res: tuple) -> tuple:
    """
    JAX-compatible function to compute CVs matching cpptraj implementation:
    
    cpptraj commands:
    dihedral cv1 :151@CA :152@CA :153@CA :154@CA
    distance cv2_d1 :41@NZ :58@CD
    distance cv2_d2 :156@NE :58@CD
    """
    # Unpack residues
    r1, r2, r3, r4 = cv1_res  # 151, 152, 153, 154
    rA, rB, rRef = cv2_res    # 41, 58, 156
    
    # Get coordinates for dihedral
    xyz1 = atom_positions[r1-1, residue_constants.atom_order['CA']]
    xyz2 = atom_positions[r2-1, residue_constants.atom_order['CA']]
    xyz3 = atom_positions[r3-1, residue_constants.atom_order['CA']]
    xyz4 = atom_positions[r4-1, residue_constants.atom_order['CA']]

    # Calculate dihedral using normal vectors method
    # Vectors between points
    b1 = xyz2 - xyz1  # coords[1] - coords[0]
    b2 = xyz3 - xyz2  # coords[2] - coords[1]
    b3 = xyz4 - xyz3  # coords[3] - coords[2]
    
    # Normal vectors
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    
    # Normalize vectors
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    
    # Calculate angle
    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2 / jnp.linalg.norm(b2))
    dihedral_angle = jnp.arctan2(sin_phi, cos_phi)

    # Calculate CV2 (distance difference)
    rA_nz = atom_positions[rA-1, residue_constants.atom_order['NZ']]    # 41@NZ
    rB_cd = atom_positions[rB-1, residue_constants.atom_order['CD']]    # 58@CD
    rRef_ne = atom_positions[rRef-1, residue_constants.atom_order['NE']] # 156@NE

    d1 = jnp.linalg.norm(rA_nz - rB_cd)    # 41@NZ to 58@CD
    d2 = jnp.linalg.norm(rRef_ne - rB_cd)  # 156@NE to 58@CD

    return (dihedral_angle, d2 - d1)

# We'll jit + vmap this: 
# "pred_batch" will be (N, L, 37, 3)
# so we'll compute (N, 2) results
def compute_cvs_batch_jax(pred_batch: jnp.ndarray,
                         cv1_res: tuple,
                         cv2_res: tuple):
    """
    Batched CV computation using a single JAX call.
    pred_batch: (N, L, 37, 3) array
    Returns tuple of (dihedrals, diffs) arrays, each of shape (N,)
    """
    batched_func = jax.jit(vmap(compute_cvs_single_jax, in_axes=(0, None, None)))
    results = batched_func(pred_batch, cv1_res, cv2_res)
    return results  # returns (dihedrals, diffs)

def process_condition(input_file: Path, config: CVConfig) -> List[CVResult]:
    """Process a single condition file using batched JAX computation."""
    print(f"\nProcessing {input_file.name}...")
    reader = CompressedPredictionReader(input_file)
    predictions = reader.get_predictions()
    
    if not predictions:
        print(f"No predictions found in {input_file}")
        return []
    
    # Debug info
    print(f"\nFirst prediction atom_positions shape: {predictions[0].atom_positions.shape}")
    print(f"Atom indices from residue_constants:")
    print(f"CA index: {residue_constants.atom_order['CA']}")
    print(f"NZ index: {residue_constants.atom_order['NZ']}")
    print(f"CD index: {residue_constants.atom_order['CD']}")
    print(f"NE index: {residue_constants.atom_order['NE']}")
    
    # Collect coordinates in a single array
    coords_list = []
    valid_names = []
    ref_shape = predictions[0].atom_positions.shape
    
    for pred in predictions:
        if pred.atom_positions.shape != ref_shape:
            print(f"Warning: Inconsistent shape in {input_file}")
            continue
        coords_list.append(pred.atom_positions)
        valid_names.append(pred.name)
    
    if not coords_list:
        return []
    
    # Convert to JAX array and compute CVs in one batch
    coords_array = jnp.array(coords_list)
    cv_output = compute_cvs_batch_jax(
        coords_array,
        config.cv1_residues,
        config.cv2_residues
    )
    dihedrals, diffs = cv_output
    
    # Debug first few results
    print("\nFirst 3 results:")
    for i in range(min(3, len(dihedrals))):
        print(f"\nPrediction {i}:")
        print(f"Dihedral (rad): {float(dihedrals[i]):.3f}")
        print(f"Dihedral (deg): {float(jnp.degrees(dihedrals[i])):.3f}")
        print(f"Distance diff: {float(diffs[i]):.3f}")
    
    # Convert back to CVResult objects
    results = []
    for i, name in enumerate(valid_names):
        results.append(CVResult(
            cv1_dihedral=float(dihedrals[i]),
            cv2_diff=float(diffs[i]),
            model_name=name
        ))
    
    return results

def main():
    # Set up paths
    base_path = Path("/work/nw99ixuq-alphamask/my_experiments/her2_39b69/apriori/Abdullah_et_al_2023_T150A_L157R")
    input_files = {
        "unmasked_unmutated": base_path / "unmasked_unmutated/out/compressed/Abdullah_et_al_2023_T150A_L157R_39b69_all_atoms.h5",
        "unmasked_mutated": base_path / "unmasked_mutated/out/compressed/Abdullah_et_al_2023_T150A_L157R_39b69_mut_T150A_L157R_all_atoms.h5",
        "masked_unmutated": base_path / "masked_unmutated/out/compressed/Abdullah_et_al_2023_T150A_L157R_39b69_mask_150_157_id_X_all_atoms.h5",
        "masked_mutated": base_path / "masked_mutated/out/compressed/Abdullah_et_al_2023_T150A_L157R_39b69_mask_150_157_mut_T150A_L157R_id_X_all_atoms.h5"
    }
    
    output_dir = Path("tests/cv_results_apriori")
    output_dir.mkdir(exist_ok=True)
    
    # Save results to HDF5 as well
    output_file = output_dir / "cv_results.h5"
    
    # Initialize config
    config = CVConfig(
        calculate_cv1=True,
        calculate_cv2=True,
        cv1_residues=(151, 152, 153, 154),
        cv2_residues=(41, 58, 156)
    )
    
    # Process all conditions and store results
    all_results = {}
    with h5py.File(output_file, 'w') as f:
        # Create groups
        meta_group = f.create_group('metadata')
        meta_group.attrs['cv1_residues'] = config.cv1_residues
        meta_group.attrs['cv2_residues'] = config.cv2_residues
        
        for condition, input_file in input_files.items():
            print(f"\nProcessing condition: {condition}")
            
            # Process condition using JAX
            results = process_condition(input_file, config)
            all_results[condition] = results
            
            # Store in HDF5
            condition_group = f.create_group(condition)
            cv1_group = condition_group.create_group('cv1')
            cv2_group = condition_group.create_group('cv2')
            
            for i, result in enumerate(results):
                pred_str = f'pred_{i:03d}'
                if result.cv1_dihedral is not None:
                    cv1_group.create_dataset(pred_str, data=result.cv1_dihedral)
                if result.cv2_diff is not None:
                    cv2_group.create_dataset(pred_str, data=result.cv2_diff)
                
                # Store metadata
                if result.model_name:
                    match = re.search(r'model_(\d+).*_r(\d+)_', result.model_name)
                    if match:
                        cv1_group[pred_str].attrs['model'] = int(match.group(1))
                        cv1_group[pred_str].attrs['recycle'] = int(match.group(2))
                        cv2_group[pred_str].attrs['model'] = int(match.group(1))
                        cv2_group[pred_str].attrs['recycle'] = int(match.group(2))
    
    # Create comparison plots
    plot_cv_comparison_scatter(all_results, output_dir)
    plot_cv_comparison_hist2d(all_results, output_dir)
    plot_cv_recycle_comparison(all_results, output_dir)
    create_cv_breakdown(all_results, output_dir)
    plot_cv_summary_comparison(all_results, output_dir)

if __name__ == "__main__":
    main() 