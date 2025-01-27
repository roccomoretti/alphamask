"""Test script for collective variables calculation - iterative case."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import re
import h5py
from tqdm import tqdm
from cv_test_plots import plot_cv_summary_comparison
import logging

# Add parent directory to path to import alphamask
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import vmap
from colabdesign.af.alphafold.common import residue_constants

from alphamask.analysis.collective_variables import CVCalculator, CVConfig, CVResult
from alphamask.utils.compression import CompressedPredictionReader

def extract_mutation_info(path: Path) -> str:
    """Extract mutation info from path."""
    # Example: .../T150A_L157R/out/compressed/...
    match = re.search(r'/([^/]+)/out/compressed', str(path))
    if match:
        return match.group(1)
    return "unknown"

def process_mutation_dir(
    compressed_dir: Path,
    config: CVConfig,
    results_file: Path,
    overwrite: bool = False
) -> Dict[int, List[CVResult]]:
    """Process all h5 files in a mutation directory."""
    mutation = extract_mutation_info(compressed_dir)
    
    # Check if results exist
    if not overwrite and results_file.exists():
        with h5py.File(results_file, 'r') as f:
            if mutation in f:
                print(f"Loading cached results for {mutation}")
                results = {}
                mutation_group = f[mutation]
                for pos_key in mutation_group.keys():
                    pos = int(pos_key.replace('pos_', ''))
                    pos_group = mutation_group[pos_key]
                    results[pos] = []
                    for pred_key in pos_group.keys():
                        pred = pos_group[pred_key]
                        results[pos].append(CVResult(
                            cv1_dihedral=float(pred['cv1_dihedral'][()]),
                            cv2_diff=float(pred['cv2_diff'][()]),
                            model_name=str(pred.attrs['model_name'])
                        ))
                return results

    # Process each h5 file
    results = {}
    h5_files = list(compressed_dir.glob('*.h5'))
    print(f"\nProcessing {len(h5_files)} files for {mutation}")
    
    for h5_file in tqdm(h5_files, desc=f"Processing {mutation}"):
        # Extract position from filename
        pos_match = re.search(r'pos_(\d+)', h5_file.name)
        if not pos_match:
            continue
        position = int(pos_match.group(1))
        
        # Process file
        reader = CompressedPredictionReader(h5_file)
        predictions = reader.get_predictions()
        
        if not predictions:
            continue
            
        # Convert to JAX array and compute CVs
        coords_array = jnp.array([p.atom_positions for p in predictions])
        cv_output = compute_cvs_batch_jax(
            coords_array,
            config.cv1_residues,
            config.cv2_residues
        )
        dihedrals, diffs = cv_output
        
        # Store results
        if position not in results:
            results[position] = []
            
        for i, pred in enumerate(predictions):
            results[position].append(CVResult(
                cv1_dihedral=float(dihedrals[i]),
                cv2_diff=float(diffs[i]),
                model_name=pred.name
            ))
    
    # Save results
    with h5py.File(results_file, 'a') as f:
        mutation_group = f.create_group(mutation)
        for pos, pos_results in results.items():
            pos_group = mutation_group.create_group(f'pos_{pos}')
            for i, result in enumerate(pos_results):
                pred_group = pos_group.create_group(f'pred_{i:03d}')
                pred_group.create_dataset('cv1_dihedral', data=result.cv1_dihedral)
                pred_group.create_dataset('cv2_diff', data=result.cv2_diff)
                pred_group.attrs['model_name'] = result.model_name
    
    return results

def compute_cvs_batch_jax(pred_batch: jnp.ndarray,
                         cv1_res: tuple,
                         cv2_res: tuple):
    """Batched CV computation using JAX."""
    batched_func = jax.jit(vmap(compute_cvs_single_jax, in_axes=(0, None, None)))
    return batched_func(pred_batch, cv1_res, cv2_res)

def compute_cvs_single_jax(atom_positions: jnp.ndarray,
                          cv1_res: tuple,
                          cv2_res: tuple) -> tuple:
    """JAX-compatible function to compute CVs."""
    # Unpack residues
    r1, r2, r3, r4 = cv1_res  # 151, 152, 153, 154
    rA, rB, rRef = cv2_res    # 41, 58, 156
    
    # Get coordinates for dihedral
    xyz1 = atom_positions[r1-1, residue_constants.atom_order['CA']]
    xyz2 = atom_positions[r2-1, residue_constants.atom_order['CA']]
    xyz3 = atom_positions[r3-1, residue_constants.atom_order['CA']]
    xyz4 = atom_positions[r4-1, residue_constants.atom_order['CA']]

    # Calculate dihedral using normal vectors method
    b1 = xyz2 - xyz1
    b2 = xyz3 - xyz2
    b3 = xyz4 - xyz3
    
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    
    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2 / jnp.linalg.norm(b2))
    dihedral_angle = jnp.arctan2(sin_phi, cos_phi)

    # Calculate CV2 (distance difference)
    rA_nz = atom_positions[rA-1, residue_constants.atom_order['NZ']]
    rB_cd = atom_positions[rB-1, residue_constants.atom_order['CD']]
    rRef_ne = atom_positions[rRef-1, residue_constants.atom_order['NE']]

    d1 = jnp.linalg.norm(rA_nz - rB_cd)
    d2 = jnp.linalg.norm(rRef_ne - rB_cd)

    return (dihedral_angle, d2 - d1)

def main():
    # Set up paths
    base_path = Path("/work/nw99ixuq-alphamask/my_experiments/her2_39b69/iterative")
    mutations = ["T150A_L157R"]  # Add more mutations as needed
    
    output_dir = Path("tests/cv_results_iterative")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "results_iterative.h5"
    
    # Initialize config
    config = CVConfig(
        calculate_cv1=True,
        calculate_cv2=True,
        cv1_residues=(151, 152, 153, 154),
        cv2_residues=(41, 58, 156)
    )
    
    # Process each mutation
    all_results = {}
    for mutation in mutations:
        compressed_dir = base_path / mutation / "out/compressed"
        if not compressed_dir.exists():
            print(f"Directory not found: {compressed_dir}")
            continue
            
        # Process mutation directory
        mutation_results = process_mutation_dir(
            compressed_dir=compressed_dir,
            config=config,
            results_file=results_file
        )
        
        # Combine all positions for this mutation
        all_results[mutation] = []
        for pos_results in mutation_results.values():
            all_results[mutation].extend(pos_results)
    
    # Create summary plot
    plot_cv_summary_comparison(all_results, output_dir)

if __name__ == "__main__":
    main() 