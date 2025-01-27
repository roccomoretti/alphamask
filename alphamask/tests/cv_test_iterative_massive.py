"""
Test script for collective variables calculation - iterative case (massive parallel approach).
Can run in debug mode (few files) or full mode (all files).
Processes WT, T150A, L157R, T150A_L157R in a loop and produces N×N summary plots.
"""

import sys
from pathlib import Path
from typing import Dict, List
import re
import h5py
import numpy as np
from tqdm import tqdm
import logging

# Add parent directory to path to import alphamask
sys.path.append(str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import vmap
from colabdesign.af.alphafold.common import residue_constants

from alphamask.analysis.collective_variables import CVConfig, CVResult
from alphamask.utils.compression import CompressedPredictionReader
from cv_test_plots_iterative import plot_iterative_cv_summary, plot_iterative_cv_scatter, plot_iterative_cv_summary_all, create_all_iterative_cv_plots, create_iterative_grid_comparison

# Set up logging with a format that includes timestamps
logging.basicConfig(
    level=logging.INFO,  # Change default to INFO instead of DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = False  # Set to False to process all files
CHUNK_SIZE = 50000  # Chunk size for GPU processing

# The mutations (including WT) that we want to analyze
ALL_MUTATIONS = ["WT", "T150A", "L157R", "T150A_L157R"]

def parse_model_name(name: str) -> dict:
    """Parse model name to extract seed, model, and recycle numbers."""
    info = {}
    parts = name.split('_')
    
    # Extract seed, model, recycle
    info['seed'] = next((p.replace('seed_', '') for p in parts if p.startswith('seed_')), "1")
    info['model'] = next((p.replace('model_', '') for p in parts if p.startswith('model_')), "1")
    info['recycle'] = next((p.replace('r', '') for p in parts if p.startswith('r')), "0")
    
    return info

def compute_cvs_batch_jax(pred_batch: jnp.ndarray,
                          cv1_res: tuple,
                          cv2_res: tuple):
    """Batched CV computation using JAX."""
    batched_func = jax.jit(vmap(compute_cvs_single_jax, in_axes=(0, None, None)))
    return batched_func(pred_batch, cv1_res, cv2_res)

def compute_cvs_single_jax(atom_positions: jnp.ndarray,
                           cv1_res: tuple,
                           cv2_res: tuple) -> tuple:
    """Compute CVs for one structure using normal-vector dihedral approach."""
    # Unpack residues
    r1, r2, r3, r4 = cv1_res
    rA, rB, rRef = cv2_res
    
    xyz1 = atom_positions[r1-1, residue_constants.atom_order['CA']]
    xyz2 = atom_positions[r2-1, residue_constants.atom_order['CA']]
    xyz3 = atom_positions[r3-1, residue_constants.atom_order['CA']]
    xyz4 = atom_positions[r4-1, residue_constants.atom_order['CA']]
    
    # Dihedral
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
    
    # Distances
    rA_nz = atom_positions[rA-1, residue_constants.atom_order['NZ']]
    rB_cd = atom_positions[rB-1, residue_constants.atom_order['CD']]
    rRef_ne = atom_positions[rRef-1, residue_constants.atom_order['NE']]
    
    d1 = jnp.linalg.norm(rA_nz - rB_cd)
    d2 = jnp.linalg.norm(rRef_ne - rB_cd)
    return (dihedral_angle, d2 - d1)

def main():
    """
    Massive parallel approach with chunking:
    1) For each mutation in ALL_MUTATIONS, collect data from H5 files 
       (all if DEBUG=False, subset if DEBUG=True).
    2) Split predictions into chunks to avoid huge GPU arrays.
    3) For each chunk, submit a batch to the GPU for CV calculations.
    4) Combine or store results in a single HDF5 for caching.
    5) Plot a summary for recycles 0-2 in an N×N layout (all mutations).
    """
    base_path = Path("/work/nw99ixuq-alphamask/my_experiments/her2_39b69/iterative/")
    
    output_dir = Path("tests/cv_results_iterative_massive")
    if DEBUG:
        output_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We'll store ALL mutations in one HDF5
    results_file = output_dir / f"results_iterative_massive{'_debug' if DEBUG else ''}.h5"
    
    # Our CV config
    config = CVConfig(
        calculate_cv1=True,
        calculate_cv2=True,
        cv1_residues=(151, 152, 153, 154),
        cv2_residues=(41, 58, 156)
    )
    
    # Dictionary to hold final CVResults for each mutation
    all_results: Dict[str, List[CVResult]] = {}
    
    # Open HDF5 in append-or-create mode: create if doesn't exist
    # We'll add or replace a group for each mutation
    mode = 'a' if results_file.exists() else 'w'
    with h5py.File(results_file, mode) as f:
        
        for mutation in ALL_MUTATIONS:
            compressed_dir = base_path / mutation / "out/compressed"
            
            # If we already have an HDF5 group for this mutation, skip reading predictions
            if mutation in f:
                logger.info(f"Skipping {mutation} - already found group in HDF5 results.")
                # But still load them into all_results if we need them for plotting
                saved_cv_results = []
                for pred_name in f[mutation].keys():
                    g = f[mutation][pred_name]
                    cv1 = float(g['cv1_dihedral'][()])
                    cv2 = float(g['cv2_diff'][()])
                    plddt = float(g['plddt'][()])
                    # Retrieve model_name from attrs
                    model_name = g.attrs.get('model_name', 'unknown_model')
                    saved_cv_results.append(CVResult(
                        cv1_dihedral=cv1, cv2_diff=cv2,
                        model_name=model_name,
                        plddt=plddt
                    ))
                all_results[mutation] = saved_cv_results
                continue
            
            # Otherwise, read/compute as before
            h5_files = list(compressed_dir.glob("*.h5"))
            if DEBUG:
                h5_files = h5_files[:10]
                logger.debug(f"Debug mode: Processing first {len(h5_files)} h5 files for {mutation}")
            else:
                logger.info(f"Processing all {len(h5_files)} h5 files for {mutation}")
            
            if not h5_files:
                logger.warning(f"No .h5 files found for mutation {mutation} in: {compressed_dir}")
                all_results[mutation] = []
                continue
            
            all_predictions = []
            all_model_names = []
            all_file_names = []
            
            desc = f"Reading predictions {mutation} (debug mode)" if DEBUG else f"Reading predictions {mutation}"
            for h5_file in tqdm(h5_files, desc=desc):
                reader = CompressedPredictionReader(h5_file)
                predictions = reader.get_predictions()
                for pred in predictions:
                    all_predictions.append(pred)
                    all_model_names.append(pred.name)
                    all_file_names.append(h5_file.name)
            
            if not all_predictions:
                logger.warning(f"No predictions discovered for {mutation}. Skipping.")
                all_results[mutation] = []
                continue
            
            total = len(all_predictions)
            logger.info(f"Total predictions for {mutation}: {total}. Using chunk size {CHUNK_SIZE}")
            
            cv_results: List[CVResult] = []
            
            start_idx = 0
            while start_idx < total:
                end_idx = min(start_idx + CHUNK_SIZE, total)
                batch_predictions = all_predictions[start_idx:end_idx]
                batch_names = all_model_names[start_idx:end_idx]
                
                batch_positions = [pred.atom_positions for pred in batch_predictions]
                coords_array = jnp.array(batch_positions)
                
                logger.debug(f"[{mutation}] Processing chunk {start_idx}:{end_idx}")
                
                dihedrals, diffs = compute_cvs_batch_jax(coords_array, config.cv1_residues, config.cv2_residues)
                
                dihedrals = np.array(dihedrals)
                diffs = np.array(diffs)
                
                for i, name in enumerate(batch_names):
                    pred_idx = start_idx + i
                    cv_results.append(CVResult(
                        cv1_dihedral=float(dihedrals[i]),
                        cv2_diff=float(diffs[i]),
                        model_name=str(name),
                        plddt=float(np.mean(all_predictions[pred_idx].plddt))
                    ))
                
                del coords_array, dihedrals, diffs
                start_idx = end_idx
            
            logger.info(f"Computed CVs for {len(cv_results)} predictions ({mutation})")
            
            # Save results in the HDF5
            if mutation in f:
                del f[mutation]  # Remove old group if it exists
            mutation_group = f.create_group(mutation)
            
            for i, result in enumerate(cv_results):
                pred_group = mutation_group.create_group(f'pred_{i:05d}')
                pred_group.create_dataset('cv1_dihedral', data=result.cv1_dihedral)
                pred_group.create_dataset('cv2_diff', data=result.cv2_diff)
                pred_group.create_dataset('plddt', data=result.plddt)
                
                model_info = parse_model_name(result.model_name)
                pred_group.attrs['model_name'] = result.model_name
                pred_group.attrs['source_file'] = all_file_names[i]
                pred_group.attrs['seed'] = model_info['seed']
                pred_group.attrs['model'] = model_info['model']
                pred_group.attrs['recycle'] = model_info['recycle']
            
            # Store them in our dictionary for plotting
            all_results[mutation] = cv_results
    
    logger.info("All mutations processed. Now generating summary plots.")
    
    # Create combined summary plots for all mutations
    plot_iterative_cv_summary(all_results, output_dir, recycle_range=(0, 2))
    plot_iterative_cv_scatter(all_results, output_dir, recycle_range=(0, 2))
    plot_iterative_cv_summary_all(all_results, output_dir, recycle_range=(0, 2))

    # Create publication-ready 2x2 grid plots
    logger.info("Creating publication-ready 2x2 grid plots...")
    
    # For recycles 0-2
    create_iterative_grid_comparison(
        all_results,
        plot_type="histogram",
        title="CV Distribution Comparison (Recycles 0-2)",
        output_path=output_dir / "cv_grid_histogram_r0-2.png",
        recycle_range=(0, 2)
    )
    
    create_iterative_grid_comparison(
        all_results,
        plot_type="scatter",
        title="CV Scatter Comparison (Recycles 0-2)",
        output_path=output_dir / "cv_grid_scatter_r0-2.png",
        recycle_range=(0, 2)
    )

    # For all recycles
    create_iterative_grid_comparison(
        all_results,
        plot_type="histogram",
        title="CV Distribution Comparison (All Recycles)",
        output_path=output_dir / "cv_grid_histogram_all.png",
        recycle_range=(0, 20)  # Adjust max recycle as needed
    )
    
    create_iterative_grid_comparison(
        all_results,
        plot_type="scatter",
        title="CV Scatter Comparison (All Recycles)",
        output_path=output_dir / "cv_grid_scatter_all.png",
        recycle_range=(0, 20)  # Adjust max recycle as needed
    )

    # Create extra iterative-style plots
    logger.info("Creating extra iterative-style plots...")
    create_all_iterative_cv_plots(all_results, output_dir / "extra_iterative_plots", recycle_range=(0, 2))

    logger.info(f"Created all summary plots in: {output_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    main() 