"""
Test script for collective variables calculation - apriori case (massive parallel approach).
Processes four conditions in a single directory set:
    1) unmasked_unmutated
    2) unmasked_mutated
    3) masked_unmutated
    4) masked_mutated
Produces an N×N summary plot at the end across these conditions.
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
from cv_test_plots_apriori import (
    create_cv_landscape,
    create_cv_scatter,
    create_cv_condition_comparison,
    create_cv_grid_comparison,
    compute_distribution_differences,
    create_all_cv_plots
)

# Set up logging with a format that includes timestamps
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG if needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEBUG = False          # If True, process a subset of each H5 file
CHUNK_SIZE = 50000     # Chunk size for GPU processing

# Define mutation sets
MUTATION_SETS = [
    "Abdullah_et_al_2023_T150A",
    "Abdullah_et_al_2023_L157R",
    "Abdullah_et_al_2023_T150A_L157R"
]

BASE_PATH = Path("/work/nw99ixuq-alphamask/my_experiments/her2_39b69/apriori")

def parse_model_name(name: str) -> dict:
    """
    Parse model name to extract seed, model, and recycle numbers.
    E.g. "seed_13_model_2_r0" => {"seed":"13", "model":"2", "recycle":"0"}
    """
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
    """
    Batched CV computation using JAX, 
    mapping compute_cvs_single_jax over the first dimension of pred_batch.
    """
    batched_func = jax.jit(vmap(compute_cvs_single_jax, in_axes=(0, None, None)))
    return batched_func(pred_batch, cv1_res, cv2_res)

def compute_cvs_single_jax(atom_positions: jnp.ndarray,
                           cv1_res: tuple,
                           cv2_res: tuple) -> tuple:
    """
    Compute CVs for one structure using a normal-vector dihedral approach (CV1)
    plus a pair of distances for CV2 (state 2 minus state 1).
    """
    # Unpack residues
    r1, r2, r3, r4 = cv1_res
    rA, rB, rRef = cv2_res

    xyz1 = atom_positions[r1-1, residue_constants.atom_order['CA']]
    xyz2 = atom_positions[r2-1, residue_constants.atom_order['CA']]
    xyz3 = atom_positions[r3-1, residue_constants.atom_order['CA']]
    xyz4 = atom_positions[r4-1, residue_constants.atom_order['CA']]

    # Dihedral angle
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

def get_input_files(mutation_set: str) -> Dict[str, Path]:
    """Get input files for a specific mutation set."""
    base = BASE_PATH / mutation_set
    
    # Extract mutation info from set name
    mutation = '_'.join(mutation_set.split('_')[3:])  # e.g., "T150A" or "L157R" or "T150A_L157R"
    
    # Extract residue numbers for masking
    residues = []
    for mut in mutation.split('_'):
        if mut.startswith('T') or mut.startswith('L'):  # T150A or L157R
            residues.append(str(int(mut[1:-1])))
    mask_residues = '_'.join(residues)  # e.g., "150" or "157" or "150_157"
    
    # Use the actual mutation string for file paths
    if mutation_set == "Abdullah_et_al_2023_T150A":
        mut_string = "T150A"
    elif mutation_set == "Abdullah_et_al_2023_L157R":
        mut_string = "L157R"
    elif mutation_set == "Abdullah_et_al_2023_T150A_L157R":
        mut_string = "T150A_L157R"
    
    return {
        "unmasked_unmutated": base / "unmasked_unmutated/out/compressed" / f"{mutation_set}_39b69_all_atoms.h5",
        "unmasked_mutated":   base / "unmasked_mutated/out/compressed" / f"{mutation_set}_39b69_mut_{mut_string}_all_atoms.h5",
        "masked_unmutated":   base / "masked_unmutated/out/compressed" / f"{mutation_set}_39b69_mask_{mask_residues}_id_X_all_atoms.h5",
        "masked_mutated":     base / "masked_mutated/out/compressed" / f"{mutation_set}_39b69_mask_{mask_residues}_mut_{mut_string}_id_X_all_atoms.h5"
    }

def main():
    """Process all mutation sets."""
    for mutation_set in MUTATION_SETS:
        logger.info(f"\nProcessing mutation set: {mutation_set}")
        
        # Get input files for this mutation set
        input_files = get_input_files(mutation_set)
        
        # Create output directory for this mutation set
        output_dir = Path("tests/cv_results_apriori_massive") / mutation_set
        if DEBUG:
            output_dir = output_dir / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"results_apriori_massive{'_debug' if DEBUG else ''}.h5"
        
        # Initialize CV config
        config = CVConfig(
            calculate_cv1=True,
            calculate_cv2=True,
            cv1_residues=(151, 152, 153, 154),
            cv2_residues=(41, 58, 156)
        )
        
        # Store results for this mutation set
        all_results: Dict[str, List[CVResult]] = {}
        
        mode = 'a' if results_file.exists() else 'w'
        with h5py.File(results_file, mode) as f:
            # Process each condition
            for condition_name, h5_path in input_files.items():
                # If group in HDF5 already exists, skip computation
                if condition_name in f:
                    logger.info(f"Skipping {condition_name} - already found group in HDF5 results.")
                    # Load existing CVResults so we can still plot them
                    saved_cv_results = []
                    for pred_group_name in f[condition_name].keys():
                        g = f[condition_name][pred_group_name]
                        cv1 = float(g['cv1_dihedral'][()])
                        cv2 = float(g['cv2_diff'][()])
                        plddt = float(g['plddt'][()])
                        model_name = g.attrs.get('model_name', 'unknown_model')
                        saved_cv_results.append(
                            CVResult(cv1_dihedral=cv1, cv2_diff=cv2, model_name=model_name, plddt=plddt)
                        )
                    all_results[condition_name] = saved_cv_results
                    continue

                # Otherwise read the compressed file
                if not h5_path.exists():
                    logger.warning(f"File not found for condition {condition_name}: {h5_path}")
                    all_results[condition_name] = []
                    continue

                logger.info(f"Processing {condition_name}: {h5_path}")

                reader = CompressedPredictionReader(h5_path)
                predictions = reader.get_predictions()

                # Debug? Subset
                if DEBUG:
                    predictions = predictions[:10]
                    logger.debug(f"Debug mode: only {len(predictions)} predictions read for {condition_name}")

                if not predictions:
                    logger.warning(f"No predictions read for condition {condition_name}. Skipping.")
                    all_results[condition_name] = []
                    continue

                cv_results: List[CVResult] = []
                start_idx = 0
                total = len(predictions)
                logger.info(f"Total predictions for {condition_name}: {total}")

                # Chunk the predictions
                while start_idx < total:
                    end_idx = min(start_idx + CHUNK_SIZE, total)
                    batch_preds = predictions[start_idx:end_idx]

                    # Convert positions to jnp
                    batch_positions = [pred.atom_positions for pred in batch_preds]
                    coords_array = jnp.array(batch_positions)

                    dihedrals, diffs = compute_cvs_batch_jax(coords_array, config.cv1_residues, config.cv2_residues)
                    dihedrals = np.array(dihedrals)
                    diffs = np.array(diffs)

                    for i, pred in enumerate(batch_preds):
                        # Build CVResult
                        cv_results.append(CVResult(
                            cv1_dihedral=float(dihedrals[i]),
                            cv2_diff=float(diffs[i]),
                            model_name=pred.name,
                            plddt=float(np.mean(pred.plddt))
                        ))
                    del coords_array, dihedrals, diffs
                    start_idx = end_idx

                logger.info(f"Computed CVs for {len(cv_results)} predictions ({condition_name})")

                # Save results to the file
                cond_group = f.create_group(condition_name)
                for i, result in enumerate(cv_results):
                    pred_group = cond_group.create_group(f'pred_{i:05d}')
                    pred_group.create_dataset('cv1_dihedral', data=result.cv1_dihedral)
                    pred_group.create_dataset('cv2_diff',     data=result.cv2_diff)
                    pred_group.create_dataset('plddt',        data=result.plddt)

                    model_info = parse_model_name(result.model_name)
                    pred_group.attrs['model_name']  = result.model_name
                    pred_group.attrs['seed']        = model_info['seed']
                    pred_group.attrs['model']       = model_info['model']
                    pred_group.attrs['recycle']     = model_info['recycle']

                all_results[condition_name] = cv_results

        # Create plots for this mutation set
        logger.info(f"Creating plots for {mutation_set}")
        
        # Individual condition plots
        for condition_name, results_list in all_results.items():
            if not results_list:
                logger.warning(f"No CV results found for {condition_name}. Skipping plots.")
                continue
            
            # Extract data and create plots
            cv1_vals = [r.cv1_dihedral for r in results_list]
            cv2_vals = [r.cv2_diff for r in results_list]
            plddt_vals = [r.plddt for r in results_list]
            
            # Create plots with mutation set in title
            landscape_title = f"{mutation_set} - {condition_name} - CV Landscape"
            create_cv_landscape(
                cv1_vals, cv2_vals,
                title=landscape_title,
                output_path=output_dir / f"{condition_name}_cv_landscape.png"
            )
            
            scatter_title = f"{mutation_set} - {condition_name} - CV Scatter by pLDDT"
            create_cv_scatter(
                cv1_vals, cv2_vals, plddt_vals,
                title=scatter_title,
                output_path=output_dir / f"{condition_name}_cv_scatter.png"
            )
        
        # Create comparison plots
        create_cv_condition_comparison(
            all_results,
            title=f"{mutation_set} - Comparison of All Conditions",
            output_path=output_dir / "cv_condition_comparison.png"
        )
        
        # Compute KL divergences
        kl_results = compute_distribution_differences(
            all_results,
            reference="unmasked_unmutated",
            output_path=output_dir / "kl_divergences.csv"
        )
        
        # Create grid comparisons
        create_cv_grid_comparison(
            all_results,
            plot_type="scatter",
            title=f"{mutation_set} - Comparison (Scatter)",
            output_path=output_dir / "cv_grid_comparison_scatter.png"
        )
        
        create_cv_grid_comparison(
            all_results,
            plot_type="histogram",
            title=f"{mutation_set} - Comparison (2D Histogram)",
            output_path=output_dir / "cv_grid_comparison_histogram.png"
        )
        
        # Create all CV plots with recycle filtering
        create_all_cv_plots(
            all_results=all_results,
            output_dir=output_dir / "cv_plots",
            recycle_range=(0, 2)
        )
        
        logger.info(f"Completed processing {mutation_set}")

if __name__ == "__main__":
    main() 