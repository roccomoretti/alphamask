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

def create_cesar_colormap():
    """Create Cesar's colormap."""
    cesar_colors = [
        (1.0, 1.0, 1.0),
        (0.416, 0.133, 0.996),
        (0.059, 0.651, 0.937),
        (0.314, 0.957, 0.800),
        (0.686, 0.957, 0.592),
        (1.0, 0.655, 0.349),
        (1.0, 0.133, 0.067),
        (1.0, 0.0, 0.0)
    ]
    positions = [0.0, 1/12, 3/12, 5/12, 7/12, 9/12, 11/12, 1.0]
    return mcolors.LinearSegmentedColormap.from_list('cesar', list(zip(positions, cesar_colors)))

def plot_cv_results(results: List[CVResult], output_dir: Path):
    """Plot CV results."""
    # Extract data
    cv1_data = [r.cv1_dihedral for r in results if r.cv1_dihedral is not None]
    cv2_diff_data = [r.cv2_diff for r in results if r.cv2_diff is not None]
    
    # Create cesar colormap
    cesar_cmap = create_cesar_colormap()
    
    # Define bandwidths for KDE
    bandwidths = [0.15]  # You can adjust this value
    
    for bw in bandwidths:
        # Set up the figure and grid layout
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], hspace=0.05, wspace=0.05)
        ax_main = plt.subplot(gs[1, 0])
        ax_histx = plt.subplot(gs[0, 0], sharex=ax_main)
        ax_histy = plt.subplot(gs[1, 1], sharey=ax_main)
        
        # Define plot ranges
        x_range_min = -0.7
        x_range_max = 3.1
        y_range_min = -15
        y_range_max = 15
        
        # Create grid points for contour
        x_grid = np.linspace(x_range_min, x_range_max, 100)
        y_grid = np.linspace(y_range_min, y_range_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Stack the CV values
        xy = np.vstack([cv1_data, cv2_diff_data])
        
        # Calculate the density
        kde = gaussian_kde(xy, bw_method=bw)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Create contour plots
        levels = np.linspace(Z.min(), Z.max(), 25)
        contourf = ax_main.contourf(X, Y, Z, levels=levels, cmap=cesar_cmap)
        contour = ax_main.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.7)
        
        # Create histograms
        bins_x = np.linspace(x_range_min, x_range_max, 250)
        bins_y = np.linspace(y_range_min, y_range_max, 250)
        
        ax_histx.hist(cv1_data, bins=bins_x, color='black', alpha=1, log=True)
        ax_histy.hist(cv2_diff_data, bins=bins_y, orientation='horizontal', color='black', alpha=1, log=True)
        
        # Customize histogram axes
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['bottom'].set_visible(False)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['left'].set_visible(False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        # Add colorbar
        plt.colorbar(contourf, ax=ax_histy, label='Density')
        
        # Set labels
        ax_main.set_xlabel('CV1 (radians)')
        ax_main.set_ylabel('CV2 (d2-d1)')
        plt.suptitle(f'CV1 vs CV2 Distribution\nBandwidth: {bw}', fontweight='bold')
        
        ax_histx.set_ylabel('Log Count')
        ax_histy.set_xlabel('Log Count')
        
        # Save plot
        plt.savefig(output_dir / f'cv_contour_bw_{bw}.pdf', dpi=700, bbox_inches='tight')
        plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nCV1 Dihedral Angle:")
    print(f"Mean: {np.mean(cv1_data):.2f} rad")
    print(f"Std: {np.std(cv1_data):.2f} rad")
    print(f"Min: {np.min(cv1_data):.2f} rad")
    print(f"Max: {np.max(cv1_data):.2f} rad")
    
    print("\nCV2 Difference (d2-d1):")
    print(f"Mean: {np.mean(cv2_diff_data):.2f} Å")
    print(f"Std: {np.std(cv2_diff_data):.2f} Å")
    print(f"Min: {np.min(cv2_diff_data):.2f} Å")
    print(f"Max: {np.max(cv2_diff_data):.2f} Å")

def plot_cv_scatter(results: List[CVResult], output_dir: Path):
    """Create scatter plot of CV1 vs CV2_diff."""
    # Extract data
    cv1_data = [r.cv1_dihedral for r in results if r.cv1_dihedral is not None]
    cv2_diff_data = [r.cv2_diff for r in results if r.cv2_diff is not None]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(cv1_data, cv2_diff_data, alpha=0.5, s=10)
    
    # Set axis limits
    plt.xlim(-0.7, 3.1)
    plt.ylim(-15, 15)
    
    # Add labels and title
    plt.xlabel('CV1 (radians)')
    plt.ylabel('CV2 (d2-d1)')
    plt.title('CV1 vs CV2 Scatter Plot', fontweight='bold')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    plt.savefig(output_dir / 'cv_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cv_comparison_scatter(all_results: Dict[str, List[CVResult]], output_dir: Path):
    """Create 2x2 subplot comparison using scatter plots."""
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Define conditions and their titles
    conditions = {
        "unmasked_unmutated": "Unmasked Unmutated",
        "unmasked_mutated": "Unmasked T150A-L157R",
        "masked_unmutated": "Masked Unmutated",
        "masked_mutated": "Masked T150A-L157R"
    }
    
    # Plot ranges
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    
    for idx, (condition, results) in enumerate(all_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Extract data
        cv1_data = [r.cv1_dihedral for r in results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in results if r.cv2_diff is not None]
        
        # Create scatter plot
        ax.scatter(cv1_data, cv2_diff_data, alpha=0.5, s=10)
        
        # Set axis limits and labels
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_xlabel('CV1 (radians)')
        ax.set_ylabel('CV2 (d2-d1)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add title
        ax.set_title(conditions[condition], fontsize=12, pad=10, fontweight='bold')
    
    plt.suptitle('Comparison of CV Distributions (Scatter)', 
                 fontsize=16, y=0.95, fontweight='bold')
    
    # Save plot
    plt.savefig(output_dir / 'cv_comparison_scatter.pdf', dpi=700, bbox_inches='tight')
    plt.close()

def plot_cv_comparison_hist2d(all_results: Dict[str, List[CVResult]], output_dir: Path):
    """Create 2x2 subplot comparison using 2D histograms."""
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Define conditions and their titles
    conditions = {
        "unmasked_unmutated": "Unmasked Unmutated",
        "unmasked_mutated": "Unmasked T150A-L157R",
        "masked_unmutated": "Masked Unmutated",
        "masked_mutated": "Masked T150A-L157R"
    }
    
    # Create cesar colormap
    cesar_cmap = create_cesar_colormap()
    
    # Plot ranges
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    bins = 50
    
    # Find global maximum count for consistent coloring
    max_count = 0
    for results in all_results.values():
        cv1_data = [r.cv1_dihedral for r in results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in results if r.cv2_diff is not None]
        counts, _, _ = np.histogram2d(cv1_data, cv2_diff_data, bins=bins,
                                    range=[x_range, y_range])
        max_count = max(max_count, counts.max())
    
    for idx, (condition, results) in enumerate(all_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Extract data
        cv1_data = [r.cv1_dihedral for r in results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in results if r.cv2_diff is not None]
        
        # Create 2D histogram
        hist = ax.hist2d(cv1_data, cv2_diff_data, bins=bins,
                        range=[x_range, y_range],
                        cmap=cesar_cmap,
                        norm=LogNorm(vmin=1, vmax=max_count))
        
        # Set axis limits and labels
        ax.set_xlabel('CV1 (radians)')
        ax.set_ylabel('CV2 (d2-d1)')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add title
        ax.set_title(conditions[condition], fontsize=12, pad=10, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(hist[3], ax=ax, label='Count')
    
    plt.suptitle('Comparison of CV Distributions (2D Histogram)', 
                 fontsize=16, y=0.95, fontweight='bold')
    
    # Save plot
    plt.savefig(output_dir / 'cv_comparison_hist2d.pdf', dpi=700, bbox_inches='tight')
    plt.close()

def plot_cv_recycle_comparison(all_results: Dict[str, List[CVResult]], output_dir: Path):
    """Create recycle comparison plots (0-N and 1-N) for each condition."""
    conditions = {
        "unmasked_unmutated": "Unmasked Unmutated",
        "unmasked_mutated": "Unmasked T150A-L157R",
        "masked_unmutated": "Masked Unmutated",
        "masked_mutated": "Masked T150A-L157R"
    }
    
    # Create figure for 0-N plots
    fig_0n, axes_0n = plt.subplots(2, 2, figsize=(20, 20))
    fig_1n, axes_1n = plt.subplots(2, 2, figsize=(20, 20))
    
    # Plot ranges
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    bins = 50
    
    # Create cesar colormap
    cesar_cmap = create_cesar_colormap()
    
    # Find global maximum count for consistent coloring
    max_count = 0
    global_max_recycle = 0  # Track global maximum recycle
    
    for results in all_results.values():
        # Extract model and recycle info
        model_recycle_data = {}
        for result in results:
            if not result.model_name:
                continue
            match = re.search(r'model_(\d+).*_r(\d+)_', result.model_name)
            if match:
                model = int(match.group(1))
                recycle = int(match.group(2))
                if model not in model_recycle_data:
                    model_recycle_data[model] = {}
                if recycle not in model_recycle_data[model]:
                    model_recycle_data[model][recycle] = []
                model_recycle_data[model][recycle].append(result)
                global_max_recycle = max(global_max_recycle, recycle)  # Update global max recycle
        
        # Calculate cumulative data
        cv1_data_0n = []
        cv2_diff_data_0n = []
        cv1_data_1n = []
        cv2_diff_data_1n = []
        
        # Use global_max_recycle instead of trying to find max from dictionary
        # 0-N cumulative
        for recycle in range(global_max_recycle + 1):
            for model_data in model_recycle_data.values():
                if recycle in model_data:
                    for result in model_data[recycle]:
                        if result.cv1_dihedral is not None and result.cv2_diff is not None:
                            cv1_data_0n.append(result.cv1_dihedral)
                            cv2_diff_data_0n.append(result.cv2_diff)
        
        # 1-N cumulative
        for recycle in range(1, global_max_recycle + 1):
            for model_data in model_recycle_data.values():
                if recycle in model_data:
                    for result in model_data[recycle]:
                        if result.cv1_dihedral is not None and result.cv2_diff is not None:
                            cv1_data_1n.append(result.cv1_dihedral)
                            cv2_diff_data_1n.append(result.cv2_diff)
        
        # Update max count
        if cv1_data_0n and cv2_diff_data_0n:
            counts_0n, _, _ = np.histogram2d(cv1_data_0n, cv2_diff_data_0n, bins=bins,
                                           range=[x_range, y_range])
            max_count = max(max_count, counts_0n.max())
        
        if cv1_data_1n and cv2_diff_data_1n:
            counts_1n, _, _ = np.histogram2d(cv1_data_1n, cv2_diff_data_1n, bins=bins,
                                           range=[x_range, y_range])
            max_count = max(max_count, counts_1n.max())
    
    # Create plots for each condition
    for idx, (condition, results) in enumerate(all_results.items()):
        row = idx // 2
        col = idx % 2
        
        # Extract model and recycle info
        model_recycle_data = {}
        for result in results:
            if not result.model_name:
                continue
            match = re.search(r'model_(\d+).*_r(\d+)_', result.model_name)
            if match:
                model = int(match.group(1))
                recycle = int(match.group(2))
                if model not in model_recycle_data:
                    model_recycle_data[model] = {}
                if recycle not in model_recycle_data[model]:
                    model_recycle_data[model][recycle] = []
                model_recycle_data[model][recycle].append(result)
        
        # Calculate cumulative data
        cv1_data_0n = []
        cv2_diff_data_0n = []
        cv1_data_1n = []
        cv2_diff_data_1n = []
        
        # Use global_max_recycle instead of trying to calculate max_recycle from model_recycle_data ...
        
        # 0-N cumulative
        for recycle in range(global_max_recycle + 1):
            for model_data in model_recycle_data.values():
                if recycle in model_data:
                    for result in model_data[recycle]:
                        if result.cv1_dihedral is not None and result.cv2_diff is not None:
                            cv1_data_0n.append(result.cv1_dihedral)
                            cv2_diff_data_0n.append(result.cv2_diff)
        
        # 1-N cumulative
        for recycle in range(1, global_max_recycle + 1):
            for model_data in model_recycle_data.values():
                if recycle in model_data:
                    for result in model_data[recycle]:
                        if result.cv1_dihedral is not None and result.cv2_diff is not None:
                            cv1_data_1n.append(result.cv1_dihedral)
                            cv2_diff_data_1n.append(result.cv2_diff)
        
        # Create 0-N plot
        hist_0n = axes_0n[row, col].hist2d(cv1_data_0n, cv2_diff_data_0n, bins=bins,
                                          range=[x_range, y_range],
                                          cmap=cesar_cmap,
                                          norm=LogNorm(vmin=1, vmax=max_count))
        
        # Create 1-N plot
        hist_1n = axes_1n[row, col].hist2d(cv1_data_1n, cv2_diff_data_1n, bins=bins,
                                          range=[x_range, y_range],
                                          cmap=cesar_cmap,
                                          norm=LogNorm(vmin=1, vmax=max_count))
        
        # Customize plots
        for ax, title in [(axes_0n[row, col], f"{conditions[condition]}\nRecycles 0-{global_max_recycle}"),
                         (axes_1n[row, col], f"{conditions[condition]}\nRecycles 1-{global_max_recycle}")]:
            ax.set_xlabel('CV1 (radians)')
            ax.set_ylabel('CV2 (d2-d1)')
            ax.set_title(title, fontsize=12, pad=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.colorbar(hist_0n[3], ax=ax, label='Count')
    
    # Add overall titles
    fig_0n.suptitle('Cumulative CV Distributions (Recycles 0-N)', 
                    fontsize=16, y=0.95, fontweight='bold')
    fig_1n.suptitle('Cumulative CV Distributions (Recycles 1-N)', 
                    fontsize=16, y=0.95, fontweight='bold')
    
    # Save plots
    fig_0n.savefig(output_dir / 'cv_comparison_cumulative_0n.pdf', dpi=700, bbox_inches='tight')
    fig_1n.savefig(output_dir / 'cv_comparison_cumulative_1n.pdf', dpi=700, bbox_inches='tight')
    plt.close(fig_0n)
    plt.close(fig_1n)

def create_cv_breakdown(all_results: Dict[str, List[CVResult]], output_dir: Path):
    """Create breakdown plots for CV analysis by model and recycle."""
    # Create output directories
    model_dir = output_dir / "model_breakdown"
    recycle_dir = output_dir / "recycle_breakdown"
    cumulative_dir = output_dir / "cumulative_breakdown"
    
    for directory in [model_dir, recycle_dir, cumulative_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get all models and recycles
    all_models = set()
    all_recycles = set()
    
    # Plot ranges and settings
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    bins = 50
    scatter_alpha = 0.5
    scatter_size = 10
    
    # Create cesar colormap
    cesar_cmap = create_cesar_colormap()
    
    # Find global maximum count for consistent coloring
    max_count = 0
    for results in all_results.values():
        cv1_data = [r.cv1_dihedral for r in results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in results if r.cv2_diff is not None]
        if cv1_data and cv2_diff_data:
            counts, _, _ = np.histogram2d(cv1_data, cv2_diff_data, bins=bins,
                                        range=[x_range, y_range])
            max_count = max(max_count, counts.max())
    
    def create_plots(cv1_data, cv2_diff_data, title, output_path_base):
        """Helper function to create both scatter and histogram plots."""
        # Create scatter plot
        fig_scatter = plt.figure(figsize=(10, 8))
        plt.scatter(cv1_data, cv2_diff_data, alpha=scatter_alpha, s=scatter_size)
        plt.xlabel('CV1 (radians)')
        plt.ylabel('CV2 (d2-d1)')
        plt.title(f'{title} (Scatter)', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.savefig(f"{output_path_base}_scatter.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create 2D histogram
        fig_hist = plt.figure(figsize=(10, 8))
        plt.hist2d(cv1_data, cv2_diff_data, bins=bins,
                  range=[x_range, y_range],
                  cmap=cesar_cmap,
                  norm=LogNorm(vmin=1, vmax=max_count))
        plt.colorbar(label='Count')
        plt.xlabel('CV1 (radians)')
        plt.ylabel('CV2 (d2-d1)')
        plt.title(f'{title} (Density)', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(f"{output_path_base}_density.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 1. Create per-model plots
    print(f"Creating per-model plots for {len(all_models)} models")
    for model in sorted(all_models):
        for condition, results in all_results.items():
            model_data = []
            for result in results:
                if not result.model_name:
                    continue
                match = re.search(r'model_(\d+)', result.model_name)
                if match and int(match.group(1)) == model:
                    model_data.append(result)
            
            if model_data:
                cv1_data = [r.cv1_dihedral for r in model_data if r.cv1_dihedral is not None]
                cv2_diff_data = [r.cv2_diff for r in model_data if r.cv2_diff is not None]
                
                if cv1_data and cv2_diff_data:
                    output_path = model_dir / f'{condition}_model_{model}'
                    create_plots(cv1_data, cv2_diff_data, 
                               f'{condition} - Model {model}',
                               output_path)
    
    # 2. Create per-recycle plots
    print(f"Creating per-recycle plots for {len(all_recycles)} recycles")
    for recycle in sorted(all_recycles):
        for condition, results in all_results.items():
            recycle_data = []
            for result in results:
                if not result.model_name:
                    continue
                match = re.search(r'_r(\d+)_', result.model_name)
                if match and int(match.group(1)) == recycle:
                    recycle_data.append(result)
            
            if recycle_data:
                cv1_data = [r.cv1_dihedral for r in recycle_data if r.cv1_dihedral is not None]
                cv2_diff_data = [r.cv2_diff for r in recycle_data if r.cv2_diff is not None]
                
                if cv1_data and cv2_diff_data:
                    output_path = recycle_dir / f'{condition}_recycle_{recycle}'
                    create_plots(cv1_data, cv2_diff_data,
                               f'{condition} - Recycle {recycle}',
                               output_path)
    
    # 3. Create cumulative plots (0-N)
    print("Creating cumulative plots (0-N)")
    for condition, results in all_results.items():
        cumulative_data = []
        for recycle in sorted(all_recycles):
            for result in results:
                if not result.model_name:
                    continue
                match = re.search(r'_r(\d+)_', result.model_name)
                if match and int(match.group(1)) <= recycle:
                    cumulative_data.append(result)
            
            if cumulative_data:
                cv1_data = [r.cv1_dihedral for r in cumulative_data if r.cv1_dihedral is not None]
                cv2_diff_data = [r.cv2_diff for r in cumulative_data if r.cv2_diff is not None]
                
                if cv1_data and cv2_diff_data:
                    output_path = cumulative_dir / f'{condition}_cumulative_0_{recycle}'
                    create_plots(cv1_data, cv2_diff_data,
                               f'{condition} - Cumulative 0-{recycle}',
                               output_path)
    
    # 4. Create cumulative plots (1-N)
    print("Creating cumulative plots (1-N)")
    for condition, results in all_results.items():
        cumulative_data = []
        for recycle in sorted(all_recycles):
            if recycle == 0:
                continue
            
            for result in results:
                if not result.model_name:
                    continue
                match = re.search(r'_r(\d+)_', result.model_name)
                if match and int(match.group(1)) <= recycle and int(match.group(1)) > 0:
                    cumulative_data.append(result)
            
            if cumulative_data:
                cv1_data = [r.cv1_dihedral for r in cumulative_data if r.cv1_dihedral is not None]
                cv2_diff_data = [r.cv2_diff for r in cumulative_data if r.cv2_diff is not None]
                
                if cv1_data and cv2_diff_data:
                    output_path = cumulative_dir / f'{condition}_cumulative_1_{recycle}'
                    create_plots(cv1_data, cv2_diff_data,
                               f'{condition} - Cumulative 1-{recycle}',
                               output_path)

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
    
    output_dir = Path("tests/cv_results")
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

if __name__ == "__main__":
    main() 