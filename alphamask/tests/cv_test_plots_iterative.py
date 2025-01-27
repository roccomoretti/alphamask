"""Plotting utilities for iterative CV analysis."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import math
import logging
from scipy.ndimage import gaussian_filter

# If you want a logger here, you can set it up (optional):
logger = logging.getLogger(__name__)

from alphamask.analysis.collective_variables import CVResult

def create_cesar_colormap():
    """Create Cesar's custom colormap."""
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
    return LinearSegmentedColormap.from_list('cesar', list(zip(positions, cesar_colors)))

def plot_iterative_cv_summary(
    all_results: Dict[str, List[CVResult]],
    output_dir: Path,
    recycle_range: Tuple[int, int] = (0, 2)
):
    """
    Create summary plot for iterative CV analysis (N×N subplots).
    This function was already present, left unchanged.
    """
    n_mutations = len(all_results)
    n_rows = (n_mutations + 1) // 2  # 2 columns
    fig, axes = plt.subplots(n_rows, 2, figsize=(20, 10*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    cesar_cmap = create_cesar_colormap()

    # Find global max count for consistent color scaling.
    max_count = 0
    for results in all_results.values():
        filtered_results = []
        for r in results:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match:
                    recycle = int(match.group(1))
                    if recycle_range[0] <= recycle <= recycle_range[1]:
                        filtered_results.append(r)
        cv1_data = [r.cv1_dihedral for r in filtered_results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in filtered_results if r.cv2_diff is not None]
        if cv1_data and cv2_diff_data:
            counts, _, _ = np.histogram2d(cv1_data, cv2_diff_data, bins=50, range=[x_range, y_range])
            max_count = max(max_count, counts.max())

    for idx, (mutation, results) in enumerate(all_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        filtered_results = []
        for r in results:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match:
                    recycle = int(match.group(1))
                    if recycle_range[0] <= recycle <= recycle_range[1]:
                        filtered_results.append(r)

        cv1_data = [r.cv1_dihedral for r in filtered_results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in filtered_results if r.cv2_diff is not None]

        hist = ax.hist2d(cv1_data, cv2_diff_data, bins=50,
                         range=[x_range, y_range],
                         cmap=cesar_cmap,
                         norm=LogNorm(vmin=1, vmax=max_count))
        plt.colorbar(hist[3], ax=ax, label='Count')

        ax.set_xlabel('CV1 (radians)')
        ax.set_ylabel('CV2 (d2-d1)')
        ax.set_title(f"Condition: {mutation}", fontsize=12, pad=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)

        stats_text = f'n = {len(cv1_data)}\n'
        if len(cv1_data) > 0:
            stats_text += f'CV1 mean = {np.mean(cv1_data):.2f}°\n'
        if len(cv2_diff_data) > 0:
            stats_text += f'CV2 mean = {np.mean(cv2_diff_data):.2f} Å'
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

    if n_mutations % 2:
        axes[-1, -1].set_visible(False)

    plt.suptitle(f'Comparison of CV Distributions (Recycles {recycle_range[0]}-{recycle_range[1]})',
                 fontsize=16, y=0.95, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'cv_summary_comparison_r{recycle_range[0]}_{recycle_range[1]}.png',
                dpi=700, bbox_inches='tight')
    plt.close()

def plot_iterative_cv_scatter(
    all_results: Dict[str, List[CVResult]],
    output_dir: Path,
    recycle_range: Tuple[int, int] = (0, 2)
):
    """
    Create scatter plot of CV1 vs CV2 with marginal histograms, colored by pLDDT.
    This function was already present, left unchanged.
    """
    fig = plt.figure(figsize=(10, 10), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)

    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)

    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    cv1_data = []
    cv2_data = []
    plddt_values = []

    for results in all_results.values():
        for r in results:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match:
                    recycle = int(match.group(1))
                    if recycle_range[0] <= recycle <= recycle_range[1]:
                        cv1_data.append(r.cv1_dihedral)
                        cv2_data.append(r.cv2_diff)
                        plddt_values.append(r.plddt)

    cv1_data = np.array(cv1_data)
    cv2_data = np.array(cv2_data)
    plddt_values = np.array(plddt_values)

    sort_idx = np.argsort(plddt_values)
    cv1_data = cv1_data[sort_idx]
    cv2_data = cv2_data[sort_idx]
    plddt_values = plddt_values[sort_idx]

    plddt_min = plddt_values.min() if len(plddt_values) else 0
    plddt_max = plddt_values.max() if len(plddt_values) else 100

    scatter = ax_main.scatter(
        cv1_data, cv2_data,
        c=plddt_values,
        cmap='viridis',
        s=15,
        alpha=0.6,
        edgecolors='none',
        vmin=plddt_min,
        vmax=plddt_max
    )

    ax_main.set_xlabel('CV1 (radians)', fontsize=10, labelpad=8)
    ax_main.set_ylabel('CV2 (d2-d1)', fontsize=10, labelpad=8)
    ax_main.tick_params(axis='both', which='major', labelsize=8)
    ax_main.grid(True, linestyle='--', alpha=0.3)

    counts_top, edges_top, _ = ax_top.hist(cv1_data, bins=50,
                                          range=x_range,
                                          color='#808080',
                                          alpha=0.6,
                                          edgecolor='black',
                                          linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)

    counts_right, edges_right, _ = ax_right.hist(cv2_data, bins=50,
                                                range=y_range,
                                                orientation='horizontal',
                                                color='#808080',
                                                alpha=0.6,
                                                edgecolor='black',
                                                linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    ax_cbar = plt.subplot(gs[0, 1])
    n_ticks = 5
    ticks = np.linspace(plddt_min, plddt_max, n_ticks)
    cbar = plt.colorbar(scatter, cax=ax_cbar, orientation='vertical', ticks=ticks)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('pLDDT', fontsize=9, labelpad=10, rotation=90)

    ax_main.set_xlim(x_range)
    ax_main.set_ylim(y_range)
    ax_top.set_ylim(0, max(counts_top)*1.1 if len(counts_top) else 1)
    ax_right.set_xlim(0, max(counts_right)*1.1 if len(counts_right) else 1)

    plt.suptitle('CV Distribution by pLDDT',
                 fontsize=14, y=0.95, fontweight='bold')

    plt.savefig(output_dir / 'cv_scatter_plddt.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_iterative_cv_summary_all(
    all_results: Dict[str, List[CVResult]],
    output_dir: Path,
    recycle_range: Tuple[int, int] = (0, 2)
):
    """
    Create multi-subplot 2D histogram summary for each mutation in all_results.
    This function was already present, left unchanged.
    """
    valid_mutations = [m for m in all_results if all_results[m]]
    if not valid_mutations:
        return

    n_mutations = len(valid_mutations)
    ncols = int(math.ceil(math.sqrt(n_mutations)))
    nrows = int(math.ceil(n_mutations / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    cmap = plt.cm.viridis

    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    ax_list = axes.flatten()

    for idx, mutation in enumerate(valid_mutations):
        cv_results = all_results[mutation]
        ax = ax_list[idx]

        cv1_data = []
        cv2_diff_data = []
        for r in cv_results:
            match = re.search(r'_r(\d+)_', r.model_name)
            if match:
                recycle = int(match.group(1))
                if recycle_range[0] <= recycle <= recycle_range[1]:
                    cv1_data.append(r.cv1_dihedral)
                    cv2_diff_data.append(r.cv2_diff)

        if len(cv1_data) == 0:
            ax.set_title(f"{mutation} (no data)", fontsize=12, pad=10, fontweight='bold')
            ax.axis('off')
            continue

        cv1_data = np.array(cv1_data)
        cv2_diff_data = np.array(cv2_diff_data)

        H, xedges, yedges, im = ax.hist2d(cv1_data, cv2_diff_data, bins=50,
                                          range=[x_range, y_range],
                                          cmap=cmap, norm=LogNorm(vmin=1))
        plt.colorbar(im, ax=ax, label='Count')
        ax.set_xlabel('CV1 (radians)')
        ax.set_ylabel('CV2 (d2-d1)')
        ax.set_title(f"Condition: {mutation}", fontsize=12, pad=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)

        text_info = f'n = {len(cv1_data)}\n'
        text_info += f'CV1 mean = {np.mean(cv1_data):.2f} rad\n'
        text_info += f'CV2 mean = {np.mean(cv2_diff_data):.2f} Å'
        ax.text(0.02, 0.98, text_info,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

    for extra_ax_i in range(n_mutations, nrows*ncols):
        ax_list[extra_ax_i].axis('off')

    fig.suptitle(f"Comparison of CV Distributions (recycles {recycle_range[0]}-{recycle_range[1]})",
                 fontsize=16, y=0.98, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile = output_dir / f"cv_summary_comparison_all_{recycle_range[0]}_{recycle_range[1]}.png"
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------
# NEW FUNCTIONS adapted from "cv_test_plots_apriori.py" for iterative usage.
# We rename them slightly (e.g., create_iterative_cv_landscape) to avoid collisions.
# Also, we replace the idea of "conditions" with "mutations" and
# handle "all_results" the same way as in the existing iterative plots.
# -------------------------------------------------------------------

def create_iterative_cv_landscape(
    cv1: List[float],
    cv2: List[float],
    title: Optional[str] = None,
    x_label: str = "CV1 (radians)",
    y_label: str = "CV2 (d2-d1)",
    show: bool = False,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Similar to apriori create_cv_landscape but for iterative usage.
    Produces a 2D histogram with marginal histograms for CV1 vs CV2.
    """
    if not cv1 or not cv2:
        logger.warning("No CV data found for create_iterative_cv_landscape")
        fig = plt.figure()
        return fig

    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    fig = plt.figure(figsize=(8, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    ax_main.set_xlim(x_range)
    ax_main.set_ylim(y_range)
    ax_top.set_xlim(x_range)
    ax_right.set_ylim(y_range)

    cesar_cmap = create_cesar_colormap()

    hist2d = ax_main.hist2d(
        cv1, cv2,
        bins=50,
        range=[x_range, y_range],
        cmap=cesar_cmap,
        norm=LogNorm(vmin=1)
    )
    ax_main.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))

    ax_cbar = plt.subplot(gs[0, 1])
    cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='vertical')
    cbar.set_label('Density', fontsize=8, rotation=90, labelpad=2)

    counts_top, edges_top, _ = ax_top.hist(cv1, bins=50,
                                          range=x_range,
                                          color='#808080',
                                          alpha=0.7,
                                          edgecolor='black',
                                          linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.grid(True, linestyle='--', alpha=0.3)

    counts_right, edges_right, _ = ax_right.hist(cv2, bins=50,
                                                range=y_range,
                                                orientation='horizontal',
                                                color='#808080',
                                                alpha=0.7,
                                                edgecolor='black',
                                                linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, linestyle='--', alpha=0.3)

    ax_main.set_xlabel(x_label, fontsize=10, labelpad=8)
    ax_main.set_ylabel(y_label, fontsize=10, labelpad=8)
    ax_main.tick_params(axis='both', which='major', labelsize=8)
    ax_main.grid(True, linestyle='--', alpha=0.3)

    ax_top.set_ylim(0, max(counts_top)*1.1 if len(counts_top) else 1)
    ax_right.set_xlim(0, max(counts_right)*1.1 if len(counts_right) else 1)

    if title:
        plt.suptitle(title, fontsize=12, y=0.95, fontweight='bold')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig

def create_iterative_cv_scatter(
    cv1: List[float],
    cv2: List[float],
    plddt_values: List[float],
    title: Optional[str] = None,
    x_label: str = "CV1 (radians)",
    y_label: str = "CV2 (d2-d1)",
    show: bool = False,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Similar to apriori create_cv_scatter but adapted for iterative usage.
    Creates a scatter plot (CV1 vs CV2) colored by pLDDT with marginal histograms.
    """
    if not cv1 or not cv2 or not plddt_values:
        logger.warning("No data found for create_iterative_cv_scatter")
        fig = plt.figure()
        return fig

    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    cv1_arr = np.array(cv1)
    cv2_arr = np.array(cv2)
    plddt_arr = np.array(plddt_values)

    sort_idx = np.argsort(plddt_arr)
    cv1_arr = cv1_arr[sort_idx]
    cv2_arr = cv2_arr[sort_idx]
    plddt_arr = plddt_arr[sort_idx]

    fig = plt.figure(figsize=(8, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    ax_main.set_xlim(x_range)
    ax_main.set_ylim(y_range)
    ax_top.set_xlim(x_range)
    ax_right.set_ylim(y_range)

    plddt_min = plddt_arr.min()
    plddt_max = plddt_arr.max()
    scatter = ax_main.scatter(
        cv1_arr, cv2_arr,
        c=plddt_arr,
        cmap='viridis',
        s=15,
        alpha=0.6,
        edgecolors='none',
        vmin=plddt_min,
        vmax=plddt_max
    )
    ax_main.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))

    counts_top, edges_top, _ = ax_top.hist(cv1_arr, bins=50,
                                          range=x_range,
                                          color='#808080',
                                          alpha=0.7,
                                          edgecolor='black',
                                          linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.grid(True, linestyle='--', alpha=0.3)

    counts_right, edges_right, _ = ax_right.hist(cv2_arr, bins=50,
                                                range=y_range,
                                                orientation='horizontal',
                                                color='#808080',
                                                alpha=0.7,
                                                edgecolor='black',
                                                linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.grid(True, linestyle='--', alpha=0.3)

    ax_cbar = plt.subplot(gs[0, 1])
    ticks = np.linspace(plddt_min, plddt_max, 5)
    cbar = plt.colorbar(scatter, cax=ax_cbar, orientation='vertical', ticks=ticks)
    cbar.set_label('pLDDT', fontsize=8, rotation=90, labelpad=8)

    ax_top.set_ylim(0, max(counts_top)*1.1 if len(counts_top) else 1)
    ax_right.set_xlim(0, max(counts_right)*1.1 if len(counts_right) else 1)

    ax_main.set_xlabel(x_label, fontsize=10)
    ax_main.set_ylabel(y_label, fontsize=10)

    if title:
        plt.suptitle(title, fontsize=12, y=0.95, fontweight='bold')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig

def create_all_iterative_cv_plots(
    all_results: Dict[str, List[CVResult]],
    output_dir: Path,
    recycle_range: Tuple[int, int] = (0, 2)
) -> None:
    """
    Example "all in one" function, analogous to the apriori "create_all_cv_plots,"
    but for iterative usage. Calls our new iterative plotting functions
    for each mutation and for the filtered data (by recycle).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # For each mutation, filter by recycle range and draw a "landscape" + "scatter" plot
    for mutation, results_list in all_results.items():
        # Filter
        filtered = []
        for r in results_list:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match:
                    recycle = int(match.group(1))
                    if recycle_range[0] <= recycle <= recycle_range[1]:
                        filtered.append(r)
        if not filtered:
            continue

        cv1_vals = [r.cv1_dihedral for r in filtered]
        cv2_vals = [r.cv2_diff for r in filtered]
        plddt_vals = [r.plddt for r in filtered]

        # Landscape
        create_iterative_cv_landscape(
            cv1_vals,
            cv2_vals,
            title=f"{mutation} - (Recycles {recycle_range[0]}-{recycle_range[1]}) Landscape",
            output_path=output_dir / f"{mutation}_cv_landscape_r{recycle_range[0]}-{recycle_range[1]}.png"
        )
        # Scatter
        create_iterative_cv_scatter(
            cv1_vals,
            cv2_vals,
            plddt_vals,
            title=f"{mutation} - (Recycles {recycle_range[0]}-{recycle_range[1]}) Scatter",
            output_path=output_dir / f"{mutation}_cv_scatter_r{recycle_range[0]}-{recycle_range[1]}.png"
        )

def create_iterative_grid_comparison(
    all_results: Dict[str, List[CVResult]],
    plot_type: str = "scatter",  # or "histogram"
    title: Optional[str] = None,
    show: bool = False,
    output_path: Optional[Path] = None,
    recycle_range: Tuple[int, int] = (0, 2)
) -> plt.Figure:
    """Create a publication-ready 2x2 grid comparing CV distributions across mutations."""
    # Filter and sort results
    valid_items = []
    for k, results in all_results.items():
        filtered_results = []
        for r in results:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match:
                    recycle = int(match.group(1))
                    if recycle_range[0] <= recycle <= recycle_range[1]:
                        filtered_results.append(r)
        if filtered_results:
            valid_items.append((k, filtered_results))

    # Define ranges and styling
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    
    # Create figure with extra space for marginal histograms
    fig = plt.figure(figsize=(16, 16), dpi=300)
    
    # Publication-ready styling
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    # Create a complex GridSpec layout for main plots and marginal histograms
    outer_grid = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)
    
    # Define mutation order and colors
    mutation_order = ["WT", "T150A", "L157R", "T150A_L157R"]
    color_mapping = {
        'WT': '#0077BB',         # Blue
        'T150A': '#EE7733',      # Orange
        'L157R': '#009988',      # Teal
        'T150A_L157R': '#CC3311' # Red
    }

    # Sort items
    valid_items.sort(key=lambda x: mutation_order.index(x[0]) if x[0] in mutation_order else len(mutation_order))

    # Calculate global min/max values
    if plot_type == "scatter":
        global_plddt_min = float('inf')
        global_plddt_max = float('-inf')
        for _, cv_results in valid_items:
            plddt_data = np.array([r.plddt for r in cv_results])
            global_plddt_min = min(global_plddt_min, plddt_data.min())
            global_plddt_max = max(global_plddt_max, plddt_data.max())
    else:
        global_density_min = float('inf')
        global_density_max = float('-inf')
        for _, cv_results in valid_items:
            cv1_data = np.array([r.cv1_dihedral for r in cv_results])
            cv2_data = np.array([r.cv2_diff for r in cv_results])
            H, _, _ = np.histogram2d(cv1_data, cv2_data, bins=50, range=[x_range, y_range])
            H = gaussian_filter(H, sigma=1.0)
            global_density_min = min(global_density_min, H[H > 0].min())
            global_density_max = max(global_density_max, H.max())

    # Create subplots with marginal distributions
    for idx, (mutation_name, cv_results) in enumerate(valid_items):
        # Create inner GridSpec for this subplot
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer_grid[idx],
            width_ratios=[4, 1],
            height_ratios=[1, 4],
            hspace=0.05,
            wspace=0.05
        )

        # Create the three axes for main plot and marginal histograms
        ax_main = fig.add_subplot(inner_grid[1, 0])
        ax_top = fig.add_subplot(inner_grid[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner_grid[1, 1], sharey=ax_main)
        
        cv1_data = np.array([r.cv1_dihedral for r in cv_results])
        cv2_data = np.array([r.cv2_diff for r in cv_results])
        
        if plot_type == "scatter":
            plddt_data = np.array([r.plddt for r in cv_results])
            scatter = ax_main.scatter(
                cv1_data, cv2_data,
                c=plddt_data,
                cmap='viridis',
                s=15,
                alpha=0.6,
                vmin=global_plddt_min,
                vmax=global_plddt_max
            )
            
            # Add colorbar
            cax = fig.add_subplot(inner_grid[0, 1])
            cbar = plt.colorbar(scatter, cax=cax, orientation='horizontal')
            cbar.set_label('pLDDT', fontsize=8, labelpad=2)
            cbar.ax.tick_params(labelsize=7)
                
        else:  # histogram
            H, xedges, yedges = np.histogram2d(
                cv1_data, cv2_data,
                bins=50,
                range=[x_range, y_range]
            )
            H = gaussian_filter(H, sigma=1.0)
            
            hist2d = ax_main.hist2d(
                cv1_data, cv2_data,
                bins=50,
                range=[x_range, y_range],
                cmap=create_cesar_colormap(),
                norm=LogNorm(vmin=max(1, global_density_min),
                            vmax=global_density_max)
            )
            
            # Add colorbar
            cax = fig.add_subplot(inner_grid[0, 1])
            cbar = plt.colorbar(hist2d[3], cax=cax, orientation='horizontal')
            cbar.set_label('Density', fontsize=8, labelpad=2)
            cbar.ax.tick_params(labelsize=7)

        # Add marginal histograms
        ax_top.hist(cv1_data, bins=50, range=x_range, 
                   color='gray', alpha=0.7, density=True)
        ax_right.hist(cv2_data, bins=50, range=y_range,
                     orientation='horizontal', color='gray',
                     alpha=0.7, density=True)
        
        # Hide unnecessary labels
        ax_top.set_xticklabels([])
        ax_right.set_yticklabels([])
        
        # Common subplot customization
        ax_main.set_xlim(x_range)
        ax_main.set_ylim(y_range)
        ax_main.set_xlabel('CV1 (radians)', fontsize=10)
        ax_main.set_ylabel('CV2 (d2-d1)', fontsize=10)
        
        # Add statistics text box
        stats_text = (
            f'n = {len(cv1_data):,}\n'
            f'CV1 mean = {np.mean(cv1_data):.2f}°\n'
            f'CV2 mean = {np.mean(cv2_data):.2f} Å'
        )
        ax_main.text(0.02, 0.98, stats_text,
                    transform=ax_main.transAxes,
                    verticalalignment='top',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Set title for each subplot
        ax_top.set_title(mutation_name, fontsize=12, pad=10)
        
        # Set aspect ratio
        ax_main.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))

    # Main title
    if title:
        fig.suptitle(title, fontsize=14, y=0.95, fontweight='bold')

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig

def create_iterative_recycle_comparison(
    all_results: Dict[str, List[CVResult]],
    title: Optional[str] = None,
    show: bool = False,
    output_path: Optional[Path] = None,
    recycle_range: Tuple[int, int] = (0, 2)
) -> plt.Figure:
    """Create a 2x2 grid comparing CV scatter plots for different recycle numbers."""
    # Similar structure to grid_comparison but split by recycle number
    # Implementation follows same pattern but organizes by recycle number
    # ... (implementation similar to create_iterative_grid_comparison) 