"""
Publication-ready plotting utilities for apriori CV analysis.

Contains two main functions:
1) create_cv_landscape: Creates a 2D histogram (CV1 vs CV2) with top/right histograms.
2) create_cv_scatter:   Creates a scatter plot of CV1 vs CV2 colored by pLDDT (or another metric).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import LogFormatterSciNotation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
from scipy.ndimage import gaussian_filter
from alphamask.analysis.collective_variables import CVResult
import re
logger = logging.getLogger(__name__)

def create_cesar_colormap():
    """
    Create a custom colormap (often used for density or 2D histograms).
    This matches the "cesar" colormap from prior examples.
    """
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

def create_cv_landscape(
    cv1: List[float],
    cv2: List[float],
    title: Optional[str] = None,
    x_label: str = "CV1 (radians)",
    y_label: str = "CV2 (d2-d1)",
    show: bool = False,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a publication-ready 2D histogram plot (CV1 vs CV2).
    Includes marginal histograms on top and right.

    Args:
        cv1: List of CV1 values (e.g., dihedral angles).
        cv2: List of CV2 values (e.g., difference of two labeled distances).
        title: Optional title.
        x_label: X-axis label (default "CV1 (radians)").
        y_label: Y-axis label (default "CV2 (d2-d1)").
        show: If True, show the plot interactively.
        output_path: If provided, save the figure to this file (PDF, PNG, etc.).

    Returns:
        The matplotlib Figure object.
    """
    if not cv1 or not cv2:
        logger.warning("No CV data found for create_cv_landscape")
        fig = plt.figure()
        return fig

    # Convert to numpy array
    cv1_arr = np.array(cv1)
    cv2_arr = np.array(cv2)

    # Use fixed ranges for consistent plotting
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    # Create figure (publication-friendly size)
    fig = plt.figure(figsize=(8, 8), dpi=300)

    # Create gridspec for main axis + top/right hist
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    # Subplots
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0])
    ax_right= fig.add_subplot(gs[1, 1])
    
    # Set the ranges explicitly for all axes
    ax_main.set_xlim(x_range)
    ax_main.set_ylim(y_range)
    ax_top.set_xlim(x_range)
    ax_right.set_ylim(y_range)

    # Use Cesar's colormap or any other
    cesar_cmap = create_cesar_colormap()

    # 2D histogram
    hist2d = ax_main.hist2d(cv1_arr, cv2_arr, bins=50,
                            range=[x_range, y_range],
                            cmap=cesar_cmap,
                            norm=LogNorm(vmin=1))

    # Force square aspect ratio for main plot
    ax_main.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))

    # Add colorbar on the right side
    ax_cbar = plt.subplot(gs[0, 1])
    cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='vertical',
                        format=LogFormatterSciNotation(base=10, labelOnlyBase=True))
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Density', fontsize=8, rotation=90, labelpad=2)
    # Optionally set ticks or custom labels on colorbar
    # cbar.set_ticks([1, 10, 100])

    # Plot top histogram
    counts_top, edges_top, _ = ax_top.hist(cv1_arr, bins=50,
                                          range=(x_range[0], x_range[1]),
                                          color='#808080',
                                          alpha=0.7,
                                          edgecolor='black', linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)

    # Plot right histogram
    counts_right, edges_right, _ = ax_right.hist(cv2_arr, bins=50,
                                                range=(y_range[0], y_range[1]),
                                                orientation='horizontal',
                                                color='#808080',
                                                alpha=0.7,
                                                edgecolor='black', linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    # Customize ax_main
    ax_main.set_xlabel(x_label, fontsize=10, labelpad=8)
    ax_main.set_ylabel(y_label, fontsize=10, labelpad=8)
    ax_main.tick_params(axis='both', which='major', labelsize=8)
    ax_main.minorticks_on()
    ax_main.grid(True, linestyle='--', alpha=0.3)

    # Adjust top & right hist plot limits
    ax_top.set_ylim(0, max(counts_top)*1.1)
    ax_right.set_xlim(0, max(counts_right)*1.1)

    # Set a main title
    if title:
        plt.suptitle(title, fontsize=12, y=0.95, fontweight='bold')

    # Adjust layout
    plt.subplots_adjust(
        left=0.15,
        right=0.9,
        top=0.9,
        bottom=0.15,
        wspace=0.05,
        hspace=0.05
    )

    # Save/Show
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig

def create_cv_scatter(
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
    Create a publication-ready scatter plot (CV1 vs CV2) colored by pLDDT.
    Marginal histograms on top and right, in a similar style to create_cv_landscape.

    Args:
        cv1: List of CV1 values.
        cv2: List of CV2 values.
        plddt_values: List of pLDDT or confidence metrics (0-100 range recommended).
        title: Optional main title.
        x_label: X-axis label (default "CV1 (radians)").
        y_label: Y-axis label (default "CV2 (d2-d1)").
        show: If True, shows the plot interactively.
        output_path: If provided, saves the figure to this file (PDF, PNG, etc.).

    Returns:
        The matplotlib Figure object.
    """
    if not cv1 or not cv2 or not plddt_values:
        logger.warning("No data found for create_cv_scatter")
        fig = plt.figure()
        return fig

    cv1_arr = np.array(cv1)
    cv2_arr = np.array(cv2)
    plddt_arr = np.array(plddt_values)

    # Use fixed ranges for consistent plotting
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    # Sort points by pLDDT so high pLDDT points appear on top
    sort_idx = np.argsort(plddt_arr)
    cv1_arr = cv1_arr[sort_idx]
    cv2_arr = cv2_arr[sort_idx]
    plddt_arr = plddt_arr[sort_idx]

    # Create figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    
    # Set the ranges explicitly
    ax_main.set_xlim(x_range)
    ax_main.set_ylim(y_range)
    ax_top.set_xlim(x_range)
    ax_right.set_ylim(y_range)

    # Use Cesar's colormap or any other
    cesar_cmap = create_cesar_colormap()

    # Scatter plot
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

    # Force square aspect ratio for main plot
    ax_main.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))

    # Histograms
    counts_top, edges_top, _ = ax_top.hist(cv1_arr, bins=50,
                                          range=(x_range[0], x_range[1]),
                                          color='#808080', alpha=0.7,
                                          edgecolor='black', linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)

    counts_right, edges_right, _ = ax_right.hist(cv2_arr, bins=50,
                                                range=(y_range[0], y_range[1]),
                                                orientation='horizontal',
                                                color='#808080', alpha=0.7,
                                                edgecolor='black', linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    # Colorbar
    ax_cbar = plt.subplot(gs[0, 1])
    divider = make_axes_locatable(ax_cbar)
    n_ticks = 5
    ticks = np.linspace(plddt_min, plddt_max, n_ticks)
    cbar = plt.colorbar(scatter, cax=ax_cbar, orientation='vertical', ticks=ticks)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('pLDDT', fontsize=8, rotation=90, labelpad=8)

    # Adjust histogram axis limits
    ax_top.set_ylim(0, max(counts_top)*1.1)
    ax_right.set_xlim(0, max(counts_right)*1.1)

    # Title
    if title:
        plt.suptitle(title, fontsize=12, y=0.95, fontweight='bold')

    plt.subplots_adjust(
        left=0.15,
        right=0.9,
        top=0.9,
        bottom=0.15,
        wspace=0.05,
        hspace=0.05
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig 

def create_cv_summary_all(
    all_results: Dict[str, List["CVResult"]],
    title: Optional[str] = "Multi-Condition CV Summary",
    bins: int = 50,
    show: bool = False,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a multi-subplot, publication-ready 2D histogram summary of CV1 vs CV2
    for each condition in all_results.

    Args:
        all_results: Dict mapping condition_name -> list of CVResult objects.
        title: Optional figure title.
        bins: Number of bins for the 2D histogram.
        show: If True, display the figure interactively.
        output_path: If provided, save the figure to this file (PDF, PNG, etc.).

    Returns:
        The matplotlib Figure object.
    """
    # Filter out conditions with empty data
    valid_items = [(k, v) for k, v in all_results.items() if v]
    if not valid_items:
        logger.warning("No valid CV data found for create_cv_summary_all")
        return plt.figure()

    # Cesar colormap for density
    cesar_cmap = create_cesar_colormap()

    # Determine how many subplots (N×N)
    n_conditions = len(valid_items)
    ncols = int(np.ceil(np.sqrt(n_conditions)))
    nrows = int(np.ceil(n_conditions / ncols))

    # Create figure
    fig = plt.figure(figsize=(6 * ncols, 6 * nrows), dpi=300)
    axes = []
    
    # Use fixed ranges for consistent plotting
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)

    # We'll store hist2d objects for a single colorbar
    hist2d_list = []

    # Second pass: create subplots
    for idx, (condition_name, cv_results) in enumerate(valid_items):
        cv1_arr = np.array([r.cv1_dihedral for r in cv_results])
        cv2_arr = np.array([r.cv2_diff    for r in cv_results])

        ax = fig.add_subplot(nrows, ncols, idx + 1)
        axes.append(ax)

        # Set ranges explicitly
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # 2D histogram
        h = ax.hist2d(
            cv1_arr, cv2_arr,
            bins=bins,
            range=[x_range, y_range],
            cmap=cesar_cmap,
            norm=LogNorm(vmin=1)
        )
        hist2d_list.append(h)

        # Force square aspect ratio
        ax.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))

        # Label and style
        ax.set_title(condition_name, fontsize=10, pad=5, fontweight='bold')
        ax.set_xlabel("CV1 (radians)", fontsize=8)
        ax.set_ylabel("CV2 (d2-d1)", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Shared colorbar
    # We can base it on the last hist2d or any of them
    cbar = fig.colorbar(hist2d_list[-1][3], ax=axes, orientation='vertical')
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Density', fontsize=8, rotation=90, labelpad=5)

    # Global title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()

    return fig 

def create_cv_condition_comparison(
    all_results: Dict[str, List[CVResult]],
    title: Optional[str] = None,
    show: bool = False,
    output_path: Optional[Path] = None,
    recycle_range: Tuple[int, int] = (0, 2)
) -> plt.Figure:
    """Create a 2x2 grid comparing CV distributions across conditions."""
    # Filter results by recycle range
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
    
    # Fixed ranges
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    
    # Create figure
    fig = plt.figure(figsize=(12, 12), dpi=300)
    
    # Use a more common font
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # Define condition order and colors
    condition_order = [
        'unmasked_unmutated',
        'unmasked_mutated',
        'masked_unmutated',
        'masked_mutated'
    ]
    
    color_mapping = {
        'unmasked_unmutated': '#0077BB',  # Blue
        'unmasked_mutated': '#EE7733',    # Orange
        'masked_unmutated': '#009988',     # Teal
        'masked_mutated': '#CC3311'        # Red
    }
    
    # Sort items to ensure consistent order
    valid_items.sort(key=lambda x: condition_order.index(x[0]) if x[0] in condition_order else len(condition_order))
    
    # Find global maximum density for consistent coloring
    global_density_max = 0
    for _, cv_results in valid_items:
        cv1_data = [r.cv1_dihedral for r in cv_results]
        cv2_data = [r.cv2_diff for r in cv_results]
        H, _, _ = np.histogram2d(cv1_data, cv2_data, bins=50,
                                range=[x_range, y_range])
        H = gaussian_filter(H, sigma=1.0)
        global_density_max = max(global_density_max, H.max())
    
    # Create each subplot
    for idx, (condition_name, cv_results) in enumerate(valid_items):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Extract data
        cv1_data = np.array([r.cv1_dihedral for r in cv_results])
        cv2_data = np.array([r.cv2_diff for r in cv_results])
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(cv1_data, cv2_data,
                                          bins=50, range=[x_range, y_range])
        H = gaussian_filter(H, sigma=1.0)
        
        hist2d = ax.hist2d(
            cv1_data, cv2_data,
            bins=50,
            range=[x_range, y_range],
            cmap=create_cesar_colormap(),
            norm=LogNorm(vmin=1, vmax=global_density_max)
        )
        
        # Add colorbar
        cbar = plt.colorbar(hist2d[3], ax=ax)
        cbar.set_label('Density', fontsize=10)
        
        # Customize axes
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('CV1 (radians)', fontsize=10)
        ax.set_ylabel('CV2 (d2-d1)', fontsize=10)
        ax.set_title(condition_name.replace('_', ' ').title(), fontsize=12, pad=10)
        ax.grid(True, linestyle=':', alpha=0.2)
        
        # Set aspect ratio
        ax.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.95, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(
        right=0.85,    # Make room for colorbars
        wspace=0.4,    # Increase spacing between plots
        hspace=0.3     # Vertical spacing
    )
    
    # Save with recycle range in filename
    if output_path:
        path = Path(output_path)
        new_path = path.parent / f"{path.stem}_r{recycle_range[0]}-{recycle_range[1]}{path.suffix}"
        fig.savefig(new_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()
    
    return fig

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate KL divergence between two distributions.
    Adds small epsilon to avoid log(0) and ensures normalization.
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    
    # Normalize and add epsilon
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate KL divergence
    return np.sum(p * np.log(p / q))

def compute_distribution_differences(
    all_results: Dict[str, List["CVResult"]],
    reference: str = "unmasked_unmutated",
    bins: int = 50,
    output_path: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute KL divergence between reference condition and all others.
    Calculates divergence for both CV1 and CV2 distributions.
    
    Args:
        all_results: Dict mapping condition_name -> list of CVResult objects
        reference: Name of reference condition (default: "unmasked_unmutated")
        bins: Number of bins for histogram calculation
        output_path: If provided, save results to CSV
        
    Returns:
        Dict containing KL divergences for CV1 and CV2
    """
    if reference not in all_results:
        raise ValueError(f"Reference condition '{reference}' not found in results")
    
    # Fixed ranges for consistent binning
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    
    # Get reference distributions
    ref_cv1 = [r.cv1_dihedral for r in all_results[reference]]
    ref_cv2 = [r.cv2_diff for r in all_results[reference]]
    
    # Calculate reference histograms
    ref_hist_cv1, _ = np.histogram(ref_cv1, bins=bins, range=x_range, density=True)
    ref_hist_cv2, _ = np.histogram(ref_cv2, bins=bins, range=y_range, density=True)
    
    # Calculate KL divergence for each condition
    results = {}
    for condition, cv_results in all_results.items():
        if condition == reference:
            continue
            
        cv1_vals = [r.cv1_dihedral for r in cv_results]
        cv2_vals = [r.cv2_diff for r in cv_results]
        
        # Calculate histograms
        hist_cv1, _ = np.histogram(cv1_vals, bins=bins, range=x_range, density=True)
        hist_cv2, _ = np.histogram(cv2_vals, bins=bins, range=y_range, density=True)
        
        # Calculate KL divergence
        kl_cv1 = calculate_kl_divergence(ref_hist_cv1, hist_cv1)
        kl_cv2 = calculate_kl_divergence(ref_hist_cv2, hist_cv2)
        
        results[condition] = {
            'KL_CV1': kl_cv1,
            'KL_CV2': kl_cv2
        }
    
    # Save to CSV if path provided
    if output_path:
        import pandas as pd
        df = pd.DataFrame.from_dict(results, orient='index')
        df.index.name = 'Condition'
        df.to_csv(output_path)
        logger.info(f"Saved KL divergence results to {output_path}")
    
    return results 

def create_cv_grid_comparison(
    all_results: Dict[str, List["CVResult"]],
    plot_type: str = "scatter",  # or "histogram"
    title: Optional[str] = None,
    show: bool = False,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a 2×2 grid of either scatter plots or 2D histograms for all conditions.
    
    Args:
        all_results: Dict mapping condition_name -> list of CVResult objects
        plot_type: Either "scatter" or "histogram"
        title: Optional title for the figure
        show: If True, display the plot
        output_path: If provided, save to this path
        
    Returns:
        The matplotlib Figure object
    """
    # Filter out empty conditions
    valid_items = [(k, v) for k, v in all_results.items() if v]
    if not valid_items:
        logger.warning("No valid CV data found for grid comparison")
        return plt.figure()
    
    # Define colors and conditions in a fixed order
    condition_order = [
        'unmasked_unmutated',
        'unmasked_mutated',
        'masked_unmutated',
        'masked_mutated'
    ]
    
    color_mapping = {
        'unmasked_unmutated': '#0077BB',  # Blue
        'unmasked_mutated': '#EE7733',    # Orange
        'masked_unmutated': '#009988',     # Teal
        'masked_mutated': '#CC3311'        # Red
    }
    
    # Sort items to ensure consistent order
    valid_items.sort(key=lambda x: condition_order.index(x[0]) if x[0] in condition_order else len(condition_order))
    
    # Fixed ranges - define these before first use
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    
    # First pass: determine global min/max values
    if plot_type == "scatter":
        global_plddt_min = float('inf')
        global_plddt_max = float('-inf')
        for _, cv_results in valid_items:
            plddt_data = np.array([r.plddt for r in cv_results])
            global_plddt_min = min(global_plddt_min, plddt_data.min())
            global_plddt_max = max(global_plddt_max, plddt_data.max())
    else:  # histogram
        global_density_min = float('inf')
        global_density_max = float('-inf')
        for _, cv_results in valid_items:
            cv1_data = np.array([r.cv1_dihedral for r in cv_results])
            cv2_data = np.array([r.cv2_diff for r in cv_results])
            H, _, _ = np.histogram2d(
                cv1_data, cv2_data,
                bins=50,
                range=[x_range, y_range]
            )
            H = gaussian_filter(H, sigma=1.0)
            global_density_min = min(global_density_min, H[H > 0].min())  # Ignore zeros
            global_density_max = max(global_density_max, H.max())

    # Create figure with 2×2 grid
    fig = plt.figure(figsize=(12, 12), dpi=300)
    
    # Use a more common font that should be available
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # Create subplots
    for idx, (condition_name, cv_results) in enumerate(valid_items):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        cv1_data = np.array([r.cv1_dihedral for r in cv_results])
        cv2_data = np.array([r.cv2_diff for r in cv_results])
        
        if plot_type == "scatter":
            # Scatter plot colored by pLDDT
            plddt_data = np.array([r.plddt for r in cv_results])
            scatter = ax.scatter(
                cv1_data, cv2_data,
                c=plddt_data,
                cmap='viridis',
                s=15,
                alpha=0.6,
                vmin=global_plddt_min,
                vmax=global_plddt_max
            )
            
            # Add colorbar to the right of each plot
            if True:  # Now we add colorbar to all plots
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('pLDDT', fontsize=10)
                
        else:  # histogram
            # 2D histogram with contours
            H, xedges, yedges = np.histogram2d(
                cv1_data, cv2_data,
                bins=50,
                range=[x_range, y_range]
            )
            H = gaussian_filter(H, sigma=1.0)
            
            # Plot contours
            x_centers = (xedges[:-1] + xedges[1:]) / 2
            y_centers = (yedges[:-1] + yedges[1:]) / 2
            
            hist2d = ax.hist2d(
                cv1_data, cv2_data,
                bins=50,
                range=[x_range, y_range],
                cmap=create_cesar_colormap(),
                norm=LogNorm(vmin=max(1, global_density_min),
                            vmax=global_density_max)
            )
            
            # Add colorbar to the right of each plot
            if True:  # Now we add colorbar to all plots
                cbar = plt.colorbar(hist2d[3], ax=ax)
                cbar.set_label('Density', fontsize=10)
        
        # Common subplot customization
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('CV1 (radians)', fontsize=10)
        ax.set_ylabel('CV2 (d2-d1)', fontsize=10)
        ax.set_title(condition_name.replace('_', ' ').title(), fontsize=12, pad=10)
        ax.grid(True, linestyle=':', alpha=0.2)
        ax.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))
    
    # Main title
    if title:
        fig.suptitle(title, fontsize=14, y=0.95, fontweight='bold')
    
    # Adjust layout to make room for colorbars
    plt.subplots_adjust(
        right=0.85,    # Make room for colorbars
        wspace=0.4,    # Increase spacing between plots
        hspace=0.3     # Vertical spacing
    )
    
    # Save/show
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()
    
    return fig 

def create_cv_recycle_comparison(
    all_results: Dict[str, List[CVResult]],
    title: Optional[str] = None,
    show: bool = False,
    output_path: Optional[Path] = None,
    recycle_range: Tuple[int, int] = (0, 2)
) -> plt.Figure:
    """Create a 2x2 grid comparing CV scatter plots for recycles 0-2."""
    # Fixed ranges
    x_range = (-0.7, 3.1)
    y_range = (-15, 15)
    
    # Create figure
    fig = plt.figure(figsize=(12, 12), dpi=300)
    
    # Use a more common font
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # Define condition order
    condition_order = [
        'unmasked_unmutated',
        'unmasked_mutated',
        'masked_unmutated',
        'masked_mutated'
    ]
    
    # Sort and filter valid items
    valid_items = [(k, v) for k, v in all_results.items() if v]
    valid_items.sort(key=lambda x: condition_order.index(x[0]) if x[0] in condition_order else len(condition_order))
    
    # Find global pLDDT range for consistent coloring
    global_plddt_min = float('inf')
    global_plddt_max = float('-inf')
    for _, results in valid_items:
        for r in results:
            if r.plddt is not None:
                global_plddt_min = min(global_plddt_min, r.plddt)
                global_plddt_max = max(global_plddt_max, r.plddt)
    
    # Create each subplot
    for idx, (condition_name, cv_results) in enumerate(valid_items):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Filter and plot each recycle
        for recycle in range(recycle_range[0], recycle_range[1] + 1):
            filtered_results = []
            for r in cv_results:
                if r.model_name:
                    match = re.search(r'_r(\d+)_', r.model_name)
                    if match and int(match.group(1)) == recycle:
                        filtered_results.append(r)
            
            if filtered_results:
                cv1_data = [r.cv1_dihedral for r in filtered_results]
                cv2_data = [r.cv2_diff for r in filtered_results]
                plddt_data = [r.plddt for r in filtered_results]
                
                scatter = ax.scatter(
                    cv1_data, cv2_data,
                    c=plddt_data,
                    cmap='viridis',
                    s=15,
                    alpha=0.6,
                    vmin=global_plddt_min,
                    vmax=global_plddt_max,
                    label=f'Recycle {recycle}'
                )
        
        # Customize axes
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('CV1 (radians)', fontsize=10)
        ax.set_ylabel('CV2 (d2-d1)', fontsize=10)
        ax.set_title(condition_name.replace('_', ' ').title(), fontsize=12, pad=10)
        ax.grid(True, linestyle=':', alpha=0.2)
        ax.legend(fontsize=8)
        
        # Set aspect ratio
        ax.set_aspect((x_range[1] - x_range[0])/(y_range[1] - y_range[0]))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('pLDDT', fontsize=10)
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.95, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(
        right=0.85,    # Make room for colorbars
        wspace=0.4,    # Increase spacing between plots
        hspace=0.3     # Vertical spacing
    )
    
    # Save/show
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close(fig)
    elif show:
        plt.show()
    
    return fig

def create_all_cv_plots(
    all_results: Dict[str, List[CVResult]],
    output_dir: Path,
    recycle_range: Tuple[int, int] = (0, 2)
) -> None:
    """
    Create and save all CV plot types.
    
    Args:
        all_results: Dictionary mapping condition names to lists of CVResults
        output_dir: Directory to save plots
        recycle_range: Tuple of (min_recycle, max_recycle) to include
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save condition comparison plot
    create_cv_condition_comparison(
        all_results,
        title=f"Comparison of All Conditions (Recycles {recycle_range[0]}-{recycle_range[1]})",
        output_path=output_dir / "cv_condition_comparison.png",
        recycle_range=recycle_range
    )
    
    # Create and save summary plot
    create_cv_summary_all(
        all_results,
        title=f"Multi-Condition CV Summary (Recycles {recycle_range[0]}-{recycle_range[1]})",
        output_path=output_dir / "cv_summary_all.png",
        show=False
    )
    
    # Create recycle comparison scatter plot
    create_cv_recycle_comparison(
        all_results,
        title=f"CV Scatter by Recycle (r{recycle_range[0]}-{recycle_range[1]})",
        output_path=output_dir / f"cv_recycle_comparison_r{recycle_range[0]}-{recycle_range[1]}.png",
        recycle_range=recycle_range
    )
    
    # Create 2x2 grid for each recycle number
    for recycle in range(recycle_range[0], recycle_range[1] + 1):
        filtered_results = {}
        for condition, results in all_results.items():
            filtered = []
            for r in results:
                if r.model_name:
                    match = re.search(r'_r(\d+)_', r.model_name)
                    if match and int(match.group(1)) == recycle:
                        filtered.append(r)
            if filtered:
                filtered_results[condition] = filtered
        
        if filtered_results:
            create_cv_grid_comparison(
                filtered_results,
                plot_type="scatter",
                title=f"CV Scatter Comparison (Recycle {recycle})",
                output_path=output_dir / f"cv_scatter_r{recycle}.png"
            )
            
            create_cv_grid_comparison(
                filtered_results,
                plot_type="histogram",
                title=f"CV Density Comparison (Recycle {recycle})",
                output_path=output_dir / f"cv_density_r{recycle}.png"
            )
    
    # Create individual condition plots
    for condition, results in all_results.items():
        # Filter results by recycle range
        filtered_results = []
        for r in results:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match:
                    recycle = int(match.group(1))
                    if recycle_range[0] <= recycle <= recycle_range[1]:
                        filtered_results.append(r)
        
        if filtered_results:
            # Create landscape plot
            cv1_data = [r.cv1_dihedral for r in filtered_results]
            cv2_data = [r.cv2_diff for r in filtered_results]
            
            create_cv_landscape(
                cv1_data,
                cv2_data,
                title=f"{condition} (Recycles {recycle_range[0]}-{recycle_range[1]})",
                output_path=output_dir / f"cv_landscape_{condition}_r{recycle_range[0]}-{recycle_range[1]}.png"
            )