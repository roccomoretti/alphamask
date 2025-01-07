"""
Recycle analysis visualization functionality.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, ScalarFormatter
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import plotly.graph_objects as go
import logging

from .landscape_plots import create_rmsd_landscape
from ..storage import Storage

logger = logging.getLogger(__name__)

def create_collage(
    plots_data: List[Tuple[np.ndarray, Optional[np.ndarray], str]], 
    title: str,
    ncols: int = 4,
    max_rmsd: Optional[float] = None
) -> plt.Figure:
    """
    Create a collage of RMSD landscape plots.
    
    Args:
        plots_data: List of (rmsd1, rmsd2, subtitle) tuples
        title: Overall title for the collage
        ncols: Number of columns in the grid
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Generated figure
    """
    nrows = (len(plots_data) + ncols - 1) // ncols
    fig = plt.figure(figsize=(5*ncols, 5*nrows), dpi=300)
    
    # Create grid for all plots with proper spacing
    gs = gridspec.GridSpec(nrows, ncols + 1, figure=fig, width_ratios=[1] * ncols + [0.05])
    
    # Create Cesar's colormap
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
    cesar_cmap = LinearSegmentedColormap.from_list('cesar', list(zip(positions, cesar_colors)))
    
    # Find global max RMSD for axis alignment
    if max_rmsd is None:
        all_rmsd1 = np.concatenate([rmsd1 for rmsd1, _, _ in plots_data])
        all_rmsd2 = []
        for _, rmsd2, _ in plots_data:
            if rmsd2 is not None:
                all_rmsd2.extend(rmsd2)
        global_max_rmsd = max(max(all_rmsd1), max(all_rmsd2) if all_rmsd2 else max(all_rmsd1)) * 1.1
    else:
        global_max_rmsd = max_rmsd
    
    hist2d_list = []  # Store hist2d objects for colorbar
    for idx, (rmsd1, rmsd2, subtitle) in enumerate(plots_data):
        if idx >= nrows * ncols:
            break
            
        # Create subplot grid with better spacing
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[idx // ncols, idx % ncols],
                                                  width_ratios=[4, 1], height_ratios=[1, 4],
                                                  hspace=0.0, wspace=0.0)
        
        # Create subplots
        ax_main = plt.subplot(inner_gs[1, 0])
        ax_top = plt.subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = plt.subplot(inner_gs[1, 1], sharey=ax_main)
        
        # Create landscape plot
        y_data = rmsd2 if rmsd2 is not None else rmsd1
        
        # Create main 2D histogram
        hist2d = ax_main.hist2d(rmsd1, y_data, bins=40,
                              range=[[0, global_max_rmsd], [0, global_max_rmsd]],
                              cmap=cesar_cmap,
                              norm=LogNorm(vmin=1))  # Set minimum value to 1 to avoid log(0)
        hist2d_list.append(hist2d[3])
        
        # Add diagonal line with improved visibility
        ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                    alpha=0.8, linewidth=1.5, dashes=(5, 5))
        
        # Customize main plot
        ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
        ax_main.set_ylabel('RMSD vs Reference 2 (Å)' if rmsd2 is not None else 'RMSD vs Reference 1 (Å)',
                          fontsize=10, labelpad=8)
        ax_main.tick_params(axis='both', which='major', labelsize=8)
        ax_main.tick_params(axis='both', which='minor', labelsize=6)
        ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
        ax_main.minorticks_on()
        
        # Create top histogram with matching color
        counts_top, edges_top, _ = ax_top.hist(rmsd1, bins=40, range=(0, global_max_rmsd),
                                             color=cesar_colors[2], alpha=0.7,
                                             edgecolor='black', linewidth=0.5)
        ax_top.tick_params(axis='x', labelbottom=False)
        ax_top.tick_params(axis='y', labelsize=8)
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        ax_top.set_title(subtitle, fontsize=10, pad=5)
        
        # Create right histogram with matching color
        counts_right, edges_right, _ = ax_right.hist(y_data, bins=40, range=(0, global_max_rmsd),
                                                   orientation='horizontal', color=cesar_colors[2],
                                                   alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_right.tick_params(axis='y', labelleft=False)
        ax_right.tick_params(axis='x', labelsize=8)
        ax_right.grid(True, linestyle='--', alpha=0.3)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        
        # Ensure square aspect ratio for main plot
        ax_main.set_aspect('equal')
        
        # Set the same limits for top and right histograms
        ax_top.set_ylim(0, max(counts_top) * 1.1)  # Add 10% padding
        ax_right.set_xlim(0, max(counts_right) * 1.1)  # Add 10% padding
        
        # Remove main plot spines between histograms
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        
        # Set x and y limits for main plot
        ax_main.set_xlim(0, global_max_rmsd)
        ax_main.set_ylim(0, global_max_rmsd)
    
    # Create shared colorbar
    if hist2d_list:
        # Create a new axis for the colorbar that spans all rows
        ax_cbar = plt.subplot(gs[:, -1])
        cbar = plt.colorbar(hist2d_list[0], cax=ax_cbar, orientation='vertical')
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_title('Density', fontsize=8, pad=3)
        
        # Format colorbar ticks to handle low frequencies better
        formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
        cbar.formatter = formatter
        cbar.update_ticks()
        
        # Adjust colorbar tick positions for better spacing
        tick_locator = LogLocator(base=10, numticks=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
    
    # Add overall title with adjusted position
    fig.suptitle(title, fontsize=12, y=0.95, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.3, hspace=0.3)
    
    return fig

def create_recycle_landscapes(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready RMSD landscape plots for recycle analysis.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Dictionary of generated figures
    """
    logger.info(f"Creating recycle landscape plots for position {position}")
    figures = {}
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get recycle data
    logger.debug(f"Retrieving recycle data for position {position}")
    recycle_data = storage.get_all_recycle_data(position)
    if not recycle_data:
        logger.warning(f"No data found for position {position}")
        return figures
    
    # Find max recycle number across all models
    max_recycles = max(max(r for r in model_data.keys()) for model_data in recycle_data.values())
    logger.debug(f"Maximum recycle number: {max_recycles}")
    
    # Create individual recycle plots
    logger.info(f"Creating individual recycle plots (0-{max_recycles})")
    for recycle in range(max_recycles + 1):
        rmsd1_values = []
        rmsd2_values = []
        
        for model in sorted(recycle_data.keys()):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    rmsd1_values.extend(np.array(data.rmsd_ref1).flatten())
                    if data.rmsd_ref2 is not None:
                        rmsd2_values.extend(np.array(data.rmsd_ref2).flatten())
        
        if rmsd1_values:
            logger.debug(f"Creating landscape plot for recycle {recycle}")
            fig = create_rmsd_landscape(
                rmsd_ref1=np.array(rmsd1_values),
                rmsd_ref2=np.array(rmsd2_values) if rmsd2_values else None,
                title=f"Position {position}\nRecycle {recycle}",
                show=False,
                max_rmsd=max_rmsd
            )
            figures[f'recycle_{recycle}'] = fig
            
            if output_dir:
                fig.savefig(output_dir / f"pos_{position}_recycle_{recycle}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    # Create cumulative plots (0-N)
    logger.info("Creating cumulative plots (0-N)")
    for recycle in range(max_recycles + 1):
        logger.debug(f"Creating cumulative plot for recycles 0-{recycle}")
        cumulative_rmsd1 = []
        cumulative_rmsd2 = []
        
        # Collect data for all models up to current recycle
        for r in range(recycle + 1):
            logger.debug(f"Processing recycle {r}")
            for model in sorted(recycle_data.keys()):
                if r in recycle_data[model]:
                    logger.debug(f"Recycle {r} found in model {model}")
                    data = recycle_data[model][r]
                    if data.rmsd_ref1 is not None:
                        cumulative_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            cumulative_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if cumulative_rmsd1:
            logger.debug(f"Creating cumulative landscape plot for recycles 0-{recycle}")
            fig = create_rmsd_landscape(
                rmsd_ref1=np.array(cumulative_rmsd1),
                rmsd_ref2=np.array(cumulative_rmsd2) if cumulative_rmsd2 else None,
                title=f"Position {position}\nCumulative Recycles 0-{recycle}",
                show=False,
                max_rmsd=max_rmsd
            )
            figures[f'cumulative_0_{recycle}'] = fig
            
            if output_dir:
                fig.savefig(output_dir / f"cumulative_plot_r0_{recycle}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    # Create cumulative plots (1-N)
    logger.info("Creating cumulative plots (1-N)")
    for recycle in range(1, max_recycles + 1):
        cumulative_rmsd1 = []
        cumulative_rmsd2 = []

        # Collect data for all models from recycle 1 to current recycle
        for r in range(1, recycle + 1):
            logger.debug(f"Processing recycle {r}")
            for model in sorted(recycle_data.keys()):
                if r in recycle_data[model]:
                    logger.debug(f"Recycle {r} found in model {model}")
                    data = recycle_data[model][r]
                    logger.debug(f"Data type: {type(data)}")
                    if data.rmsd_ref1 is not None:
                        cumulative_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        logger.debug(f"Cumulative RMSD1 values: {cumulative_rmsd1}")
                        if data.rmsd_ref2 is not None:
                            cumulative_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
                            logger.debug(f"Cumulative RMSD2 values: {cumulative_rmsd2}")
        if cumulative_rmsd1:
            logger.debug(f"Creating cumulative landscape plot for recycles 1-{recycle}")
            fig = create_rmsd_landscape(
                rmsd_ref1=np.array(cumulative_rmsd1),
                rmsd_ref2=np.array(cumulative_rmsd2) if cumulative_rmsd2 else None,
                title=f"Position {position}\nCumulative Recycles 1-{recycle}",
                show=False,
                max_rmsd=max_rmsd
            )
            figures[f'cumulative_1_{recycle}'] = fig
            
            if output_dir:
                fig.savefig(output_dir / f"cumulative_plot_r1_{recycle}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    logger.info(f"Created {len(figures)} recycle landscape plots for position {position}")
    return figures

def create_recycle_progression(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready recycle progression plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    # Get recycle data
    recycle_data = storage.get_all_recycle_data(position)
    if not recycle_data:
        logger.warning(f"No data found for position {position}")
        return figures
    
    # Create figure with publication-friendly size and DPI
    fig = plt.figure(figsize=(10, 15), dpi=300)
    
    # Determine if we have rmsd2 data
    has_rmsd2 = any(hasattr(data, 'rmsd_ref2') and data.rmsd_ref2 is not None
                    for model_data in recycle_data.values()
                    for data in model_data.values())
    
    # Adjust grid based on presence of rmsd2
    if has_rmsd2:
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 2], hspace=0.3)
    else:
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 2], hspace=0.3)
    
    # Plot RMSD1 progression
    ax1 = plt.subplot(gs[0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(recycle_data)))
    
    rmsd1_max = max_rmsd if max_rmsd is not None else 0
    for idx, (model, model_data) in enumerate(sorted(recycle_data.items())):
        recycles = sorted(model_data.keys())
        rmsd_means = []
        rmsd_stds = []
        
        for r in recycles:
            data = model_data[r]
            if data.rmsd_ref1 is not None:
                rmsd_array = np.array(data.rmsd_ref1)
                rmsd_means.append(np.mean(rmsd_array))
                rmsd_stds.append(np.std(rmsd_array))
        
        if rmsd_means:
            if max_rmsd is None:
                rmsd1_max = max(rmsd1_max, max(m + s for m, s in zip(rmsd_means, rmsd_stds)))
            ax1.errorbar(recycles, rmsd_means, yerr=rmsd_stds,
                        fmt='o-', label=f'Model {model}',
                        color=colors[idx], capsize=3,
                        markersize=5, linewidth=1.5, capthick=1)
    
    ax1.set_xlabel('Recycle', fontsize=10, labelpad=8)
    ax1.set_ylabel('RMSD to Reference 1 (Å)', fontsize=10, labelpad=8)
    ax1.set_title(f'Position {position} - RMSD Progression',
                  fontsize=12, pad=10, fontweight='bold')
    ax1.legend(fontsize=8, frameon=True, fancybox=True,
              facecolor='white', edgecolor='gray')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.set_ylim(0, rmsd1_max * 1.1)  # Add 10% padding
    
    # Plot RMSD2 progression if available
    if has_rmsd2:
        ax2 = plt.subplot(gs[1])
        rmsd2_max = max_rmsd if max_rmsd is not None else 0
        for idx, (model, model_data) in enumerate(sorted(recycle_data.items())):
            recycles = sorted(model_data.keys())
            rmsd_means = []
            rmsd_stds = []
            valid_recycles = []
            
            for r in recycles:
                data = model_data[r]
                if hasattr(data, 'rmsd_ref2') and data.rmsd_ref2 is not None:
                    rmsd_array = np.array(data.rmsd_ref2)
                    rmsd_means.append(np.mean(rmsd_array))
                    rmsd_stds.append(np.std(rmsd_array))
                    valid_recycles.append(r)
            
            if valid_recycles:
                if max_rmsd is None:
                    rmsd2_max = max(rmsd2_max, max(m + s for m, s in zip(rmsd_means, rmsd_stds)))
                ax2.errorbar(valid_recycles, rmsd_means, yerr=rmsd_stds,
                           fmt='o-', label=f'Model {model}',
                           color=colors[idx], capsize=3,
                           markersize=5, linewidth=1.5, capthick=1)
        
        ax2.set_xlabel('Recycle', fontsize=10, labelpad=8)
        ax2.set_ylabel('RMSD to Reference 2 (Å)', fontsize=10, labelpad=8)
        ax2.set_title('RMSD to Reference 2 Progression',
                     fontsize=12, pad=10, fontweight='bold')
        ax2.legend(fontsize=8, frameon=True, fancybox=True,
                  facecolor='white', edgecolor='gray')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.set_ylim(0, rmsd2_max * 1.1)  # Add 10% padding
        
        # pLDDT progression plot goes in third position
        ax3 = plt.subplot(gs[2])
    else:
        # If no rmsd2, pLDDT plot is second
        ax3 = plt.subplot(gs[1])
    
    # Plot pLDDT progression
    plddt_max = 0
    for idx, (model, model_data) in enumerate(sorted(recycle_data.items())):
        recycles = sorted(model_data.keys())
        plddt_means = []
        plddt_stds = []
        
        for r in recycles:
            data = model_data[r]
            if data.plddt is not None:
                plddt_array = np.array(data.plddt)
                if len(plddt_array.shape) > 2:
                    plddt_array = plddt_array.mean(axis=0)
                if len(plddt_array.shape) == 2:
                    plddt_array = plddt_array[0]
                # Scale pLDDT values to 0-100 range if they're in 0-1 range
                if np.max(plddt_array) <= 1.0:
                    plddt_array = plddt_array * 100
                plddt_means.append(np.mean(plddt_array))
                plddt_stds.append(np.std(plddt_array))
        
        if plddt_means:
            plddt_max = max(plddt_max, max(m + s for m, s in zip(plddt_means, plddt_stds)))
            ax3.errorbar(recycles, plddt_means, yerr=plddt_stds,
                        fmt='o-', label=f'Model {model}',
                        color=colors[idx], capsize=3,
                        markersize=5, linewidth=1.5, capthick=1)
    
    ax3.set_xlabel('Recycle', fontsize=10, labelpad=8)
    ax3.set_ylabel('Mean pLDDT', fontsize=10, labelpad=8)
    ax3.set_title('pLDDT Progression', fontsize=12, pad=10, fontweight='bold')
    ax3.legend(fontsize=8, frameon=True, fancybox=True,
              facecolor='white', edgecolor='gray')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.set_ylim(0, 100)  # Set fixed range for pLDDT
    
    # Per-residue pLDDT heatmap
    ax4 = plt.subplot(gs[-1])
    plddt_matrix = []
    row_labels = []
    
    for model in sorted(recycle_data.keys()):
        for recycle in sorted(recycle_data[model].keys()):
            data = recycle_data[model][recycle]
            if data.plddt is not None:
                plddt_array = np.array(data.plddt)
                if len(plddt_array.shape) > 2:
                    plddt_array = plddt_array.mean(axis=0)
                if len(plddt_array.shape) == 2:
                    plddt_array = plddt_array[0]
                # Scale pLDDT values to 0-100 range if they're in 0-1 range
                if np.max(plddt_array) <= 1.0:
                    plddt_array = plddt_array * 100
                plddt_matrix.append(plddt_array)
                row_labels.append(f'M{model} R{recycle}')  # Shortened labels
    
    if plddt_matrix:
        plddt_matrix = np.array(plddt_matrix)
        # Create AlphaFold-style colormap
        af_colors = [
            (0.0, '#FF7D45'),  # Very low
            (0.5, '#FFF300'),  # Low
            (0.7, '#00A1D3'),  # Confident
            (0.9, '#0053D6'),  # Very high
            (1.0, '#0053D6')   # Very high (extended)
        ]
        af_cmap = LinearSegmentedColormap.from_list('alphafold', af_colors)
        
        im = ax4.imshow(plddt_matrix, aspect='auto', cmap=af_cmap,
                       vmin=0, vmax=100)
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('pLDDT', fontsize=10, labelpad=8)
        cbar.ax.tick_params(labelsize=8)
        
        ax4.set_yticks(range(len(row_labels)))
        ax4.set_yticklabels(row_labels, fontsize=8)
        ax4.set_xlabel('Residue', fontsize=10, labelpad=8)
        ax4.set_title('Per-residue pLDDT', fontsize=12, pad=10, fontweight='bold')
        ax4.tick_params(axis='both', which='major', labelsize=8)
    
    plt.suptitle(f'Position {position} - Recycle Analysis',
                fontsize=14, y=0.95, fontweight='bold')
    plt.tight_layout()
    figures['progression'] = fig
    
    if output_dir:
        output_path = output_dir / f"pos_{position}_progression.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return figures

def create_recycle_summary(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> plt.Figure:
    """
    Create summary plot comparing all recycles.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Generated figure
    """
    # Get recycle data
    recycle_data = storage.get_all_recycle_data(position)
    if not recycle_data:
        logger.warning(f"No data found for position {position}")
        return None
    
    # Create figure with publication-friendly size and DPI
    fig = plt.figure(figsize=(15, 10), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Plot RMSD1 progression
    ax1 = plt.subplot(gs[0, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(recycle_data)))
    
    rmsd1_max = max_rmsd if max_rmsd is not None else 0
    for idx, (model, model_data) in enumerate(sorted(recycle_data.items())):
        recycles = sorted(model_data.keys())
        rmsd_means = []
        rmsd_stds = []
        
        for r in recycles:
            data = model_data[r]
            if data.rmsd_ref1 is not None:
                rmsd_array = np.array(data.rmsd_ref1)
                rmsd_means.append(np.mean(rmsd_array))
                rmsd_stds.append(np.std(rmsd_array))
        
        if rmsd_means:
            if max_rmsd is None:
                rmsd1_max = max(rmsd1_max, max(m + s for m, s in zip(rmsd_means, rmsd_stds)))
            ax1.errorbar(recycles, rmsd_means, yerr=rmsd_stds,
                        fmt='o-', label=f'Model {model}',
                        color=colors[idx], capsize=3,
                        markersize=5, linewidth=1.5, capthick=1)
    
    ax1.set_xlabel('Recycle', fontsize=10, labelpad=8)
    ax1.set_ylabel('RMSD to Reference 1 (Å)', fontsize=10, labelpad=8)
    ax1.set_title('RMSD1 Progression', fontsize=12, pad=10, fontweight='bold')
    ax1.legend(fontsize=8, frameon=True, fancybox=True,
              facecolor='white', edgecolor='gray')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.set_ylim(0, rmsd1_max * 1.1)  # Add 10% padding
    
    # Plot RMSD2 progression
    ax2 = plt.subplot(gs[0, 1])
    rmsd2_max = max_rmsd if max_rmsd is not None else 0
    for idx, (model, model_data) in enumerate(sorted(recycle_data.items())):
        recycles = sorted(model_data.keys())
        rmsd_means = []
        rmsd_stds = []
        valid_recycles = []
        
        for r in recycles:
            data = model_data[r]
            if hasattr(data, 'rmsd_ref2') and data.rmsd_ref2 is not None:
                rmsd_array = np.array(data.rmsd_ref2)
                rmsd_means.append(np.mean(rmsd_array))
                rmsd_stds.append(np.std(rmsd_array))
                valid_recycles.append(r)
        
        if valid_recycles:
            if max_rmsd is None:
                rmsd2_max = max(rmsd2_max, max(m + s for m, s in zip(rmsd_means, rmsd_stds)))
            ax2.errorbar(valid_recycles, rmsd_means, yerr=rmsd_stds,
                        fmt='o-', label=f'Model {model}',
                        color=colors[idx], capsize=3,
                        markersize=5, linewidth=1.5, capthick=1)
    
    ax2.set_xlabel('Recycle', fontsize=10, labelpad=8)
    ax2.set_ylabel('RMSD to Reference 2 (Å)', fontsize=10, labelpad=8)
    ax2.set_title('RMSD2 Progression', fontsize=12, pad=10, fontweight='bold')
    ax2.legend(fontsize=8, frameon=True, fancybox=True,
              facecolor='white', edgecolor='gray')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.set_ylim(0, rmsd2_max * 1.1)  # Add 10% padding
    
    # Plot pLDDT progression
    ax3 = plt.subplot(gs[1, 0])
    for idx, (model, model_data) in enumerate(sorted(recycle_data.items())):
        recycles = sorted(model_data.keys())
        plddt_means = []
        plddt_stds = []
        
        for r in recycles:
            data = model_data[r]
            if data.plddt is not None:
                plddt_array = np.array(data.plddt)
                if len(plddt_array.shape) > 2:
                    plddt_array = plddt_array.mean(axis=0)
                if len(plddt_array.shape) == 2:
                    plddt_array = plddt_array[0]
                # Scale pLDDT values to 0-100 range if they're in 0-1 range
                if np.max(plddt_array) <= 1.0:
                    plddt_array = plddt_array * 100
                plddt_means.append(np.mean(plddt_array))
                plddt_stds.append(np.std(plddt_array))
        
        if plddt_means:
            ax3.errorbar(recycles, plddt_means, yerr=plddt_stds,
                        fmt='o-', label=f'Model {model}',
                        color=colors[idx], capsize=3,
                        markersize=5, linewidth=1.5, capthick=1)
    
    ax3.set_xlabel('Recycle', fontsize=10, labelpad=8)
    ax3.set_ylabel('Mean pLDDT', fontsize=10, labelpad=8)
    ax3.set_title('pLDDT Progression', fontsize=12, pad=10, fontweight='bold')
    ax3.legend(fontsize=8, frameon=True, fancybox=True,
              facecolor='white', edgecolor='gray')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.set_ylim(0, 100)  # Set fixed range for pLDDT
    
    # Plot per-residue pLDDT heatmap
    ax4 = plt.subplot(gs[1, 1])
    plddt_matrix = []
    row_labels = []
    
    for model in sorted(recycle_data.keys()):
        for recycle in sorted(recycle_data[model].keys()):
            data = recycle_data[model][recycle]
            if data.plddt is not None:
                plddt_array = np.array(data.plddt)
                if len(plddt_array.shape) > 2:
                    plddt_array = plddt_array.mean(axis=0)
                if len(plddt_array.shape) == 2:
                    plddt_array = plddt_array[0]
                # Scale pLDDT values to 0-100 range if they're in 0-1 range
                if np.max(plddt_array) <= 1.0:
                    plddt_array = plddt_array * 100
                plddt_matrix.append(plddt_array)
                row_labels.append(f'M{model} R{recycle}')  # Shortened labels
    
    if plddt_matrix:
        plddt_matrix = np.array(plddt_matrix)
        # Create AlphaFold-style colormap
        af_colors = [
            (0.0, '#FF7D45'),  # Very low
            (0.5, '#FFF300'),  # Low
            (0.7, '#00A1D3'),  # Confident
            (0.9, '#0053D6'),  # Very high
            (1.0, '#0053D6')   # Very high (extended)
        ]
        af_cmap = LinearSegmentedColormap.from_list('alphafold', af_colors)
        
        im = ax4.imshow(plddt_matrix, aspect='auto', cmap=af_cmap,
                       vmin=0, vmax=100)
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('pLDDT', fontsize=10, labelpad=8)
        cbar.ax.tick_params(labelsize=8)
        
        ax4.set_yticks(range(len(row_labels)))
        ax4.set_yticklabels(row_labels, fontsize=8)
        ax4.set_xlabel('Residue', fontsize=10, labelpad=8)
        ax4.set_title('Per-residue pLDDT', fontsize=12, pad=10, fontweight='bold')
        ax4.tick_params(axis='both', which='major', labelsize=8)
    
    # Add overall title with adjusted position
    fig.suptitle(f'Position {position} - Recycle Analysis Summary',
                fontsize=14, y=0.95, fontweight='bold')
    
    # Adjust layout with specific spacing
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                       wspace=0.3, hspace=0.3)
    
    if output_dir:
        output_path = output_dir / f"pos_{position}_summary.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return fig

def create_model_comparison(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready model comparison plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    # Get recycle data
    recycle_data = storage.get_all_recycle_data(position)
    if not recycle_data:
        logger.warning(f"No data found for position {position}")
        return figures
    
    # Create figure with publication-friendly size and DPI
    fig = plt.figure(figsize=(15, 10), dpi=300)
    
    # Determine if we have rmsd2 data
    has_rmsd2 = any(hasattr(data, 'rmsd_ref2') and data.rmsd_ref2 is not None
                    for model_data in recycle_data.values()
                    for data in model_data.values())
    
    # Adjust grid based on presence of rmsd2
    if has_rmsd2:
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Get models and recycles
    models = sorted(recycle_data.keys())
    recycles = sorted(set(r for model_data in recycle_data.values() for r in model_data.keys()))
    
    # Create matrices for heatmaps
    rmsd1_matrix = np.zeros((len(models), len(recycles)))
    rmsd2_matrix = np.zeros((len(models), len(recycles))) if has_rmsd2 else None
    plddt_matrix = np.zeros((len(models), len(recycles)))
    
    # Calculate global max RMSD if not provided
    if max_rmsd is None:
        max_rmsd = 0
        for model in models:
            for recycle in recycles:
                if recycle in recycle_data[model]:
                    data = recycle_data[model][recycle]
                    if data.rmsd_ref1 is not None:
                        max_rmsd = max(max_rmsd, np.max(data.rmsd_ref1))
                        if has_rmsd2 and data.rmsd_ref2 is not None:
                            max_rmsd = max(max_rmsd, np.max(data.rmsd_ref2))
        max_rmsd *= 1.1  # Add 10% padding
    
    for i, model in enumerate(models):
        for j, recycle in enumerate(recycles):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    rmsd1_matrix[i, j] = np.mean(data.rmsd_ref1)
                    if has_rmsd2 and data.rmsd_ref2 is not None:
                        rmsd2_matrix[i, j] = np.mean(data.rmsd_ref2)
                    if data.plddt is not None:
                        plddt_array = np.array(data.plddt)
                        if len(plddt_array.shape) > 2:
                            plddt_array = plddt_array.mean(axis=0)
                        if len(plddt_array.shape) == 2:
                            plddt_array = plddt_array[0]
                        if np.max(plddt_array) <= 1.0:
                            plddt_array = plddt_array * 100
                        plddt_matrix[i, j] = np.mean(plddt_array)
    
    # Create Cesar's colormap
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
    cesar_cmap = LinearSegmentedColormap.from_list('cesar', list(zip(positions, cesar_colors)))
    
    # Plot RMSD1 heatmap
    ax1 = plt.subplot(gs[0, 0])
    im1 = ax1.imshow(rmsd1_matrix, aspect='auto', cmap=cesar_cmap,
                    vmin=0, vmax=max_rmsd)
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.ax.set_ylabel('RMSD to Reference 1 (Å)', fontsize=10, labelpad=8)
    cbar1.ax.tick_params(labelsize=8)
    
    # Customize heatmap appearance
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels([f'Model {m}' for m in models], fontsize=8)
    ax1.set_xticks(range(len(recycles)))
    ax1.set_xticklabels([f'R{r}' for r in recycles], fontsize=8)
    ax1.set_xlabel('Recycle', fontsize=10, labelpad=8)
    ax1.set_title('Model-Recycle RMSD1 Performance', 
                 fontsize=12, pad=10, fontweight='bold')
    
    # Plot RMSD2 heatmap if available
    if has_rmsd2:
        ax2 = plt.subplot(gs[0, 1])
        im2 = ax2.imshow(rmsd2_matrix, aspect='auto', cmap=cesar_cmap,
                        vmin=0, vmax=max_rmsd)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.ax.set_ylabel('RMSD to Reference 2 (Å)', fontsize=10, labelpad=8)
        cbar2.ax.tick_params(labelsize=8)
        
        # Customize heatmap appearance
        ax2.set_yticks(range(len(models)))
        ax2.set_yticklabels([f'Model {m}' for m in models], fontsize=8)
        ax2.set_xticks(range(len(recycles)))
        ax2.set_xticklabels([f'R{r}' for r in recycles], fontsize=8)
        ax2.set_xlabel('Recycle', fontsize=10, labelpad=8)
        ax2.set_title('Model-Recycle RMSD2 Performance', 
                     fontsize=12, pad=10, fontweight='bold')
    
    # Plot best performance metrics
    ax3 = plt.subplot(gs[1, :])  # Span all columns
    
    # Calculate metrics
    metrics = {
        'Min RMSD1': [],
        'Max pLDDT': []
    }
    
    for model in models:
        model_data = recycle_data[model]
        rmsd1_values = []
        plddt_values = []
        
        for recycle, data in model_data.items():
            if data.rmsd_ref1 is not None:
                rmsd1_values.extend(np.array(data.rmsd_ref1).flatten())
            if data.plddt is not None:
                plddt_array = np.array(data.plddt)
                if len(plddt_array.shape) > 2:
                    plddt_array = plddt_array.mean(axis=0)
                if len(plddt_array.shape) == 2:
                    plddt_array = plddt_array[0]
                if np.max(plddt_array) <= 1.0:
                    plddt_array = plddt_array * 100
                plddt_values.extend(plddt_array.flatten())
        
        metrics['Min RMSD1'].append(min(rmsd1_values) if rmsd1_values else np.nan)
        metrics['Max pLDDT'].append(max(plddt_values) if plddt_values else np.nan)
    
    if has_rmsd2:
        metrics['Min RMSD2'] = []
        for model in models:
            model_data = recycle_data[model]
            rmsd2_values = []
            
            for recycle, data in model_data.items():
                if hasattr(data, 'rmsd_ref2') and data.rmsd_ref2 is not None:
                    rmsd2_values.extend(np.array(data.rmsd_ref2).flatten())
            
            metrics['Min RMSD2'].append(min(rmsd2_values) if rmsd2_values else np.nan)
    
    # Create bar plot
    x = np.arange(len(models))
    width = 0.2
    n_metrics = len(metrics)
    offsets = np.linspace(-(n_metrics-1)*width/2, (n_metrics-1)*width/2, n_metrics)
    
    # Plot bars for each metric
    for i, (metric, values) in enumerate(metrics.items()):
        ax3.bar(x + offsets[i], values, width, label=metric)
    
    # Customize bar plot
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Model {m}' for m in models], fontsize=8)
    ax3.set_ylabel('Value', fontsize=10, labelpad=8)
    ax3.set_title('Best Performance Metrics by Model', 
                 fontsize=12, pad=10, fontweight='bold')
    ax3.legend(fontsize=8, frameon=True, fancybox=True,
              facecolor='white', edgecolor='gray')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    
    # Add overall title
    plt.suptitle(f'Position {position} - Model Performance Analysis',
                fontsize=14, y=1.02, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    figures['model_comparison'] = fig
    
    if output_dir:
        output_path = output_dir / f"pos_{position}_model_comparison.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return figures

def create_cumulative_landscapes(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready cumulative RMSD landscape plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    # Get recycle data
    recycle_data = storage.get_all_recycle_data(position)
    
    # Process each model
    for model in sorted(recycle_data.keys()):
        # Initialize data collections
        all_recycles_rmsd1 = []  # All recycles combined
        all_recycles_rmsd2 = []
        cumulative_0n_rmsd1 = []  # Cumulative from recycle 0 to N
        cumulative_0n_rmsd2 = []
        cumulative_1n_rmsd1 = []  # Cumulative from recycle 1 to N
        cumulative_1n_rmsd2 = []
        
        # Collect data for all plots
        for recycle in sorted(recycle_data[model].keys()):
            data = recycle_data[model][recycle]
            if data.rmsd_ref1 is not None:
                rmsd1_values = np.array(data.rmsd_ref1).flatten()
                rmsd2_values = np.array(data.rmsd_ref2).flatten() if data.rmsd_ref2 is not None else None
                
                # Store all recycles data
                all_recycles_rmsd1.extend(rmsd1_values)
                if rmsd2_values is not None:
                    all_recycles_rmsd2.extend(rmsd2_values)
                
                # Add to cumulative 0-N data
                cumulative_0n_rmsd1.extend(rmsd1_values)
                if rmsd2_values is not None:
                    cumulative_0n_rmsd2.extend(rmsd2_values)
                
                # Add to 1-N data if not recycle 0
                if int(recycle) > 0:
                    cumulative_1n_rmsd1.extend(rmsd1_values)
                    if rmsd2_values is not None:
                        cumulative_1n_rmsd2.extend(rmsd2_values)
        
        # Create figure with publication-friendly size
        fig = plt.figure(figsize=(15, 5), dpi=300)
        
        # Create grid for all three plots with proper spacing
        gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])
        
        # Create Cesar's colormap
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
        cesar_cmap = LinearSegmentedColormap.from_list('cesar', list(zip(positions, cesar_colors)))
        
        # Find global max RMSD for axis alignment
        if max_rmsd is None:
            all_rmsd1 = np.concatenate([all_recycles_rmsd1, cumulative_0n_rmsd1, cumulative_1n_rmsd1])
            all_rmsd2 = []
            if all_recycles_rmsd2:
                all_rmsd2.extend(all_recycles_rmsd2)
            if cumulative_0n_rmsd2:
                all_rmsd2.extend(cumulative_0n_rmsd2)
            if cumulative_1n_rmsd2:
                all_rmsd2.extend(cumulative_1n_rmsd2)
            
            global_max_rmsd = max(max(all_rmsd1), max(all_rmsd2) if all_rmsd2 else max(all_rmsd1)) * 1.1
        else:
            global_max_rmsd = max_rmsd
        
        # Create each landscape plot
        hist2d_list = []  # Store hist2d objects for colorbar
        for i, (rmsd1, rmsd2, title) in enumerate([
            (all_recycles_rmsd1, all_recycles_rmsd2, "All Recycles Combined"),
            (cumulative_0n_rmsd1, cumulative_0n_rmsd2, "Cumulative (0-N)"),
            (cumulative_1n_rmsd1, cumulative_1n_rmsd2, "Cumulative (1-N)")
        ]):
            # Create subplot grid with better spacing
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, i],
                                                      width_ratios=[4, 1],
                                                      height_ratios=[1, 4],
                                                      hspace=0.0, wspace=0.0)
            
            # Create subplots
            ax_main = plt.subplot(inner_gs[1, 0])
            ax_top = plt.subplot(inner_gs[0, 0], sharex=ax_main)
            ax_right = plt.subplot(inner_gs[1, 1], sharey=ax_main)
            
            # Create landscape plot
            y_data = rmsd2 if rmsd2 is not None else rmsd1
            
            # Create main 2D histogram
            hist2d = ax_main.hist2d(rmsd1, y_data, bins=40,
                                  range=[[0, global_max_rmsd], [0, global_max_rmsd]],
                                  cmap=cesar_cmap,
                                  norm=LogNorm(vmin=1))  # Set minimum value to 1 to avoid log(0)
            hist2d_list.append(hist2d[3])
            
            # Add diagonal line with improved visibility
            ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                        alpha=0.8, linewidth=1.5, dashes=(5, 5))
            
            # Customize main plot
            ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
            ax_main.set_ylabel('RMSD vs Reference 2 (Å)' if rmsd2 is not None else 'RMSD vs Reference 1 (Å)',
                             fontsize=10, labelpad=8)
            ax_main.tick_params(axis='both', which='major', labelsize=8)
            ax_main.tick_params(axis='both', which='minor', labelsize=6)
            ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
            ax_main.minorticks_on()
            
            # Create top histogram with matching color
            counts_top, edges_top, _ = ax_top.hist(rmsd1, bins=40, range=(0, global_max_rmsd),
                                                 color=cesar_colors[2], alpha=0.7,
                                                 edgecolor='black', linewidth=0.5)
            ax_top.tick_params(axis='x', labelbottom=False)
            ax_top.tick_params(axis='y', labelsize=8)
            ax_top.grid(True, linestyle='--', alpha=0.3)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['bottom'].set_visible(False)
            ax_top.set_title(title, fontsize=10, pad=5)
            
            # Create right histogram with matching color
            counts_right, edges_right, _ = ax_right.hist(y_data, bins=40, range=(0, global_max_rmsd),
                                                       orientation='horizontal', color=cesar_colors[2],
                                                       alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_right.tick_params(axis='y', labelleft=False)
            ax_right.tick_params(axis='x', labelsize=8)
            ax_right.grid(True, linestyle='--', alpha=0.3)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            
            # Ensure square aspect ratio for main plot
            ax_main.set_aspect('equal')
            
            # Set the same limits for top and right histograms
            ax_top.set_ylim(0, max(counts_top) * 1.1)  # Add 10% padding
            ax_right.set_xlim(0, max(counts_right) * 1.1)  # Add 10% padding
            
            # Remove main plot spines between histograms
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)
            
            # Set x and y limits for main plot
            ax_main.set_xlim(0, global_max_rmsd)
            ax_main.set_ylim(0, global_max_rmsd)
        
        # Create shared colorbar
        ax_cbar = plt.subplot(gs[0, -1])
        cbar = plt.colorbar(hist2d_list[0], cax=ax_cbar, orientation='vertical')
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_title('Density', fontsize=8, pad=3)
        
        # Format colorbar ticks to handle low frequencies better
        formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
        cbar.formatter = formatter
        cbar.update_ticks()
        
        # Adjust colorbar tick positions for better spacing
        tick_locator = LogLocator(base=10, numticks=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        
        # Add overall title with adjusted position
        fig.suptitle(f"Position {position} - Model {model} RMSD Landscapes",
                    fontsize=12, y=0.95, fontweight='bold')
        
        # Adjust layout
        plt.subplots_adjust(left=0.01, right=0.95, top=0.9, bottom=0.15, wspace=0.32)
        
        figures[f'model_{model}_landscapes'] = fig
        
        if output_dir:
            output_path = output_dir / f"pos_{position}_model_{model}_landscapes.{format}"
            fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
            if not show:
                plt.close(fig)
    
    return figures

def create_summary_collages(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Create summary collages of RMSD landscapes.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    if output_dir:
        # Create summary directory
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Get recycle data
    recycle_data = storage.get_all_recycle_data(position)
    if not recycle_data:
        logger.warning(f"No data found for position {position}")
        return figures
    
    # Find max recycle number across all models
    max_recycles = max(max(r for r in model_data.keys()) for model_data in recycle_data.values())
    
    # 1. Individual recycles collage
    individual_plots = []
    for recycle in range(max_recycles + 1):
        rmsd1_values = []
        rmsd2_values = []
        
        for model in sorted(recycle_data.keys()):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    rmsd1_values.extend(np.array(data.rmsd_ref1).flatten())
                    if data.rmsd_ref2 is not None:
                        rmsd2_values.extend(np.array(data.rmsd_ref2).flatten())
        
        if rmsd1_values:
            individual_plots.append((
                np.array(rmsd1_values),
                np.array(rmsd2_values) if rmsd2_values else None,
                f"Recycle {recycle}"
            ))
    
    if individual_plots:
        fig = create_collage(
            individual_plots,
            f"Position {position} - Individual Recycle Landscapes",
            max_rmsd=max_rmsd
        )
        figures['individual_collage'] = fig
        
        if output_dir:
            fig.savefig(summary_dir / f"pos_{position}_individual_recycles.{format}",
                       format=format, bbox_inches='tight', dpi=300)
            if not show:
                plt.close(fig)
    
    # 2. Cumulative plots (0-N) collage
    cumulative_plots = []
    cumulative_rmsd1 = []
    cumulative_rmsd2 = []
    
    for recycle in range(max_recycles + 1):
        for model in sorted(recycle_data.keys()):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    cumulative_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                    if data.rmsd_ref2 is not None:
                        cumulative_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if cumulative_rmsd1:
            cumulative_plots.append((
                np.array(cumulative_rmsd1),
                np.array(cumulative_rmsd2) if cumulative_rmsd2 else None,
                f"Cumulative 0-{recycle}"
            ))
    
    if cumulative_plots:
        fig = create_collage(
            cumulative_plots,
            f"Position {position} - Cumulative Landscapes (0-N)",
            max_rmsd=max_rmsd
        )
        figures['cumulative_collage'] = fig
        
        if output_dir:
            fig.savefig(summary_dir / f"pos_{position}_cumulative_0n.{format}",
                       format=format, bbox_inches='tight', dpi=300)
            if not show:
                plt.close(fig)
    
    # 3. Cumulative (1-N) collage
    cumulative_1n_plots = []
    cumulative_1n_rmsd1 = []
    cumulative_1n_rmsd2 = []
    
    for recycle in range(1, max_recycles + 1):
        for model in sorted(recycle_data.keys()):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    cumulative_1n_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                    if data.rmsd_ref2 is not None:
                        cumulative_1n_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if cumulative_1n_rmsd1:
            cumulative_1n_plots.append((
                np.array(cumulative_1n_rmsd1),
                np.array(cumulative_1n_rmsd2) if cumulative_1n_rmsd2 else None,
                f"Cumulative 1-{recycle}"
            ))
    
    if cumulative_1n_plots:
        fig = create_collage(
            cumulative_1n_plots,
            f"Position {position} - Cumulative Landscapes (1-N)",
            max_rmsd=max_rmsd
        )
        figures['cumulative_1n_collage'] = fig
        
        if output_dir:
            fig.savefig(summary_dir / f"pos_{position}_cumulative_1n.{format}",
                       format=format, bbox_inches='tight', dpi=300)
            if not show:
                plt.close(fig)
    
    return figures

def create_model_comparisons(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Create model comparison plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    if output_dir:
        # Create comparisons directory
        comparison_dir = output_dir / "model_comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Get recycle data
    recycle_data = storage.get_all_recycle_data(position)
    if not recycle_data:
        logger.warning(f"No data found for position {position}")
        return figures
    
    # Find max recycle number across all models
    max_recycles = max(max(r for r in model_data.keys()) for model_data in recycle_data.values())
    
    # Create Cesar's colormap
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
    cesar_cmap = LinearSegmentedColormap.from_list('cesar', list(zip(positions, cesar_colors)))
    
    # 1. Per-recycle model comparison
    for recycle in range(max_recycles + 1):
        # Create figure with publication-friendly size
        fig = plt.figure(figsize=(15, 5), dpi=300)
        
        # Create grid for all plots with proper spacing
        n_models = len(recycle_data)
        gs = gridspec.GridSpec(1, n_models + 1, figure=fig, width_ratios=[1] * n_models + [0.05])
        
        # Find global max RMSD for axis alignment
        if max_rmsd is None:
            all_rmsd1 = []
            all_rmsd2 = []
            for model in sorted(recycle_data.keys()):
                if recycle in recycle_data[model]:
                    data = recycle_data[model][recycle]
                    if data.rmsd_ref1 is not None:
                        all_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            all_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
            
            global_max_rmsd = max(max(all_rmsd1), max(all_rmsd2) if all_rmsd2 else max(all_rmsd1)) * 1.1
        else:
            global_max_rmsd = max_rmsd
        
        # Create each model plot
        hist2d_list = []  # Store hist2d objects for colorbar
        for i, model in enumerate(sorted(recycle_data.keys())):
            if recycle not in recycle_data[model]:
                continue
                
            data = recycle_data[model][recycle]
            if data.rmsd_ref1 is None:
                continue
                
            rmsd1 = np.array(data.rmsd_ref1).flatten()
            rmsd2 = np.array(data.rmsd_ref2).flatten() if data.rmsd_ref2 is not None else None
            y_data = rmsd2 if rmsd2 is not None else rmsd1
            
            # Create subplot grid with better spacing
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, i],
                                                      width_ratios=[4, 1],
                                                      height_ratios=[1, 4],
                                                      hspace=0.0, wspace=0.0)
            
            # Create subplots
            ax_main = plt.subplot(inner_gs[1, 0])
            ax_top = plt.subplot(inner_gs[0, 0], sharex=ax_main)
            ax_right = plt.subplot(inner_gs[1, 1], sharey=ax_main)
            
            # Create main 2D histogram
            hist2d = ax_main.hist2d(rmsd1, y_data, bins=40,
                                  range=[[0, global_max_rmsd], [0, global_max_rmsd]],
                                  cmap=cesar_cmap,
                                  norm=LogNorm(vmin=1))  # Set minimum value to 1 to avoid log(0)
            hist2d_list.append(hist2d[3])
            
            # Add diagonal line with improved visibility
            ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                        alpha=0.8, linewidth=1.5, dashes=(5, 5))
            
            # Customize main plot
            ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
            ax_main.set_ylabel('RMSD vs Reference 2 (Å)' if rmsd2 is not None else 'RMSD vs Reference 1 (Å)',
                             fontsize=10, labelpad=8)
            ax_main.tick_params(axis='both', which='major', labelsize=8)
            ax_main.tick_params(axis='both', which='minor', labelsize=6)
            ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
            ax_main.minorticks_on()
            
            # Create top histogram with matching color
            counts_top, edges_top, _ = ax_top.hist(rmsd1, bins=40, range=(0, global_max_rmsd),
                                                 color=cesar_colors[2], alpha=0.7,
                                                 edgecolor='black', linewidth=0.5)
            ax_top.tick_params(axis='x', labelbottom=False)
            ax_top.tick_params(axis='y', labelsize=8)
            ax_top.grid(True, linestyle='--', alpha=0.3)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['bottom'].set_visible(False)
            ax_top.set_title(f"Model {model}", fontsize=10, pad=5)
            
            # Create right histogram with matching color
            counts_right, edges_right, _ = ax_right.hist(y_data, bins=40, range=(0, global_max_rmsd),
                                                       orientation='horizontal', color=cesar_colors[2],
                                                       alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_right.tick_params(axis='y', labelleft=False)
            ax_right.tick_params(axis='x', labelsize=8)
            ax_right.grid(True, linestyle='--', alpha=0.3)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['left'].set_visible(False)
            
            # Ensure square aspect ratio for main plot
            ax_main.set_aspect('equal')
            
            # Set the same limits for top and right histograms
            ax_top.set_ylim(0, max(counts_top) * 1.1)  # Add 10% padding
            ax_right.set_xlim(0, max(counts_right) * 1.1)  # Add 10% padding
            
            # Remove main plot spines between histograms
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)
            
            # Set x and y limits for main plot
            ax_main.set_xlim(0, global_max_rmsd)
            ax_main.set_ylim(0, global_max_rmsd)
        
        # Create shared colorbar
        ax_cbar = plt.subplot(gs[0, -1])
        cbar = plt.colorbar(hist2d_list[0], cax=ax_cbar, orientation='vertical')
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_title('Density', fontsize=8, pad=3)
        
        # Format colorbar ticks to handle low frequencies better
        formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
        cbar.formatter = formatter
        cbar.update_ticks()
        
        # Adjust colorbar tick positions for better spacing
        tick_locator = LogLocator(base=10, numticks=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        
        # Add overall title with adjusted position
        fig.suptitle(f"Position {position} - Model Comparison for Recycle {recycle}",
                    fontsize=12, y=0.95, fontweight='bold')
        
        # Adjust layout
        plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
        
        figures[f'model_comparison_r{recycle}'] = fig
        
        if output_dir:
            fig.savefig(comparison_dir / f"pos_{position}_model_comparison_r{recycle}.{format}",
                       format=format, bbox_inches='tight', dpi=300)
            if not show:
                plt.close(fig)
    
    return figures 