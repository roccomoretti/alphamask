"""
Recycle analysis visualization functionality.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
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
    ncols: int = 4
) -> plt.Figure:
    """
    Create a collage of RMSD landscape plots.
    
    Args:
        plots_data: List of (rmsd1, rmsd2, subtitle) tuples
        title: Overall title for the collage
        ncols: Number of columns in the grid
        
    Returns:
        Generated figure
    """
    nrows = (len(plots_data) + ncols - 1) // ncols
    fig = plt.figure(figsize=(5*ncols, 5*nrows), dpi=300)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    
    for idx, (rmsd1, rmsd2, subtitle) in enumerate(plots_data):
        if idx >= nrows * ncols:
            break
            
        # Create subplot with proper grid spec
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[idx // ncols, idx % ncols],
                                                  width_ratios=[4, 1], height_ratios=[1, 4],
                                                  hspace=0.02, wspace=0.02)
        
        # Create subplots
        ax_main = plt.subplot(inner_gs[1, 0])
        ax_top = plt.subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = plt.subplot(inner_gs[1, 1], sharey=ax_main)
        ax_cbar = plt.subplot(inner_gs[0, 1])
        
        # Create landscape plot
        y_data = rmsd2 if rmsd2 is not None else rmsd1
        max_rmsd = max(max(rmsd1), max(y_data)) * 1.1 if rmsd2 is not None else max(rmsd1) * 1.1
        
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
        
        # Create main 2D histogram
        hist2d = ax_main.hist2d(rmsd1, y_data, bins=40,
                              range=[[0, max_rmsd], [0, max_rmsd]],
                              cmap=cesar_cmap,
                              norm=LogNorm())
        
        # Add diagonal line
        ax_main.plot([0, max_rmsd], [0, max_rmsd], '--', color='black',
                    alpha=0.8, linewidth=1.5, dashes=(5, 5))
        
        # Customize main plot
        ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=8, labelpad=5)
        ax_main.set_ylabel('RMSD vs Reference 2 (Å)', fontsize=8, labelpad=5)
        ax_main.tick_params(axis='both', which='major', labelsize=6)
        ax_main.tick_params(axis='both', which='minor', labelsize=4)
        ax_main.grid(True, linestyle='--', alpha=0.3)
        
        # Create top histogram
        ax_top.hist(rmsd1, bins=40, range=(0, max_rmsd),
                   color=cesar_colors[2], alpha=0.7,
                   edgecolor='black', linewidth=0.5)
        ax_top.tick_params(axis='x', labelbottom=False)
        ax_top.tick_params(axis='y', labelsize=6)
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        
        # Create right histogram
        ax_right.hist(y_data, bins=40, range=(0, max_rmsd),
                     orientation='horizontal', color=cesar_colors[2],
                     alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_right.tick_params(axis='y', labelleft=False)
        ax_right.tick_params(axis='x', labelsize=6)
        ax_right.grid(True, linestyle='--', alpha=0.3)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        
        # Add colorbar
        cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='horizontal')
        cbar.ax.tick_params(labelsize=6)
        cbar.ax.set_title('Density', fontsize=6, pad=2)
        
        # Set subplot title
        ax_top.set_title(subtitle, fontsize=10, pad=5)
        
        # Ensure square aspect ratio
        ax_main.set_aspect('equal')
    
    # Set overall title
    plt.suptitle(title, fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_recycle_landscapes(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready RMSD landscape plots for recycle analysis.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
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
                show=False
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
                show=False
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
                show=False
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
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready recycle progression plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
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
    
    rmsd1_max = 0
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
        rmsd2_max = 0
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
    show: bool = False
) -> plt.Figure:
    """
    Create summary plot comparing all recycles.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
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
    
    rmsd1_max = 0
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
    rmsd2_max = 0
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
    
    plt.suptitle(f'Position {position} - Recycle Analysis Summary',
                fontsize=14, y=0.95, fontweight='bold')
    plt.tight_layout()
    
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
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready model comparison plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
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
    im1 = ax1.imshow(rmsd1_matrix, aspect='auto', cmap=cesar_cmap)
    
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
        im2 = ax2.imshow(rmsd2_matrix, aspect='auto', cmap=cesar_cmap)
        
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
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready cumulative RMSD landscape plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    # Get recycle data
    recycle_data = storage.get_all_recycle_data(position)
    
    # Process each model
    for model in sorted(recycle_data.keys()):
        # Initialize data collections
        recycle_rmsd1 = []
        recycle_rmsd2 = []
        cumulative_rmsd1 = []
        cumulative_rmsd2 = []
        cumulative_1n_rmsd1 = []
        cumulative_1n_rmsd2 = []
        
        # Collect data for all plots
        for recycle in sorted(recycle_data[model].keys()):
            data = recycle_data[model][recycle]
            if data.rmsd_ref1 is not None:
                rmsd1_values = np.array(data.rmsd_ref1).flatten()
                rmsd2_values = np.array(data.rmsd_ref2).flatten() if data.rmsd_ref2 is not None else None
                
                # Store current recycle data
                recycle_rmsd1.extend(rmsd1_values)
                if rmsd2_values is not None:
                    recycle_rmsd2.extend(rmsd2_values)
                
                # Add to cumulative data
                cumulative_rmsd1.extend(rmsd1_values)
                if rmsd2_values is not None:
                    cumulative_rmsd2.extend(rmsd2_values)
                
                # Add to 1-N data if not recycle 0
                if int(recycle) > 0:
                    cumulative_1n_rmsd1.extend(rmsd1_values)
                    if rmsd2_values is not None:
                        cumulative_1n_rmsd2.extend(rmsd2_values)
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(15, 5), dpi=300)
        
        # Create grid for all three plots
        gs = gridspec.GridSpec(1, 3)
        
        # Helper function to create landscape in subplot
        def create_landscape_subplot(ax_main, ax_top, ax_right, ax_cbar, rmsd1, rmsd2, subtitle):
            y_data = rmsd2 if rmsd2 is not None else rmsd1
            max_rmsd = max(max(rmsd1), max(y_data)) * 1.1 if rmsd2 is not None else max(rmsd1) * 1.1
            
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
            
            # Create main 2D histogram
            hist2d = ax_main.hist2d(rmsd1, y_data, bins=40,
                                  range=[[0, max_rmsd], [0, max_rmsd]],
                                  cmap=cesar_cmap,
                                  norm=LogNorm())
            
            # Add diagonal line
            ax_main.plot([0, max_rmsd], [0, max_rmsd], '--', color='black',
                        alpha=0.8, linewidth=1.5, dashes=(5, 5))
            
            # Customize main plot
            ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=8, labelpad=5)
            ax_main.set_ylabel('RMSD vs Reference 2 (Å)', fontsize=8, labelpad=5)
            ax_main.tick_params(axis='both', which='major', labelsize=6)
            ax_main.tick_params(axis='both', which='minor', labelsize=4)
            ax_main.grid(True, linestyle='--', alpha=0.3)
            
            # Create top histogram
            ax_top.hist(rmsd1, bins=40, range=(0, max_rmsd),
                       color=cesar_colors[2], alpha=0.7,
                       edgecolor='black', linewidth=0.5)
            ax_top.tick_params(axis='x', labelbottom=False)
            ax_top.tick_params(axis='y', labelsize=6)
            ax_top.grid(True, linestyle='--', alpha=0.3)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            
            # Create right histogram
            ax_right.hist(y_data, bins=40, range=(0, max_rmsd),
                         orientation='horizontal', color=cesar_colors[2],
                         alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_right.tick_params(axis='y', labelleft=False)
            ax_right.tick_params(axis='x', labelsize=6)
            ax_right.grid(True, linestyle='--', alpha=0.3)
            ax_right.spines['top'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            
            # Add colorbar
            cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='horizontal')
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.set_title('Density', fontsize=6, pad=2)
            
            # Set subplot title
            ax_top.set_title(subtitle, fontsize=10, pad=5)
            
            # Ensure square aspect ratio
            ax_main.set_aspect('equal')
        
        # Create each landscape plot
        for i, (rmsd1, rmsd2, title) in enumerate([
            (recycle_rmsd1, recycle_rmsd2, "Current Recycle"),
            (cumulative_rmsd1, cumulative_rmsd2, "Cumulative (0-N)"),
            (cumulative_1n_rmsd1, cumulative_1n_rmsd2, "Cumulative (1-N)")
        ]):
            # Create subplot grid
            inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[i],
                                                      width_ratios=[4, 1],
                                                      height_ratios=[1, 4],
                                                      hspace=0.02, wspace=0.02)
            
            # Create subplots
            ax_main = plt.subplot(inner_gs[1, 0])
            ax_top = plt.subplot(inner_gs[0, 0], sharex=ax_main)
            ax_right = plt.subplot(inner_gs[1, 1], sharey=ax_main)
            ax_cbar = plt.subplot(inner_gs[0, 1])
            
            create_landscape_subplot(ax_main, ax_top, ax_right, ax_cbar,
                                  rmsd1, rmsd2, title)
        
        plt.suptitle(f"Position {position} - Model {model} RMSD Landscapes",
                    fontsize=14, y=1.02, fontweight='bold')
        
        plt.tight_layout()
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
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create summary collages of RMSD landscapes.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
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
            f"Position {position} - Individual Recycle Landscapes"
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
            f"Position {position} - Cumulative Landscapes (0-N)"
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
            f"Position {position} - Cumulative Landscapes (1-N)"
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
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create model comparison plots.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
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
    
    # 1. Per-recycle model comparison
    for recycle in range(max_recycles + 1):
        model_plots = []
        
        for model in sorted(recycle_data.keys()):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    rmsd1_values = np.array(data.rmsd_ref1).flatten()
                    rmsd2_values = np.array(data.rmsd_ref2).flatten() if data.rmsd_ref2 is not None else None
                    
                    model_plots.append((
                        rmsd1_values,
                        rmsd2_values,
                        f"Model {model}"
                    ))
        
        if model_plots:
            fig = create_collage(
                model_plots,
                f"Position {position} - Model Comparison for Recycle {recycle}",
                ncols=3
            )
            figures[f'model_comparison_r{recycle}'] = fig
            
            if output_dir:
                fig.savefig(comparison_dir / f"pos_{position}_model_comparison_r{recycle}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    # 2. Per-model cumulative comparison
    model_cumulative_plots = []
    
    for model in sorted(recycle_data.keys()):
        cumulative_rmsd1 = []
        cumulative_rmsd2 = []
        
        for recycle in range(max_recycles + 1):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    cumulative_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                    if data.rmsd_ref2 is not None:
                        cumulative_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
    
        if cumulative_rmsd1:
            model_cumulative_plots.append((
                np.array(cumulative_rmsd1),
                np.array(cumulative_rmsd2) if cumulative_rmsd2 else None,
                f"Model {model}\nAll Recycles"
            ))
    
    if model_cumulative_plots:
        fig = create_collage(
            model_cumulative_plots,
            f"Position {position} - Cumulative Model Comparison",
            ncols=3
        )
        figures['model_cumulative_comparison'] = fig
        
        if output_dir:
            fig.savefig(comparison_dir / f"pos_{position}_model_cumulative_comparison.{format}",
                       format=format, bbox_inches='tight', dpi=300)
            if not show:
                plt.close(fig)
    
    # 3. Model performance metrics
    fig = plt.figure(figsize=(15, 5), dpi=300)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # RMSD1 boxplot
    ax1 = plt.subplot(gs[0])
    rmsd1_data = []
    labels = []
    
    for model in sorted(recycle_data.keys()):
        model_rmsd1 = []
        for recycle in range(max_recycles + 1):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    model_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
        if model_rmsd1:
            rmsd1_data.append(model_rmsd1)
            labels.append(f"Model {model}")
    
    if rmsd1_data:
        ax1.boxplot(rmsd1_data, labels=labels)
        ax1.set_ylabel('RMSD vs Reference 1 (Å)', fontsize=10)
        ax1.set_title('RMSD1 Distribution by Model', fontsize=12, pad=10)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.tick_params(axis='both', labelsize=8)
    
    # RMSD2 boxplot
    ax2 = plt.subplot(gs[1])
    rmsd2_data = []
    
    for model in sorted(recycle_data.keys()):
        model_rmsd2 = []
        for recycle in range(max_recycles + 1):
            if recycle in recycle_data[model]:
                data = recycle_data[model][recycle]
                if data.rmsd_ref2 is not None:
                    model_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        if model_rmsd2:
            rmsd2_data.append(model_rmsd2)
    
    if rmsd2_data:
        ax2.boxplot(rmsd2_data, labels=labels)
        ax2.set_ylabel('RMSD vs Reference 2 (Å)', fontsize=10)
        ax2.set_title('RMSD2 Distribution by Model', fontsize=12, pad=10)
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.tick_params(axis='both', labelsize=8)
    
    # pLDDT boxplot
    ax3 = plt.subplot(gs[2])
    plddt_data = []
    
    for model in sorted(recycle_data.keys()):
        model_plddt = []
        for recycle in range(max_recycles + 1):
            if recycle in recycle_data[model]:
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
                    model_plddt.extend(plddt_array.flatten())
        if model_plddt:
            plddt_data.append(model_plddt)
    
    if plddt_data:
        ax3.boxplot(plddt_data, labels=labels)
        ax3.set_ylabel('pLDDT', fontsize=10)
        ax3.set_title('pLDDT Distribution by Model', fontsize=12, pad=10)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.tick_params(axis='both', labelsize=8)
    
    plt.suptitle(f"Position {position} - Model Performance Comparison",
                 fontsize=14, y=1.02, fontweight='bold')
    
    plt.tight_layout()
    figures['model_performance'] = fig
    
    if output_dir:
        fig.savefig(comparison_dir / f"pos_{position}_model_performance.{format}",
                    format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return figures 