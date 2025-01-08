"""
Scatter plot functionality for RMSD analysis.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
from ..storage import Storage

logger = logging.getLogger(__name__)

def get_plddt_colors(plddt_values: np.ndarray) -> np.ndarray:
    """
    Convert pLDDT values to discrete colors using AlphaFold2 color scheme.
    
    Args:
        plddt_values: Array of pLDDT values
        
    Returns:
        Array of colors for each pLDDT value
    """
    # Define AF2 colors for each range
    af_colors = {
        'very_low': '#FF7D45',  # pLDDT < 50
        'low': '#FFF300',       # 50 <= pLDDT < 70
        'confident': '#00A1D3', # 70 <= pLDDT < 90
        'very_high': '#0053D6'  # pLDDT >= 90
    }
    
    # Convert to RGB arrays
    colors = np.zeros((len(plddt_values), 3))
    
    # Assign colors based on pLDDT ranges
    colors[plddt_values < 50] = plt.matplotlib.colors.to_rgb(af_colors['very_low'])
    colors[(plddt_values >= 50) & (plddt_values < 70)] = plt.matplotlib.colors.to_rgb(af_colors['low'])
    colors[(plddt_values >= 70) & (plddt_values < 90)] = plt.matplotlib.colors.to_rgb(af_colors['confident'])
    colors[plddt_values >= 90] = plt.matplotlib.colors.to_rgb(af_colors['very_high'])
    
    return colors

def create_scatter_plot(
    storage: Storage,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None,
    system_name: Optional[str] = None
) -> plt.Figure:
    """
    Create a publication-ready scatter plot combining all predictions from all positions.
    Points are colored by pLDDT values using the AlphaFold2 color scheme.
    
    Args:
        storage: H5 storage instance
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        system_name: Optional name of the system for the title
        
    Returns:
        Generated figure
    """
    # Create figure with publication-friendly size
    fig = plt.figure(figsize=(6, 6), dpi=300)
    
    # Create grid for plots with better spacing
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    
    # Create subplots
    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)
    
    # Collect all RMSD and pLDDT data
    all_rmsd1 = []
    all_rmsd2 = []
    all_plddt = []
    
    # Get all positions
    positions = storage.get_positions()
    
    # Collect data from all positions and models
    for position in positions:
        recycle_data = storage.get_all_recycle_data(position)
        for model_data in recycle_data.values():
            for data in model_data.values():
                if data.rmsd_ref1 is not None:
                    all_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                    if data.rmsd_ref2 is not None:
                        all_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
                    if data.plddt is not None:
                        all_plddt.extend(np.array(data.plddt).flatten())
    
    if not all_rmsd1:
        logger.warning("No RMSD data found")
        return None
    
    # Convert to numpy arrays
    all_rmsd1 = np.array(all_rmsd1)
    all_rmsd2 = np.array(all_rmsd2) if all_rmsd2 else all_rmsd1
    all_plddt = np.array(all_plddt)
    
    # Calculate global max RMSD if not provided
    if max_rmsd is None:
        global_max_rmsd = max(max(all_rmsd1), max(all_rmsd2)) * 1.1
    else:
        global_max_rmsd = max_rmsd
    
    # Get discrete colors for pLDDT values
    colors = get_plddt_colors(all_plddt)
    
    # Create main scatter plot
    scatter = ax_main.scatter(
        all_rmsd1, all_rmsd2,
        c=colors,
        s=15,
        alpha=0.5,
        edgecolors='none'
    )
    
    # Add diagonal line with improved visibility
    ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                 alpha=0.9, linewidth=2.0, dashes=(5, 5))
    
    # Customize main plot
    ax_main.set_xlabel('RMSD vs State 1 (Å)', fontsize=10, labelpad=8)
    ax_main.set_ylabel('RMSD vs State 2 (Å)', fontsize=10, labelpad=8)
    ax_main.tick_params(axis='both', which='major', labelsize=8)
    ax_main.tick_params(axis='both', which='minor', labelsize=6)
    ax_main.grid(True, linestyle='--', alpha=0.2, which='major')
    ax_main.minorticks_on()
    
    # Create top histogram with neutral color and better bins
    counts_top, edges_top, _ = ax_top.hist(all_rmsd1, bins=50,
                                         range=(0, global_max_rmsd),
                                         color='#808080',
                                         alpha=0.6,
                                         edgecolor='black', linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax_top.tick_params(axis='y', labelrotation=45)
    ax_top.grid(True, linestyle='--', alpha=0.2)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    
    # Create right histogram with matching style
    counts_right, edges_right, _ = ax_right.hist(all_rmsd2, bins=50,
                                               range=(0, global_max_rmsd),
                                               orientation='horizontal',
                                               color='#808080',
                                               alpha=0.6,
                                               edgecolor='black', linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax_right.tick_params(axis='x', labelrotation=45)
    ax_right.grid(True, linestyle='--', alpha=0.2)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    
    # Create custom colorbar with improved positioning and width
    af_colors = ['#FF7D45', '#FFF300', '#00A1D3', '#0053D6']
    bounds = [0, 50, 70, 90, 100]
    cmap = ListedColormap(af_colors)
    
    # Create a dummy scatter plot for the colorbar
    norm = plt.Normalize(0, 100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add colorbar with better positioning
    ax_cbar = plt.subplot(gs[0, 1])
    divider = make_axes_locatable(ax_cbar)
    cbar = plt.colorbar(sm, cax=ax_cbar, orientation='vertical',
                       boundaries=bounds, ticks=[25, 60, 80, 95])
    
    # Customize colorbar appearance with better spacing and alignment
    cbar.ax.tick_params(labelsize=8, pad=5)
    cbar.set_label('pLDDT', fontsize=9, labelpad=10, rotation=90, y=0.5)
    cbar.set_ticklabels(['Very low\n(<50)', 'Low\n(50-70)', 
                        'Confident\n(70-90)', 'Very high\n(>90)'])
    
    # Adjust colorbar position and width
    pos = ax_cbar.get_position()
    ax_cbar.set_position([
        pos.x0 + pos.width * 0.2,
        pos.y0 + pos.height * 0.1,
        pos.width * 0.3,
        pos.height * 0.8
    ])
    
    # Set aspect ratio and limits
    ax_main.set_aspect('equal')
    ax_top.set_ylim(0, max(counts_top) * 1.1)
    ax_right.set_xlim(0, max(counts_right) * 1.1)
    ax_main.set_xlim(0, global_max_rmsd)
    ax_main.set_ylim(0, global_max_rmsd)
    
    # Add title with system name if provided
    if system_name:
        plt.suptitle(f"{system_name} Scatter Plot",
                     fontsize=12, y=0.95, fontweight='bold')
    
    # Adjust layout with better spacing
    plt.subplots_adjust(
        left=0.15,    # Increased left margin
        right=0.9,    # Decreased right margin
        top=0.9,      # Decreased top margin
        bottom=0.15,  # Increased bottom margin
        wspace=0.05,  # Minimal spacing between plots
        hspace=0.05   # Minimal spacing between plots
    )
    
    if output_dir:
        output_path = output_dir / f"scatter_plot.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return fig 

def create_combined_scatter_plot(
    storage: Storage,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None,
    system_name: Optional[str] = None
) -> plt.Figure:
    """
    Create a publication-ready scatter plot combining all predictions from all positions.
    Points are colored by pLDDT values using a continuous color scale.
    
    Args:
        storage: H5 storage instance
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        system_name: Optional name of the system for the title
        
    Returns:
        Generated figure
    """
    # Scale factor for higher resolution
    scale = 2
    
    # Create figure with publication-friendly size (scaled up proportionally)
    fig = plt.figure(figsize=(6*scale, 6*scale), dpi=300)  # Original size * scale
    
    # Adjust font sizes for the scaled figure
    SMALL_SIZE = 8 * scale
    MEDIUM_SIZE = 10 * scale
    BIGGER_SIZE = 12 * scale
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # Create grid for plots with better spacing
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    
    # Create subplots
    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)
    
    # Collect all RMSD and pLDDT data
    all_rmsd1 = []
    all_rmsd2 = []
    all_plddt = []
    
    # Get all positions
    positions = storage.get_positions()
    
    # Collect data from all positions and models
    for position in positions:
        recycle_data = storage.get_all_recycle_data(position)
        for model_data in recycle_data.values():
            for data in model_data.values():
                if data.rmsd_ref1 is not None:
                    rmsd1 = np.array(data.rmsd_ref1).flatten()
                    all_rmsd1.extend(rmsd1)
                    if data.rmsd_ref2 is not None:
                        rmsd2 = np.array(data.rmsd_ref2).flatten()
                        all_rmsd2.extend(rmsd2)
                        if data.plddt is not None:
                            # Average pLDDT over the region for each prediction
                            plddt = np.array(data.plddt)
                            avg_plddt = np.mean(plddt, axis=1)  # Average over residues
                            all_plddt.extend(avg_plddt)
    
    if not all_rmsd1:
        logger.warning("No RMSD data found")
        return None
    
    # Convert to numpy arrays
    all_rmsd1 = np.array(all_rmsd1)
    all_rmsd2 = np.array(all_rmsd2) if all_rmsd2 else all_rmsd1
    all_plddt = np.array(all_plddt)
    
    # Calculate global max RMSD if not provided
    if max_rmsd is None:
        global_max_rmsd = max(max(all_rmsd1), max(all_rmsd2)) * 1.1
    else:
        global_max_rmsd = max_rmsd
    
    # Sort points by pLDDT values (ascending order so high values are plotted last)
    sort_idx = np.argsort(all_plddt)
    all_rmsd1 = all_rmsd1[sort_idx]
    all_rmsd2 = all_rmsd2[sort_idx]
    all_plddt = all_plddt[sort_idx]
    
    # Get pLDDT range for colorbar
    plddt_min = np.min(all_plddt)
    plddt_max = np.max(all_plddt)
    
    # Create main scatter plot with continuous color scale and scaled point size
    scatter = ax_main.scatter(
        all_rmsd1, all_rmsd2,
        c=all_plddt,
        cmap='viridis',
        s=3*scale,  # Scale the point size
        alpha=0.7,
        edgecolors='none',
        vmin=plddt_min,  # Set color scale to actual data range
        vmax=plddt_max
    )
    
    # Add diagonal line with improved visibility and scaled width
    ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                 alpha=0.9, linewidth=2.0*scale, dashes=(5*scale, 5*scale))
    
    # Customize main plot with scaled sizes
    ax_main.set_xlabel('RMSD vs State 1 (Å)', fontsize=10*scale, labelpad=8*scale)
    ax_main.set_ylabel('RMSD vs State 2 (Å)', fontsize=10*scale, labelpad=8*scale)
    ax_main.tick_params(axis='both', which='major', labelsize=8*scale)
    ax_main.tick_params(axis='both', which='minor', labelsize=6*scale)
    ax_main.grid(True, linestyle='--', alpha=0.2, which='major', linewidth=0.5*scale)
    ax_main.minorticks_on()
    
    # Create top histogram with neutral color and better bins
    counts_top, edges_top, _ = ax_top.hist(all_rmsd1, bins=50,
                                         range=(0, global_max_rmsd),
                                         color='#808080',
                                         alpha=0.6,
                                         edgecolor='black', linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8*scale)
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax_top.tick_params(axis='y', labelrotation=45)
    ax_top.grid(True, linestyle='--', alpha=0.2)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    
    # Create right histogram with matching style
    counts_right, edges_right, _ = ax_right.hist(all_rmsd2, bins=50,
                                               range=(0, global_max_rmsd),
                                               orientation='horizontal',
                                               color='#808080',
                                               alpha=0.6,
                                               edgecolor='black', linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8*scale)
    ax_right.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
    ax_right.tick_params(axis='x', labelrotation=45)
    ax_right.grid(True, linestyle='--', alpha=0.2)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    
    # Add colorbar with continuous scale using data range
    ax_cbar = plt.subplot(gs[0, 1])
    divider = make_axes_locatable(ax_cbar)
    cbar = plt.colorbar(scatter, cax=ax_cbar, orientation='vertical',
                       ticks=np.linspace(plddt_min, plddt_max, 5))  # 5 evenly spaced ticks
    
    # Customize colorbar appearance with scaled sizes
    cbar.ax.tick_params(labelsize=8*scale, pad=5*scale)
    cbar.set_label('pLDDT', fontsize=9*scale, labelpad=10*scale, rotation=90, y=0.5)
    
    # Adjust colorbar position and width
    pos = ax_cbar.get_position()
    ax_cbar.set_position([
        pos.x0 + pos.width * 0.2,
        pos.y0 + pos.height * 0.1,
        pos.width * 0.3,
        pos.height * 0.8
    ])
    
    # Set aspect ratio and limits
    ax_main.set_aspect('equal')
    ax_top.set_ylim(0, max(counts_top) * 1.1)
    ax_right.set_xlim(0, max(counts_right) * 1.1)
    ax_main.set_xlim(0, global_max_rmsd)
    ax_main.set_ylim(0, global_max_rmsd)
    
    # Add title with system name if provided
    if system_name:
        plt.suptitle(f"{system_name} Scatter Plot",
                     fontsize=12*scale, y=0.95, fontweight='bold')
    
    # Adjust layout with better spacing
    plt.subplots_adjust(
        left=0.15,    # Increased left margin
        right=0.9,    # Decreased right margin
        top=0.9,      # Decreased top margin
        bottom=0.15,  # Increased bottom margin
        wspace=0.05,  # Minimal spacing between plots
        hspace=0.05   # Minimal spacing between plots
    )
    
    # Save with high DPI
    if output_dir:
        output_path = output_dir / f"combined_scatter_plot.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300*scale)
        if not show:
            plt.close(fig)
    
    return fig 