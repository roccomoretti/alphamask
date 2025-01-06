"""
Landscape plot functionality for RMSD analysis.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def create_rmsd_landscape(
    rmsd_ref1: List[float],
    rmsd_ref2: Optional[List[float]] = None,
    title: Optional[str] = None,
    max_rmsd: Optional[float] = None,
    show: bool = False
) -> plt.Figure:
    """
    Create publication-ready RMSD landscape plot with 2D histogram.
    
    Args:
        rmsd_ref1: List of RMSD values for reference 1
        rmsd_ref2: Optional list of RMSD values for reference 2
        title: Optional title for the plot
        max_rmsd: Optional maximum RMSD value for plot scaling
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Use rmsd_ref1 for both axes if rmsd_ref2 not provided
    y_data = rmsd_ref2 if rmsd_ref2 is not None else rmsd_ref1
    
    # Calculate max RMSD if not provided
    if max_rmsd is None:
        max_rmsd = max(max(rmsd_ref1), max(y_data)) * 1.1
    
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
    
    # Create figure with publication-friendly size
    fig = plt.figure(figsize=(6, 6), dpi=300)
    
    # Create grid for plots with better spacing
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.02, wspace=0.02)
    
    # Create main 2D histogram
    ax_main = plt.subplot(gs[1, 0])
    hist2d = ax_main.hist2d(rmsd_ref1, y_data, bins=40,
                           range=[[0, max_rmsd], [0, max_rmsd]],
                           cmap=cesar_cmap,
                           norm=LogNorm())
    
    # Add diagonal line with improved visibility
    ax_main.plot([0, max_rmsd], [0, max_rmsd], '--', color='black',
                 alpha=0.8, linewidth=1.5, dashes=(5, 5))
    
    # Customize main plot
    ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
    ax_main.set_ylabel('RMSD vs Reference 2 (Å)', fontsize=10, labelpad=8)
    ax_main.tick_params(axis='both', which='major', labelsize=8)
    ax_main.tick_params(axis='both', which='minor', labelsize=6)
    ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
    ax_main.minorticks_on()
    
    # Create top histogram with matching color
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    counts, edges, _ = ax_top.hist(rmsd_ref1, bins=40, range=(0, max_rmsd),
                                 color=cesar_colors[2], alpha=0.7,
                                 edgecolor='black', linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    
    # Create right histogram with matching color
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(y_data, bins=40, range=(0, max_rmsd),
                 orientation='horizontal', color=cesar_colors[2],
                 alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    # Add colorbar with better positioning
    cax = plt.subplot(gs[0, 1])
    cbar = plt.colorbar(hist2d[3], cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=7)  # Smaller font size
    cbar.ax.set_title('Density', fontsize=8, pad=3)
    
    # Format colorbar ticks to avoid overlap
    cbar.formatter = plt.ScalarFormatter(useMathText=True)  # Use scientific notation
    cbar.formatter.set_scientific(True)
    cbar.update_ticks()
    
    # Rotate tick labels for better fit
    cbar.ax.xaxis.set_tick_params(rotation=45)
    
    # Adjust colorbar position
    cax.set_position([cax.get_position().x0, 
                     cax.get_position().y0 + 0.05,  # Move up slightly
                     cax.get_position().width,
                     cax.get_position().height * 0.8])  # Reduce height
    
    # Set title if provided with better formatting
    if title:
        if 'WT_pos_' in title:
            position = title.split('_')[2]
            title = f"RMSD Landscape - Position {position}"
        fig.suptitle(title, fontsize=12, y=0.95, fontweight='bold')
    
    # Ensure square aspect ratio for main plot
    ax_main.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig 