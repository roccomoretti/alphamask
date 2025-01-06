"""
Violin plot functionality for RMSD analysis.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from ..rmsd import RMSDResult
from ..storage import Storage

logger = logging.getLogger(__name__)

def create_violin_plots(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create publication-ready violin plots of RMSD distributions.
    
    Args:
        storage: Storage instance to use for data access
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    # Get all recycle data
    recycle_data = storage.get_all_recycle_data(position)
    if not recycle_data:
        logger.warning(f"No data found for position {position}")
        return figures
    
    # Extract data for both references
    rmsd_ref1_data = []
    rmsd_ref2_data = []
    labels = []
    
    # Process each model and recycle
    for model in sorted(recycle_data.keys()):
        for recycle in sorted(recycle_data[model].keys()):
            data = recycle_data[model][recycle]
            if data.rmsd_ref1 is not None:
                rmsd_ref1_data.append(data.rmsd_ref1.flatten())
                if data.rmsd_ref2 is not None:
                    rmsd_ref2_data.append(data.rmsd_ref2.flatten())
                labels.append(f"M{model}R{recycle}")
    
    if not rmsd_ref1_data:
        logger.warning(f"No RMSD data found for position {position}")
        return figures
    
    # Calculate figure dimensions
    n_positions = len(labels)
    width_per_position = 0.8
    min_width = 12
    fig_width = max(min_width, n_positions * width_per_position)
    
    # Create figure with calculated size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 14),
                                  dpi=300,
                                  sharex=True,
                                  gridspec_kw={'height_ratios': [1, 1],
                                             'hspace': 0.2})
    
    # Calculate global y-axis range
    all_values = rmsd_ref1_data + rmsd_ref2_data if rmsd_ref2_data else rmsd_ref1_data
    max_rmsd = max(max(max(values) for values in all_values), 5.0)
    
    # Plot Reference 1 (top)
    parts1 = ax1.violinplot(
        rmsd_ref1_data,
        showmeans=True,
        showmedians=True,
        showextrema=True,
        widths=0.8
    )
    
    # Plot Reference 2 (bottom) if available
    if rmsd_ref2_data:
        parts2 = ax2.violinplot(
            rmsd_ref2_data,
            showmeans=True,
            showmedians=True,
            showextrema=True,
            widths=0.8
        )
    
    # Style violins for both plots
    for parts in [parts1, parts2] if rmsd_ref2_data else [parts1]:
        for pc in parts['bodies']:
            pc.set_facecolor('#4477AA')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.0)
        
        # Customize statistics markers
        parts['cmeans'].set_color('white')
        parts['cmeans'].set_linewidth(2.5)
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2.0)
        
        # Customize whiskers
        parts['cbars'].set_color('black')
        parts['cbars'].set_linewidth(1.5)
        parts['cmins'].set_color('black')
        parts['cmins'].set_linewidth(1.5)
        parts['cmaxes'].set_color('black')
        parts['cmaxes'].set_linewidth(1.5)
    
    # Customize axes
    for ax, ref_num in [(ax1, 1), (ax2, 2)]:
        ax.set_title(f'Reference {ref_num}', fontsize=26, pad=20, fontweight='bold')
        if ax == ax2:
            ax.set_xlabel('Model-Recycle', fontsize=34, labelpad=12)
        ax.set_ylabel('RMSD (Å)', fontsize=34, labelpad=12)
        
        # Set ticks
        ax.set_xticks(range(1, len(labels) + 1))
        if ax == ax2:
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=22)
        else:
            ax.set_xticklabels([])
        
        # Customize tick parameters
        ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8)
        ax.tick_params(axis='both', which='minor', width=1, length=4)
        
        # Set y-axis range and ticks
        ax.set_ylim(0, max_rmsd)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray', linewidth=1.0,
                     which='major')
        ax.yaxis.grid(True, linestyle=':', alpha=0.3, color='gray', linewidth=0.5,
                     which='minor')
        ax.set_axisbelow(True)
    
    # Set main title
    fig.suptitle(f"Position {position} - RMSD Distributions", fontsize=18, y=0.95, fontweight='bold')
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.12)
    
    figures['violin'] = fig
    
    if output_dir:
        fig.savefig(output_dir / f"pos_{position}_violin.{format}",
                   format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return figures 