"""
Landscape plot functionality for RMSD analysis.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import logging
from ..storage import Storage
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
                          hspace=0.05, wspace=0.05)  # Slightly increased spacing
    
    # Create subplots
    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)
    
    # Create main 2D histogram
    hist2d = ax_main.hist2d(rmsd_ref1, y_data, bins=40,
                           range=[[0, max_rmsd], [0, max_rmsd]],
                           cmap=cesar_cmap,
                           norm=LogNorm(vmin=1))  # Set minimum value to 1 to avoid log(0)
    
    # Add diagonal line with improved visibility
    ax_main.plot([0, max_rmsd], [0, max_rmsd], '--', color='black',
                 alpha=0.8, linewidth=1.5, dashes=(5, 5))
    
    # Customize main plot
    ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
    ax_main.set_ylabel('RMSD vs Reference 2 (Å)' if rmsd_ref2 is not None else 'RMSD vs Reference 1 (Å)',
                      fontsize=10, labelpad=8)
    ax_main.tick_params(axis='both', which='major', labelsize=8)
    ax_main.tick_params(axis='both', which='minor', labelsize=6)
    ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
    ax_main.minorticks_on()
    
    # Create top histogram with matching color
    counts_top, edges_top, _ = ax_top.hist(rmsd_ref1, bins=40, range=(0, max_rmsd),
                                         color=cesar_colors[2], alpha=0.7,
                                         edgecolor='black', linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.set_title("")  # Remove this title since we'll use suptitle
    
    # Create right histogram with matching color
    counts_right, edges_right, _ = ax_right.hist(y_data, bins=40, range=(0, max_rmsd),
                                               orientation='horizontal', color=cesar_colors[2],
                                               alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    
    # Add colorbar with better positioning
    ax_cbar = plt.subplot(gs[0, 1])
    divider = make_axes_locatable(ax_cbar)
    cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='vertical',
                       format=LogFormatterSciNotation(base=10, labelOnlyBase=True))
    
    # Customize colorbar appearance
    cbar.ax.tick_params(labelsize=7, pad=2)
    cbar.set_label('Density', fontsize=8, labelpad=2, rotation=90, y=0.5, x=-0.5)
    
    # Set specific tick locations to avoid overlap
    cbar.set_ticks([1, 10, 100])  # Explicitly set tick locations
    cbar.set_ticklabels(['$10^0$', '$10^1$', '$10^2$'])  # Use LaTeX formatting
    
    # Make colorbar more compact and adjust position
    pos = ax_cbar.get_position()
    ax_cbar.set_position([
        pos.x0 + pos.width * 0.4,  # Move more to the right
        pos.y0 + pos.height * 0.1,  # Move up slightly
        pos.width * 0.15,  # Make width even smaller
        pos.height * 0.8  # Make height smaller
    ])
    
    # Ensure square aspect ratio for main plot
    ax_main.set_aspect('equal')
    
    # Set the same limits for top and right histograms
    ax_top.set_ylim(0, max(counts_top) * 1.1)  # Add 10% padding
    ax_right.set_xlim(0, max(counts_right) * 1.1)  # Add 10% padding
    
    # Remove main plot spines between histograms
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # Set x and y limits for main plot
    ax_main.set_xlim(0, max_rmsd)
    ax_main.set_ylim(0, max_rmsd)
    
    # Set title if provided with better formatting
    if title:
        if 'WT_pos_' in title:
            position = title.split('_')[2]
            title = f"Position {position}"
        plt.suptitle(title, fontsize=12, y=0.95, fontweight='bold')
    
    # Adjust layout with better spacing
    plt.subplots_adjust(
        left=0.15,    # Increased left margin
        right=0.9,    # Decreased right margin
        top=0.9,      # Decreased top margin
        bottom=0.15,  # Increased bottom margin
        wspace=0.05,  # Minimal spacing between plots
        hspace=0.05   # Minimal spacing between plots
    )
    
    if show:
        plt.show()
    
    return fig 

def create_combined_landscape(
    storage: Storage,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> plt.Figure:
    """
    Create a single landscape plot combining all predictions from all positions.
    
    Args:
        storage: H5 storage instance
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Generated figure
    """
    # Create figure with publication-friendly size
    fig = plt.figure(figsize=(6, 6), dpi=300)
    
    # Create grid for plots with better spacing
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)  # Slightly increased spacing
    
    # Create subplots
    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)
    
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
    
    # Collect all RMSD data
    all_rmsd1 = []
    all_rmsd2 = []
    
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
    
    if not all_rmsd1:
        logger.warning("No RMSD data found")
        return None
    
    # Convert to numpy arrays
    all_rmsd1 = np.array(all_rmsd1)
    all_rmsd2 = np.array(all_rmsd2) if all_rmsd2 else all_rmsd1
    
    # Calculate global max RMSD if not provided
    if max_rmsd is None:
        global_max_rmsd = max(max(all_rmsd1), max(all_rmsd2)) * 1.1
    else:
        global_max_rmsd = max_rmsd
    
    # Create main 2D histogram
    hist2d = ax_main.hist2d(all_rmsd1, all_rmsd2, bins=40,
                           range=[[0, global_max_rmsd], [0, global_max_rmsd]],
                           cmap=cesar_cmap,
                           norm=LogNorm(vmin=1))  # Set minimum value to 1 to avoid log(0)
    
    # Add diagonal line with improved visibility
    ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                 alpha=0.8, linewidth=1.5, dashes=(5, 5))
    
    # Customize main plot
    ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
    ax_main.set_ylabel('RMSD vs Reference 2 (Å)', fontsize=10, labelpad=8)
    ax_main.tick_params(axis='both', which='major', labelsize=8)
    ax_main.tick_params(axis='both', which='minor', labelsize=6)
    ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
    ax_main.minorticks_on()
    
    # Create top histogram with matching color
    counts_top, edges_top, _ = ax_top.hist(all_rmsd1, bins=40, range=(0, global_max_rmsd),
                                         color=cesar_colors[2], alpha=0.7,
                                         edgecolor='black', linewidth=0.5)
    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.grid(True, linestyle='--', alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.set_title("")  # Remove this title since we'll use suptitle
    
    # Create right histogram with matching color
    counts_right, edges_right, _ = ax_right.hist(all_rmsd2, bins=40, range=(0, global_max_rmsd),
                                               orientation='horizontal', color=cesar_colors[2],
                                               alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelsize=8)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    
    # Add colorbar with better positioning
    ax_cbar = plt.subplot(gs[0, 1])
    divider = make_axes_locatable(ax_cbar)
    cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='vertical',
                       format=LogFormatterSciNotation(base=10, labelOnlyBase=True))
    
    # Customize colorbar appearance
    cbar.ax.tick_params(labelsize=7, pad=2)
    cbar.set_label('Density', fontsize=8, labelpad=2, rotation=90, y=0.5, x=-0.5)
    
    # Set specific tick locations to avoid overlap
    cbar.set_ticks([1, 10, 100])  # Explicitly set tick locations
    cbar.set_ticklabels(['$10^0$', '$10^1$', '$10^2$'])  # Use LaTeX formatting
    
    # Make colorbar more compact and adjust position
    pos = ax_cbar.get_position()
    ax_cbar.set_position([
        pos.x0 + pos.width * 0.4,  # Move more to the right
        pos.y0 + pos.height * 0.1,  # Move up slightly
        pos.width * 0.15,  # Make width even smaller
        pos.height * 0.8  # Make height smaller
    ])
    
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
    
    # Add single title with adjusted position
    plt.suptitle("Combined RMSD Landscape",
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
        output_path = output_dir / f"combined_landscape.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return fig 

def create_combined_landscape_breakdown(
    storage: Storage,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> Dict[str, plt.Figure]:
    """
    Create breakdown plots of combined RMSD landscapes by model and recycle.
    
    Args:
        storage: H5 storage instance
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Dictionary of generated figures
    """
    figures = {}
    
    # Get all positions and models
    positions = storage.get_positions()
    all_models = set()
    all_recycles = set()
    
    # Collect all RMSD data and find max values
    all_rmsd1 = []
    all_rmsd2 = []
    
    # First pass to get all models and recycles
    for position in positions:
        recycle_data = storage.get_all_recycle_data(position)
        for model in recycle_data.keys():
            all_models.add(model)
            for recycle in recycle_data[model].keys():
                all_recycles.add(recycle)
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    all_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                    if data.rmsd_ref2 is not None:
                        all_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
    
    # Calculate global max RMSD if not provided
    if max_rmsd is None:
        global_max_rmsd = max(max(all_rmsd1), max(all_rmsd2) if all_rmsd2 else max(all_rmsd1)) * 1.1
    else:
        global_max_rmsd = max_rmsd
    
    # 1. Create per-model plots
    logger.info(f"Creating per-model plots for {len(all_models)} models")
    for model in sorted(all_models):
        model_rmsd1 = []
        model_rmsd2 = []
        
        for position in positions:
            recycle_data = storage.get_all_recycle_data(position)
            if model in recycle_data:
                for data in recycle_data[model].values():
                    if data.rmsd_ref1 is not None:
                        model_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            model_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if model_rmsd1:
            fig = create_rmsd_landscape(
                rmsd_ref1=np.array(model_rmsd1),
                rmsd_ref2=np.array(model_rmsd2) if model_rmsd2 else None,
                title=f"All Positions - Model {model}",
                max_rmsd=global_max_rmsd,
                show=False
            )
            figures[f'model_{model}'] = fig
            
            if output_dir:
                model_dir = output_dir / "model_breakdown"
                model_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(model_dir / f"model_{model}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    # 2. Create per-recycle plots
    logger.info(f"Creating per-recycle plots for {len(all_recycles)} recycles")
    for recycle in sorted(all_recycles):
        recycle_rmsd1 = []
        recycle_rmsd2 = []
        
        for position in positions:
            recycle_data = storage.get_all_recycle_data(position)
            for model_data in recycle_data.values():
                if recycle in model_data:
                    data = model_data[recycle]
                    if data.rmsd_ref1 is not None:
                        recycle_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            recycle_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if recycle_rmsd1:
            fig = create_rmsd_landscape(
                rmsd_ref1=np.array(recycle_rmsd1),
                rmsd_ref2=np.array(recycle_rmsd2) if recycle_rmsd2 else None,
                title=f"All Positions - Recycle {recycle}",
                max_rmsd=global_max_rmsd,
                show=False
            )
            figures[f'recycle_{recycle}'] = fig
            
            if output_dir:
                recycle_dir = output_dir / "recycle_breakdown"
                recycle_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(recycle_dir / f"recycle_{recycle}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    # 3. Create cumulative recycle plots (0-N)
    logger.info(f"Creating cumulative recycle plots for {len(all_recycles)} recycles")
    cumulative_0n_rmsd1 = []
    cumulative_0n_rmsd2 = []
    
    for recycle in sorted(all_recycles):
        for position in positions:
            recycle_data = storage.get_all_recycle_data(position)
            for model_data in recycle_data.values():
                if recycle in model_data:
                    data = model_data[recycle]
                    if data.rmsd_ref1 is not None:
                        cumulative_0n_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            cumulative_0n_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if cumulative_0n_rmsd1:
            fig = create_rmsd_landscape(
                rmsd_ref1=np.array(cumulative_0n_rmsd1),
                rmsd_ref2=np.array(cumulative_0n_rmsd2) if cumulative_0n_rmsd2 else None,
                title=f"All Positions - Cumulative Recycles 0-{recycle}",
                max_rmsd=global_max_rmsd,
                show=False
            )
            figures[f'cumulative_0_{recycle}'] = fig
            
            if output_dir:
                cumulative_dir = output_dir / "cumulative_breakdown"
                cumulative_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(cumulative_dir / f"cumulative_0_{recycle}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    # 4. Create cumulative recycle plots (1-N)
    logger.info(f"Creating cumulative recycle plots for {len(all_recycles)} recycles")
    cumulative_1n_rmsd1 = []
    cumulative_1n_rmsd2 = []
    
    for recycle in sorted(all_recycles):
        if recycle == 0:  # Skip recycle 0 for 1-N plots
            continue
            
        for position in positions:
            recycle_data = storage.get_all_recycle_data(position)
            for model_data in recycle_data.values():
                if recycle in model_data:
                    data = model_data[recycle]
                    if data.rmsd_ref1 is not None:
                        cumulative_1n_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            cumulative_1n_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if cumulative_1n_rmsd1:
            fig = create_rmsd_landscape(
                rmsd_ref1=np.array(cumulative_1n_rmsd1),
                rmsd_ref2=np.array(cumulative_1n_rmsd2) if cumulative_1n_rmsd2 else None,
                title=f"All Positions - Cumulative Recycles 1-{recycle}",
                max_rmsd=global_max_rmsd,
                show=False
            )
            figures[f'cumulative_1_{recycle}'] = fig
            
            if output_dir:
                cumulative_dir = output_dir / "cumulative_breakdown"
                cumulative_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(cumulative_dir / f"cumulative_1_{recycle}.{format}",
                           format=format, bbox_inches='tight', dpi=300)
                if not show:
                    plt.close(fig)
    
    return figures 

def apriori_create_summary_landscape(
    storages: Dict[str, Storage],
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> plt.Figure:
    """
    Create a summary plot comparing combined RMSD landscapes across different conditions.
    
    Args:
        storages: Dictionary mapping condition names to their Storage objects
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
    Returns:
        Generated figure
    """
    # Create figure with publication-friendly size
    fig = plt.figure(figsize=(12, 12), dpi=300)
    
    # Create 2x2 grid
    gs = gridspec.GridSpec(2, 2, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)
    
    # Calculate global max RMSD across all conditions
    global_max_rmsd = max_rmsd
    all_data = {}
    
    # First pass: collect all data and find global max
    for condition, storage in storages.items():
        all_rmsd1 = []
        all_rmsd2 = []
        
        positions = storage.get_positions()
        for position in positions:
            recycle_data = storage.get_all_recycle_data(position)
            for model_data in recycle_data.values():
                for data in model_data.values():
                    if data.rmsd_ref1 is not None:
                        all_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            all_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if all_rmsd1:
            all_data[condition] = {
                'rmsd1': np.array(all_rmsd1),
                'rmsd2': np.array(all_rmsd2) if all_rmsd2 else np.array(all_rmsd1)
            }
            global_max_rmsd = max(global_max_rmsd, 
                                max(max(all_data[condition]['rmsd1']),
                                    max(all_data[condition]['rmsd2'])))
    
    global_max_rmsd *= 1.1  # Add 10% padding
    
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
    
    # Second pass: create subplots
    for idx, (condition, data) in enumerate(all_data.items()):
        row = idx // 2
        col = idx % 2
        
        # Create subplot with internal grid for histograms
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[row, col],
                                                width_ratios=[4, 1], height_ratios=[1, 4],
                                                hspace=0.05, wspace=0.05)
        
        # Create subplots
        ax_main = plt.subplot(gs_sub[1, 0])
        ax_top = plt.subplot(gs_sub[0, 0], sharex=ax_main)
        ax_right = plt.subplot(gs_sub[1, 1], sharey=ax_main)
        
        # Create main 2D histogram
        hist2d = ax_main.hist2d(data['rmsd1'], data['rmsd2'], bins=40,
                               range=[[0, global_max_rmsd], [0, global_max_rmsd]],
                               cmap=cesar_cmap,
                               norm=LogNorm(vmin=1))
        
        # Add diagonal line
        ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                     alpha=0.8, linewidth=1.5, dashes=(5, 5))
        
        # Customize main plot
        ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
        ax_main.set_ylabel('RMSD vs Reference 2 (Å)', fontsize=10, labelpad=8)
        ax_main.tick_params(axis='both', which='major', labelsize=8)
        ax_main.tick_params(axis='both', which='minor', labelsize=6)
        ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
        ax_main.minorticks_on()
        
        # Create top histogram
        counts_top, edges_top, _ = ax_top.hist(data['rmsd1'], bins=40,
                                             range=(0, global_max_rmsd),
                                             color=cesar_colors[2], alpha=0.7,
                                             edgecolor='black', linewidth=0.5)
        ax_top.tick_params(axis='x', labelbottom=False)
        ax_top.tick_params(axis='y', labelsize=8)
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        
        # Create right histogram
        counts_right, edges_right, _ = ax_right.hist(data['rmsd2'], bins=40,
                                                   range=(0, global_max_rmsd),
                                                   orientation='horizontal',
                                                   color=cesar_colors[2], alpha=0.7,
                                                   edgecolor='black', linewidth=0.5)
        ax_right.tick_params(axis='y', labelleft=False)
        ax_right.tick_params(axis='x', labelsize=8)
        ax_right.grid(True, linestyle='--', alpha=0.3)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        
        # Add colorbar
        if col == 1:  # Only add colorbar for right-side plots
            ax_cbar = plt.subplot(gs_sub[0, 1])
            divider = make_axes_locatable(ax_cbar)
            cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='vertical',
                              format=LogFormatterSciNotation(base=10, labelOnlyBase=True))
            
            # Customize colorbar
            cbar.ax.tick_params(labelsize=7, pad=2)
            cbar.set_label('Density', fontsize=8, labelpad=2, rotation=90, y=0.5, x=-0.5)
            cbar.set_ticks([1, 10, 100])
            cbar.set_ticklabels(['$10^0$', '$10^1$', '$10^2$'])
            
            # Adjust colorbar position
            pos = ax_cbar.get_position()
            ax_cbar.set_position([
                pos.x0 + pos.width * 0.4,
                pos.y0 + pos.height * 0.1,
                pos.width * 0.15,
                pos.height * 0.8
            ])
        
        # Set aspect ratio and limits
        ax_main.set_aspect('equal')
        ax_top.set_ylim(0, max(counts_top) * 1.1)
        ax_right.set_xlim(0, max(counts_right) * 1.1)
        ax_main.set_xlim(0, global_max_rmsd)
        ax_main.set_ylim(0, global_max_rmsd)

        # Add condition title
        ax_top.set_title(condition.capitalize(), fontsize=12, pad=10, fontweight='bold')
    
    # Add overall title
    plt.suptitle("RMSD Landscape Comparison", fontsize=14, y=0.95)
    
    if output_dir:
        output_path = output_dir / f"summary_landscape.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return fig 

def iterative_create_summary_landscape(
    storages: Dict[str, Storage],
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False,
    max_rmsd: Optional[float] = None
) -> plt.Figure:
    """
    Create a summary plot comparing combined RMSD landscapes across different iterative experiments.
    
    Args:
        storages: Dictionary mapping experiment names to their Storage objects
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        max_rmsd: Optional maximum RMSD value for plot scaling
        
    Returns:
        Generated figure
    """
    # Create figure with publication-friendly size
    fig = plt.figure(figsize=(12, 12), dpi=300)
    
    # Calculate number of rows needed (2 plots per row)
    n_experiments = len(storages)
    n_rows = (n_experiments + 1) // 2  # Round up division
    
    # Create grid
    gs = gridspec.GridSpec(n_rows, 2, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)
    
    # Calculate global max RMSD across all experiments
    global_max_rmsd = max_rmsd if max_rmsd is not None else 0
    all_data = {}
    
    # First pass: collect all data and find global max
    for exp_name, storage in storages.items():
        all_rmsd1 = []
        all_rmsd2 = []
        
        positions = storage.get_positions()
        for position in positions:
            recycle_data = storage.get_all_recycle_data(position)
            for model_data in recycle_data.values():
                for data in model_data.values():
                    if data.rmsd_ref1 is not None:
                        all_rmsd1.extend(np.array(data.rmsd_ref1).flatten())
                        if data.rmsd_ref2 is not None:
                            all_rmsd2.extend(np.array(data.rmsd_ref2).flatten())
        
        if all_rmsd1:
            all_data[exp_name] = {
                'rmsd1': np.array(all_rmsd1),
                'rmsd2': np.array(all_rmsd2) if all_rmsd2 else np.array(all_rmsd1)
            }
            if max_rmsd is None:
                global_max_rmsd = max(global_max_rmsd, 
                                    max(max(all_data[exp_name]['rmsd1']),
                                        max(all_data[exp_name]['rmsd2'])))
    
    if max_rmsd is None:
        global_max_rmsd *= 1.1  # Add 10% padding
    
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
    
    # Second pass: create subplots
    for idx, (exp_name, data) in enumerate(all_data.items()):
        row = idx // 2
        col = idx % 2
        
        # Create subplot with internal grid for histograms
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[row, col],
                                                width_ratios=[4, 1], height_ratios=[1, 4],
                                                hspace=0.05, wspace=0.05)
        
        # Create subplots
        ax_main = plt.subplot(gs_sub[1, 0])
        ax_top = plt.subplot(gs_sub[0, 0], sharex=ax_main)
        ax_right = plt.subplot(gs_sub[1, 1], sharey=ax_main)
        
        # Create main 2D histogram
        hist2d = ax_main.hist2d(data['rmsd1'], data['rmsd2'], bins=40,
                               range=[[0, global_max_rmsd], [0, global_max_rmsd]],
                               cmap=cesar_cmap,
                               norm=LogNorm(vmin=1))
        
        # Add diagonal line
        ax_main.plot([0, global_max_rmsd], [0, global_max_rmsd], '--', color='black',
                     alpha=0.8, linewidth=1.5, dashes=(5, 5))
        
        # Customize main plot
        ax_main.set_xlabel('RMSD vs Reference 1 (Å)', fontsize=10, labelpad=8)
        ax_main.set_ylabel('RMSD vs Reference 2 (Å)', fontsize=10, labelpad=8)
        ax_main.tick_params(axis='both', which='major', labelsize=8)
        ax_main.tick_params(axis='both', which='minor', labelsize=6)
        ax_main.grid(True, linestyle='--', alpha=0.3, which='major')
        ax_main.minorticks_on()
        
        # Create top histogram
        counts_top, edges_top, _ = ax_top.hist(data['rmsd1'], bins=40,
                                             range=(0, global_max_rmsd),
                                             color=cesar_colors[2], alpha=0.7,
                                             edgecolor='black', linewidth=0.5)
        ax_top.tick_params(axis='x', labelbottom=False)
        ax_top.tick_params(axis='y', labelsize=8)
        ax_top.grid(True, linestyle='--', alpha=0.3)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['bottom'].set_visible(False)
        
        # Create right histogram
        counts_right, edges_right, _ = ax_right.hist(data['rmsd2'], bins=40,
                                                   range=(0, global_max_rmsd),
                                                   orientation='horizontal',
                                                   color=cesar_colors[2], alpha=0.7,
                                                   edgecolor='black', linewidth=0.5)
        ax_right.tick_params(axis='y', labelleft=False)
        ax_right.tick_params(axis='x', labelsize=8)
        ax_right.grid(True, linestyle='--', alpha=0.3)
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        
        # Add colorbar
        if col == 1:  # Only add colorbar for right-side plots
            ax_cbar = plt.subplot(gs_sub[0, 1])
            divider = make_axes_locatable(ax_cbar)
            cbar = plt.colorbar(hist2d[3], cax=ax_cbar, orientation='vertical',
                              format=LogFormatterSciNotation(base=10, labelOnlyBase=True))
            
            # Customize colorbar
            cbar.ax.tick_params(labelsize=7, pad=2)
            cbar.set_label('Density', fontsize=8, labelpad=2, rotation=90, y=0.5, x=-0.5)
            cbar.set_ticks([1, 10, 100])
            cbar.set_ticklabels(['$10^0$', '$10^1$', '$10^2$'])
            
            # Adjust colorbar position
            pos = ax_cbar.get_position()
            ax_cbar.set_position([
                pos.x0 + pos.width * 0.4,
                pos.y0 + pos.height * 0.1,
                pos.width * 0.15,
                pos.height * 0.8
            ])
        
        # Set aspect ratio and limits
        ax_main.set_aspect('equal')
        ax_top.set_ylim(0, max(counts_top) * 1.1)
        ax_right.set_xlim(0, max(counts_right) * 1.1)
        ax_main.set_xlim(0, global_max_rmsd)
        ax_main.set_ylim(0, global_max_rmsd)
        
        # Add experiment title
        ax_top.set_title(f"Mutation {exp_name}", fontsize=12, pad=10, fontweight='bold')
    
    # Add overall title
    plt.suptitle("Iterative RMSD Landscape Comparison", fontsize=14, y=0.95)
    
    if output_dir:
        output_path = output_dir / f"iterative_summary_landscape.{format}"
        fig.savefig(output_path, format=format, bbox_inches='tight', dpi=300)
        if not show:
            plt.close(fig)
    
    return fig 