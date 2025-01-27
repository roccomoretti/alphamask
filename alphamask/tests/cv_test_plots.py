"""Plotting functions for collective variables analysis."""
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import re
from alphamask.analysis.collective_variables import CVCalculator, CVConfig, CVResult
from pathlib import Path

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
    
    # First collect all models and recycles
    all_models = set()
    all_recycles = set()
    
    # Extract model and recycle info from all results first
    for results in all_results.values():
        for result in results:
            if result.model_name:
                match = re.search(r'model_(\d+).*_r(\d+)_', result.model_name)
                if match:
                    all_models.add(int(match.group(1)))
                    all_recycles.add(int(match.group(2)))
    
    print(f"Found {len(all_models)} models and {len(all_recycles)} recycles")
    print(f"Models: {sorted(all_models)}")
    print(f"Recycles: {sorted(all_recycles)}")
    
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

def plot_cv_summary_comparison(all_results: Dict[str, List[CVResult]], output_dir: Path):
    """Create a 2x2 summary plot comparing all conditions using recycles 0-2."""
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
    
    # Create cesar colormap
    cesar_cmap = create_cesar_colormap()
    
    # Find global maximum count for consistent coloring
    max_count = 0
    for results in all_results.values():
        # Filter for recycles 0-2
        filtered_results = []
        for r in results:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match and int(match.group(1)) <= 2:
                    filtered_results.append(r)
        
        cv1_data = [r.cv1_dihedral for r in filtered_results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in filtered_results if r.cv2_diff is not None]
        if cv1_data and cv2_diff_data:
            counts, _, _ = np.histogram2d(cv1_data, cv2_diff_data, bins=50,
                                        range=[x_range, y_range])
            max_count = max(max_count, counts.max())
    
    # Plot each condition
    for idx, (condition, results) in enumerate(all_results.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Filter for recycles 0-2
        filtered_results = []
        for r in results:
            if r.model_name:
                match = re.search(r'_r(\d+)_', r.model_name)
                if match and int(match.group(1)) <= 2:
                    filtered_results.append(r)
        
        # Extract data
        cv1_data = [r.cv1_dihedral for r in filtered_results if r.cv1_dihedral is not None]
        cv2_diff_data = [r.cv2_diff for r in filtered_results if r.cv2_diff is not None]
        
        # Create 2D histogram
        hist = ax.hist2d(cv1_data, cv2_diff_data, bins=50,
                        range=[x_range, y_range],
                        cmap=cesar_cmap,
                        norm=LogNorm(vmin=1, vmax=max_count))
        
        # Add colorbar
        plt.colorbar(hist[3], ax=ax, label='Count')
        
        # Customize plot
        ax.set_xlabel('CV1 (radians)')
        ax.set_ylabel('CV2 (d2-d1)')
        ax.set_title(conditions[condition], fontsize=12, pad=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add summary statistics
        stats_text = f'n = {len(cv1_data)}\n'
        stats_text += f'CV1 mean = {np.mean(cv1_data):.2f}°\n'
        stats_text += f'CV2 mean = {np.mean(cv2_diff_data):.2f} Å'
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add overall title
    plt.suptitle('Comparison of CV Distributions (Recycles 0-2)', 
                 fontsize=16, y=0.95, fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_summary_comparison_r0_2.pdf', dpi=700, bbox_inches='tight')
    plt.close()
