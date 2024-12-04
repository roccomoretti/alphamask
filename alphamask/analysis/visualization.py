"""
Visualization functionality for RMSD analysis results.
"""
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import logging
from dataclasses import dataclass
from .rmsd import RMSDResult

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    plot_width: int = 15
    plot_height: int = 10
    colorscale: List[Tuple[float, Tuple[float, float, float]]] = None
    
    def __post_init__(self):
        if self.colorscale is None:
            # Default colorscale (Cesar's colorscale)
            self.colorscale = [
                (0.0, (1.0, 1.0, 1.0)),
                (1/12, (0.416, 0.133, 0.996)),
                (3/12, (0.059, 0.651, 0.937)),
                (5/12, (0.314, 0.957, 0.800)),
                (7/12, (0.686, 0.957, 0.592)),
                (9/12, (1.0, 0.655, 0.349)),
                (11/12, (1.0, 0.133, 0.067)),
                (1.0, (1.0, 0.0, 0.0))
            ]

class RMSDVisualizer:
    """
    Creates visualizations for RMSD analysis results.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.cmap = LinearSegmentedColormap.from_list('cesar', self.config.colorscale)
        
    def plot_violin_distributions(
        self,
        results_dict: Dict[str, List[RMSDResult]],
        control_results: Optional[List[RMSDResult]] = None,
        title: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create violin plots for RMSD distributions.
        
        Args:
            results_dict: Dictionary mapping structure names to RMSD results
            control_results: Optional control structure results
            title: Optional plot title
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        data = []
        
        if control_results:
            for result in control_results:
                data.append({
                    'Structure': 'Control',
                    'RMSD_Ref1': result.rmsd_ref1,
                    'RMSD_Ref2': result.rmsd_ref2
                })
                
        for structure, results in results_dict.items():
            for result in results:
                data.append({
                    'Structure': structure,
                    'RMSD_Ref1': result.rmsd_ref1,
                    'RMSD_Ref2': result.rmsd_ref2
                })
                
        df = pd.DataFrame(data)
        
        # Create plot
        fig, axes = plt.subplots(1, 2 if 'RMSD_Ref2' in df.columns else 1,
                                figsize=(self.config.plot_width, self.config.plot_height))
        if 'RMSD_Ref2' not in df.columns:
            axes = [axes]
            
        # Plot Reference 1
        sns.violinplot(data=df, x='Structure', y='RMSD_Ref1', ax=axes[0])
        axes[0].set_title('RMSD Distribution vs Reference 1')
        axes[0].set_ylabel('RMSD (Å)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot Reference 2 if available
        if 'RMSD_Ref2' in df.columns:
            sns.violinplot(data=df, x='Structure', y='RMSD_Ref2', ax=axes[1])
            axes[1].set_title('RMSD Distribution vs Reference 2')
            axes[1].set_ylabel('RMSD (Å)')
            axes[1].tick_params(axis='x', rotation=45)
            
        if title:
            fig.suptitle(title)
            
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig

    def plot_rmsd_landscape(
        self,
        rmsd_ref1: np.ndarray,
        rmsd_ref2: Optional[np.ndarray] = None,
        title: str = "",
        show: bool = True
    ) -> plt.Figure:
        """
        Create 2D histogram landscape plot for RMSD values.
        
        Args:
            rmsd_ref1: RMSD values for reference 1
            rmsd_ref2: Optional RMSD values for reference 2
            title: Plot title
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if rmsd_ref2 is None:
            rmsd_ref2 = rmsd_ref1
            
        fig = plt.figure(figsize=(self.config.plot_width, self.config.plot_height))
        gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                             hspace=0.2, wspace=0.2)
                             
        ax_main = plt.subplot(gs[1, 0])
        ax_histx = plt.subplot(gs[0, 0], sharex=ax_main)
        ax_histy = plt.subplot(gs[1, 1], sharey=ax_main)
        
        # Get common range
        max_rmsd = max(max(rmsd_ref1), max(rmsd_ref2))
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(
            rmsd_ref1, rmsd_ref2,
            bins=50,
            range=[[0, max_rmsd], [0, max_rmsd]]
        )
        
        # Plot main heatmap
        vmin = 0.1
        vmax = H.max()
        im = ax_main.pcolormesh(xedges, yedges, H.T,
                               norm=LogNorm(vmin=vmin, vmax=vmax),
                               cmap=self.cmap)
                               
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax_histy)
        cbar.set_label('log(counts)', labelpad=10)
        
        # Create 1D histograms
        ax_histx.hist(rmsd_ref1, bins=80, range=(0, max_rmsd),
                     density=True, alpha=1, color='black')
        ax_histy.hist(rmsd_ref2, bins=80, range=(0, max_rmsd),
                     density=True, orientation='horizontal',
                     alpha=1, color='black')
                     
        # Customize appearance
        ax_histx.spines['right'].set_visible(False)
        ax_histx.spines['top'].set_visible(False)
        ax_histx.spines['bottom'].set_visible(False)
        ax_histx.tick_params(axis="x", labelbottom=False)
        
        ax_histy.spines['top'].set_visible(False)
        ax_histy.spines['right'].set_visible(False)
        ax_histy.spines['left'].set_visible(False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        ax_main.set_xlabel(r"RMSD vs Reference 1 (Å)", labelpad=10)
        ax_main.set_ylabel(r"RMSD vs Reference 2 (Å)", labelpad=10)
        
        if title:
            plt.suptitle(title, fontweight='bold', y=0.95)
            
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig

    def create_interactive_plot(
        self,
        results_dict: Dict[str, List[RMSDResult]],
        control_results: Optional[List[RMSDResult]] = None
    ) -> go.Figure:
        """
        Create interactive Plotly scatter plots of RMSDs colored by pLDDT.
        
        Args:
            results_dict: Dictionary mapping structure names to RMSD results
            control_results: Optional control structure results
            
        Returns:
            Plotly figure
        """
        # Calculate layout
        n_plots = len(results_dict) + (1 if control_results else 0)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=['Control' if control_results else ''] + list(results_dict.keys()),
            shared_xaxes=True,
            shared_yaxes=True
        )
        
        # Define marker symbols
        marker_symbols = ['circle', 'diamond', 'square', 'triangle-up', 'star',
                         'pentagon', 'hexagon', 'cross', 'x', 'triangle-down']
                         
        # Plot data
        plot_idx = 0
        row = 1
        col = 1
        
        if control_results:
            self._add_scatter_trace(
                fig, control_results, 'Control',
                marker_symbols[plot_idx], row, col
            )
            plot_idx += 1
            col = col + 1 if col < n_cols else 1
            row = row + 1 if col == 1 else row
            
        for structure_name, results in results_dict.items():
            self._add_scatter_trace(
                fig, results, structure_name,
                marker_symbols[plot_idx % len(marker_symbols)],
                row, col
            )
            plot_idx += 1
            col = col + 1 if col < n_cols else 1
            row = row + 1 if col == 1 else row
            
        # Update layout
        fig.update_layout(
            title_text="RMSD Comparison (colored by pLDDT)",
            showlegend=True,
            height=300*n_rows,
            width=600*n_cols + 100,
            template="plotly_white",
            margin=dict(r=120)
        )
        
        return fig
        
    def _add_scatter_trace(
        self,
        fig: go.Figure,
        results: List[RMSDResult],
        name: str,
        marker_symbol: str,
        row: int,
        col: int
    ) -> None:
        """Helper method to add scatter trace to Plotly figure."""
        plddt_values = [r.plddt for r in results if r.plddt is not None]
        rmsd1_values = [r.rmsd_ref1 for r in results if r.plddt is not None]
        rmsd2_values = [r.rmsd_ref2 for r in results if r.plddt is not None and r.rmsd_ref2 is not None]
        
        fig.add_trace(
            go.Scatter(
                x=rmsd1_values,
                y=rmsd2_values if rmsd2_values else rmsd1_values,
                mode='markers',
                marker=dict(
                    size=8,
                    symbol=marker_symbol,
                    color=plddt_values,
                    colorscale='Viridis',
                    cmin=50,
                    cmax=100,
                    showscale=True,
                    colorbar=dict(
                        title='pLDDT',
                        x=1.02,
                        xanchor='left',
                        yanchor='middle'
                    )
                ),
                text=[f"Model: {r.model_name}<br>pLDDT: {r.plddt:.2f}"
                      for r in results if r.plddt is not None],
                hoverinfo='text',
                name=f'{name} ({marker_symbol})'
            ),
            row=row,
            col=col
        ) 