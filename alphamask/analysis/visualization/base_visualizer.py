"""
Base visualization functionality and shared utilities.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.axes import Axes
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
from .plot_config import PlotConfig
from ..statistics import RMSDStatistics

logger = logging.getLogger(__name__)

class BaseVisualizer:
    """
    Common functionality that other visualizers can inherit.
    """
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize base visualizer with configuration.
        
        Args:
            config: Optional plot configuration
        """
        self.config = config or PlotConfig()
        # Create matplotlib colormap from the colorscale
        self.cmap = LinearSegmentedColormap.from_list('cesar', self.config.colorscale)
        # Create plotly colorscale from the same data
        self.plotly_colorscale = [
            [pos, f'rgb({r*255},{g*255},{b*255})']
            for pos, (r,g,b) in self.config.colorscale
        ]
        plt.style.use(self.config.style)
        plt.rcParams.update({'font.size': self.config.font_size})
    
    def _add_statistics_annotations(
        self,
        ax: Axes,
        stats: List[RMSDStatistics],
        reference: str
    ) -> None:
        """
        Add statistical annotations to plot.
        
        Args:
            ax: Matplotlib axes to add annotations to
            stats: List of RMSD statistics
            reference: Reference identifier
        """
        if not self.config.show_statistics:
            return
            
        ymin, ymax = ax.get_ylim()
        text_y = ymax * 1.05
        
        for i, stat in enumerate(stats):
            if stat.reference != reference:
                continue
                
            text = f"μ={stat.mean:.2f}\nσ={stat.std:.2f}"
            if self.config.show_significance and stat.significance_tests:
                if stat.significance_tests.get('shapiro', {}).get('is_normal'):
                    text += "\n(normal)"
                    
            ax.text(
                i, text_y, text,
                horizontalalignment='center',
                verticalalignment='bottom'
            )
    
    def _save_figure(
        self,
        fig: plt.Figure,
        path: Path,
        format: str = "pdf",
        show: bool = False
    ) -> None:
        """
        Helper function to safely save figures.
        
        Args:
            fig: Matplotlib figure to save
            path: Path to save the figure to
            format: Output format (e.g., 'pdf', 'png')
            show: Whether to display the figure
        """
        try:
            fig.savefig(path, format=format, bbox_inches='tight', dpi=self.config.dpi)
            if not show:
                plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to save figure to {path}: {str(e)}")
    
    def _apply_publication_styling(
        self,
        ax: Axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        legend: bool = True
    ) -> None:
        """
        Apply publication-quality styling to an axes.
        
        Args:
            ax: Matplotlib axes to style
            xlabel: Optional x-axis label
            ylabel: Optional y-axis label
            title: Optional title
            legend: Whether to show legend
        """
        # Set labels if provided
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
        if title:
            ax.set_title(title, fontsize=14, pad=10, fontweight='bold')
        
        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=10, width=1.2, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3, which='major')
        ax.grid(True, linestyle=':', alpha=0.2, which='minor')
        ax.set_axisbelow(True)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        # Legend styling
        if legend and ax.get_legend():
            ax.legend(
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                edgecolor='gray',
                fontsize=10
            )
    
    def _create_cesar_colormap(self) -> LinearSegmentedColormap:
        """
        Create Cesar's colormap for consistent use across plots.
        
        Returns:
            Matplotlib colormap
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
    
    def _get_alphafold_colorscale(self) -> List[Tuple[float, str]]:
        """
        Get AlphaFold pLDDT color scheme for consistent use.
        
        Returns:
            List of (position, color) tuples
        """
        return [
            [0.0, '#FF7D45'],  # Orange for very low
            [0.5, '#FFF300'],  # Yellow for low
            [0.7, '#00A1D3'],  # Light blue for confident
            [0.9, '#0053D6'],  # Dark blue for very high
            [1.0, '#0053D6']   # Extend dark blue to end
        ] 