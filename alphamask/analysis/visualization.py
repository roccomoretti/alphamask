"""
Visualization functionality for RMSD analysis results.
"""
from typing import List, Dict, Optional, Union, Any, Tuple
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
from .statistics import RMSDStatistics
import traceback
from matplotlib import colors

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    plot_width: int = 15
    plot_height: int = 10
    colorscale: List[Tuple[float, Tuple[float, float, float]]] = None
    font_size: int = 12
    dpi: int = 300
    style: str = "default"
    interactive: bool = True
    show_statistics: bool = True
    show_significance: bool = True
    
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
        """Initialize visualizer with configuration."""
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
    
    def create_plots(
        self,
        results: Dict[str, Any],
        output_dir: Path,
        format: str = "pdf"
    ) -> None:
        """Create visualization plots."""
        try:
            if not results:
                logger.warning("No results to plot")
                return
                
            # Create violin plot
            self._create_violin_plot(results, output_dir / f"rmsd_violin.{format}")
            
            # Create other plots as needed...
            
        except Exception as e:
            logger.error(f"Failed to create plots: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
    
    def _create_violin_plot(self, results: Dict[str, Any], output_path: Path) -> None:
        """Create violin plot of RMSD distributions."""
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            data = []
            for structure, stats in results.items():
                if isinstance(stats, dict) and 'rmsd_values' in stats:
                    for rmsd in stats['rmsd_values']:
                        data.append({
                            'Structure': structure,
                            'RMSD': rmsd
                        })
            
            if not data:
                logger.warning("No RMSD data available for plotting")
                return
                
            # Create plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=data, x='Structure', y='RMSD')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create violin plot: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
    
    def create_violin_plots(
        self,
        results_dict: Dict[str, List[RMSDResult]],
        title: str = "RMSD Distributions"
    ) -> plt.Figure:
        """Create publication-ready violin plots of RMSD distributions."""
        # Extract data for both references
        rmsd_ref1_data = []
        rmsd_ref2_data = []
        labels = []
        
        # Sort by position number
        for name, results in sorted(results_dict.items(), key=lambda x: int(x[0].split('_')[2])):
            rmsd_ref1_values = [r.rmsd_ref1 for r in results]
            rmsd_ref2_values = [r.rmsd_ref2 for r in results if r.rmsd_ref2 is not None]
            
            if rmsd_ref1_values:
                rmsd_ref1_data.append(rmsd_ref1_values)
                if rmsd_ref2_values:
                    rmsd_ref2_data.append(rmsd_ref2_values)
                
                # Clean up position labels
                pos_num = name.split('_')[2]
                labels.append(f"{pos_num}")  # Simplified labels

        # Calculate figure dimensions
        n_positions = len(labels)
        width_per_position = 0.8
        min_width = 12
        fig_width = max(min_width, n_positions * width_per_position)
        
        # Create figure with calculated size - increased height for better proportions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, 14),  # Increased height to 14
                                      dpi=300,
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [1, 1],
                                                 'hspace': 0.2})  # Increased spacing
        
        # Calculate global y-axis range
        all_values = rmsd_ref1_data + rmsd_ref2_data
        max_rmsd = max(max(max(values) for values in all_values), 5.0)
        
        # Plot Reference 1 (top)
        parts1 = ax1.violinplot(
            rmsd_ref1_data,
            showmeans=True,
            showmedians=True,
            showextrema=True,
            widths=0.8  # Wider violins
        )
        
        # Plot Reference 2 (bottom)
        if rmsd_ref2_data:
            parts2 = ax2.violinplot(
                rmsd_ref2_data,
                showmeans=True,
                showmedians=True,
                showextrema=True,
                widths=0.8  # Wider violins
            )
        
        # Style violins for both plots
        for parts in [parts1, parts2] if rmsd_ref2_data else [parts1]:
            for pc in parts['bodies']:
                pc.set_facecolor('#4477AA')
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.0)  # Thicker lines
            
            # Customize statistics markers
            parts['cmeans'].set_color('white')
            parts['cmeans'].set_linewidth(2.5)  # Thicker lines
            parts['cmedians'].set_color('red')
            parts['cmedians'].set_linewidth(2.0)  # Thicker lines
            
            # Customize whiskers
            parts['cbars'].set_color('black')
            parts['cbars'].set_linewidth(1.5)  # Thicker lines
            parts['cmins'].set_color('black')
            parts['cmins'].set_linewidth(1.5)  # Thicker lines
            parts['cmaxes'].set_color('black')
            parts['cmaxes'].set_linewidth(1.5)  # Thicker lines
        
        # Customize axes with larger fonts
        for ax, ref_num in [(ax1, 1), (ax2, 2)]:
            ax.set_title(f'Reference {ref_num}', fontsize=26, pad=20, fontweight='bold')  # Increased from 14
            if ax == ax2:
                ax.set_xlabel('Position', fontsize=34, labelpad=12)  # Increased from 12
            ax.set_ylabel('RMSD (Å)', fontsize=34, labelpad=12)  # Increased from 12
            
            # Set ticks with larger fonts
            ax.set_xticks(range(1, len(labels) + 1))
            if ax == ax2:
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=22)  # Increased from 10
            else:
                ax.set_xticklabels([])
            
            # Customize tick parameters with larger sizes
            ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=8)  # Increased sizes
            ax.tick_params(axis='both', which='minor', width=1, length=4)
            
            # Set y-axis range and ticks
            ax.set_ylim(0, max_rmsd)
            ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))  # Major ticks every 1.0
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor ticks every 0.5
            
            # Add grid with better visibility
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray', linewidth=1.0,
                         which='major')
            ax.yaxis.grid(True, linestyle=':', alpha=0.3, color='gray', linewidth=0.5,
                         which='minor')
            ax.set_axisbelow(True)
        
        # Set main title with larger font
        fig.suptitle(title, fontsize=18, y=0.95, fontweight='bold')  # Increased from 16
        
        # Adjust layout with better margins for larger labels
        plt.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.12)  # Adjusted margins
        
        return fig
    
    def create_interactive_violin(
        self,
        results: Dict[str, Any]
    ) -> go.Figure:
        """Create interactive violin plots using plotly."""
        df = pd.DataFrame(results['detailed'])
        
        fig = make_subplots(
            rows=1,
            cols=2 if 'RMSD_Ref2' in df.columns else 1,
            subplot_titles=['RMSD vs Reference 1', 'RMSD vs Reference 2'] if 'RMSD_Ref2' in df.columns else ['RMSD vs Reference 1']
        )
        
        # Add violin plots for Reference 1
        for structure in df['Structure'].unique():
            struct_data = df[df['Structure'] == structure]
            fig.add_trace(
                go.Violin(
                    x=[structure] * len(struct_data),
                    y=struct_data['RMSD_Ref1'],
                    name=structure,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    customdata=struct_data[['Model', 'pLDDT']],
                    hovertemplate=(
                        "Structure: %{x}<br>"
                        "RMSD: %{y:.2f} Å<br>"
                        "Model: %{customdata[0]}<br>"
                        "pLDDT: %{customdata[1]:.2f}<br>"
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )
        
        # Add violin plots for Reference 2 if available
        if 'RMSD_Ref2' in df.columns:
            for structure in df['Structure'].unique():
                struct_data = df[df['Structure'] == structure]
                fig.add_trace(
                    go.Violin(
                        x=[structure] * len(struct_data),
                        y=struct_data['RMSD_Ref2'],
                        name=structure,
                        box_visible=True,
                        meanline_visible=True,
                        points="all",
                        customdata=struct_data[['Model', 'pLDDT']],
                        hovertemplate=(
                            "Structure: %{x}<br>"
                            "RMSD: %{y:.2f} Å<br>"
                            "Model: %{customdata[0]}<br>"
                            "pLDDT: %{customdata[1]:.2f}<br>"
                            "<extra></extra>"
                        )
                    ),
                    row=1, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="RMSD Distributions",
            showlegend=False,
            height=600,
            width=1200 if 'RMSD_Ref2' in df.columns else 800
        )
        
        fig.update_yaxes(title_text="RMSD (Å)")
        
        return fig
    
    def create_rmsd_landscape(
        self,
        rmsd_ref1: List[float],
        rmsd_ref2: Optional[List[float]] = None,
        title: Optional[str] = None,
        max_rmsd: Optional[float] = None,
        show: bool = False
    ) -> plt.Figure:
        """Create publication-ready RMSD landscape plot with 2D histogram."""
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
    
    def create_interactive_plot(
        self,
        results_dict: Dict[str, List[RMSDResult]],
        control_results: Optional[List[RMSDResult]] = None,
        max_rmsd: Optional[float] = None
    ) -> List[go.Figure]:
        """Create interactive Plotly scatter plots of RMSDs colored by pLDDT."""
        # Sort results by position number
        sorted_results = dict(sorted(results_dict.items(), 
                                   key=lambda x: int(x[0].split('_')[2])))  # Sort by position number
        
        # Calculate number of plots needed
        n_plots = len(sorted_results)
        n_cols = 3  # Changed to 3x3 grid
        n_rows = 3
        n_batches = (n_plots + 8) // 9  # Ceiling division for 3x3 grid
        
        # If max_rmsd not provided, calculate it
        if max_rmsd is None:
            max_rmsd = 0
            if control_results:
                max_rmsd = max(
                    max(r.rmsd_ref1 for r in control_results),
                    max(r.rmsd_ref2 for r in control_results if r.rmsd_ref2 is not None)
                )
            
            for results in sorted_results.values():
                max_rmsd = max(
                    max_rmsd,
                    max(r.rmsd_ref1 for r in results),
                    max(r.rmsd_ref2 for r in results if r.rmsd_ref2 is not None)
                )
            max_rmsd *= 1.1
        
        figures = []
        for batch in range(n_batches):
            start_idx = batch * 9
            end_idx = min((batch + 1) * 9, n_plots)
            batch_results = dict(list(sorted_results.items())[start_idx:end_idx])
            
            # Create subplot for this batch
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[f"Position {name.split('_')[2]}" for name in batch_results.keys()],
                shared_xaxes=True,
                shared_yaxes=True,
                horizontal_spacing=0.12,  # Increased spacing
                vertical_spacing=0.12
            )
            
            # Plot data
            plot_idx = 0
            for name, results in batch_results.items():
                row = (plot_idx // n_cols) + 1
                col = (plot_idx % n_cols) + 1
                
                self._add_scatter_trace(
                    fig, results, name,
                    'circle',  # Use consistent marker
                    row, col, max_rmsd,
                    n_points=len(results),
                    n_cols=n_cols  # Pass n_cols parameter
                )
                
                # Add diagonal line
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_rmsd],
                        y=[0, max_rmsd],
                        mode='lines',
                        line=dict(color='black', dash='dash', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row,
                    col=col
                )
                
                plot_idx += 1
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"RMSD Comparison - Positions {start_idx+1}-{end_idx} (Batch {batch+1}/{n_batches})",
                    y=0.95  # Moved down slightly
                ),
                showlegend=True,
                height=1200,
                width=1400,  # Increased width to accommodate colorbar
                template="plotly_white",
                margin=dict(r=150, t=100, b=60, l=60),  # Increased right margin for colorbar
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",  # Changed to right align
                    x=0.95,  # Moved left slightly
                    font=dict(size=10),
                    bordercolor="Black",
                    borderwidth=1,
                    itemsizing='constant'  # Make legend items consistent size
                )
            )
            
            # Update axes
            for i in range(1, n_rows + 1):
                for j in range(1, n_cols + 1):
                    # Update X axes - only show title for bottom row
                    fig.update_xaxes(
                        title_text="RMSD vs Reference 1 (Å)" if i == n_rows else None,
                        range=[0, max_rmsd],
                        zeroline=True,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        row=i,
                        col=j
                    )
                    
                    # Update Y axes - only show title for leftmost column
                    fig.update_yaxes(
                        title_text="RMSD vs Reference 2 (Å)" if j == 1 else None,
                        range=[0, max_rmsd],
                        zeroline=True,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray',
                        row=i,
                        col=j
                    )
            
            figures.append(fig)
        
        return figures
    
    def _add_statistics_annotations(
        self,
        ax: plt.Axes,
        stats: List[RMSDStatistics],
        reference: str
    ) -> None:
        """Add statistical annotations to plot."""
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
    
    def plot_multiple_landscapes(
        self,
        results_dict: Dict[str, List[RMSDResult]],
        output_dir: Optional[Path] = None,
        format: str = "pdf",
        control_results: Optional[List[RMSDResult]] = None,
        show: bool = False,
        overwrite: bool = False
    ) -> Dict[str, Union[plt.Figure, List[go.Figure]]]:
        """Create both aggregated and individual RMSD landscape plots."""
        output_dir = Path(output_dir) if output_dir else None
        figures = {}
        
        # Calculate global max RMSD across all results
        max_rmsd = 0
        for results in results_dict.values():
            max_rmsd = max(
                max_rmsd,
                max(r.rmsd_ref1 for r in results),
                max(r.rmsd_ref2 for r in results if r.rmsd_ref2 is not None)
            )
        max_rmsd *= 1.1
        
        # Create and save plots one at a time
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create aggregated plot if needed
        agg_plot_path = output_dir / f"rmsd_landscape_aggregated.{format}" if output_dir else None
        if not output_dir or overwrite or not agg_plot_path.exists():
            all_results = []
            for results in results_dict.values():
                all_results.extend(results)
            
            fig_agg = self.create_rmsd_landscape(
                rmsd_ref1=[r.rmsd_ref1 for r in all_results],
                rmsd_ref2=[r.rmsd_ref2 for r in all_results if r.rmsd_ref2 is not None],
                title="Aggregated RMSD Landscape",
                max_rmsd=max_rmsd,
                show=False
            )
            
            if output_dir:
                fig_agg.savefig(agg_plot_path, 
                               format=format, bbox_inches='tight', dpi=300)
                plt.close(fig_agg)
            else:
                figures['aggregated'] = fig_agg
        else:
            logger.debug("Aggregated plot already exists, skipping.")
        
        # Create individual plots
        for name, results in results_dict.items():
            ind_plot_path = output_dir / f"rmsd_landscape_{name}.{format}" if output_dir else None
            if not output_dir or overwrite or not ind_plot_path.exists():
                fig_ind = self.create_rmsd_landscape(
                    rmsd_ref1=[r.rmsd_ref1 for r in results],
                    rmsd_ref2=[r.rmsd_ref2 for r in results if r.rmsd_ref2 is not None],
                    title=f"RMSD Landscape - {name}",
                    max_rmsd=max_rmsd,
                    show=False
                )
                
                if output_dir:
                    fig_ind.savefig(ind_plot_path, 
                                  format=format, bbox_inches='tight', dpi=300)
                    plt.close(fig_ind)
                else:
                    figures[name] = fig_ind
            else:
                logger.debug(f"Plot for {name} already exists, skipping.")
        
        # Create interactive plots
        interactive_figures = self.create_interactive_plot(
            results_dict, 
            control_results,
            max_rmsd=max_rmsd
        )
        
        if output_dir:
            # Save interactive plots
            for i, fig in enumerate(interactive_figures):
                output_path = output_dir / f"rmsd_interactive_batch_{i+1}.html"
                if overwrite or not output_path.exists():
                    fig.write_html(output_path)
                    logger.debug(f"Saved interactive plot batch {i+1}")
                    # if format is pdf or png save
                    fig.write_image(output_path.with_suffix(f".{format}"))
                else:
                    logger.debug(f"Interactive plot batch {i+1} already exists, skipping.")
        else:
            figures['interactive'] = interactive_figures
        
        # Create violin plot
        violin_path = output_dir / f"rmsd_violin.{format}" if output_dir else None
        if not output_dir or overwrite or not violin_path.exists():
            fig_violin = self.create_violin_plots(
                results_dict,
                title="RMSD Distributions"
            )
            
            if output_dir:
                fig_violin.savefig(violin_path, 
                                 format=format, bbox_inches='tight', dpi=300)
                plt.close(fig_violin)
            else:
                figures['violin'] = fig_violin
        else:
            logger.debug("Violin plot already exists, skipping.")
        
        return figures 
    
    def _add_scatter_trace(
        self,
        fig: go.Figure,
        results: List[RMSDResult],
        name: str,
        marker_symbol: str,
        row: int,
        col: int,
        max_rmsd: float,
        n_points: int = 100,
        n_cols: int = 3  # Add n_cols parameter with default value
    ) -> None:
        """Helper method to add scatter trace to Plotly figure."""
        plddt_values = [r.plddt for r in results if r.plddt is not None]
        rmsd1_values = [r.rmsd_ref1 for r in results if r.plddt is not None]
        rmsd2_values = [r.rmsd_ref2 for r in results if r.plddt is not None and r.rmsd_ref2 is not None]
        
        # Calculate marker size based on number of points
        marker_size = max(4, min(8, 12 - 0.02 * n_points))  # Dynamic sizing
        
        # Clean up the name for legend - extract position number
        position = name.split('_')[2]
        clean_name = f"Position {position}"
        
        # Calculate mean pLDDT for hover info
        mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else 0
        
        # AlphaFold pLDDT color scheme
        af_colorscale = [
            [0.0, '#FF7D45'],  # Orange for very low
            [0.5, '#FFF300'],  # Yellow for low
            [0.7, '#00A1D3'],  # Light blue for confident
            [0.9, '#0053D6'],  # Dark blue for very high
            [1.0, '#0053D6']   # Extend dark blue to end
        ]
        
        fig.add_trace(
            go.Scatter(
                x=rmsd1_values,
                y=rmsd2_values if rmsd2_values else rmsd1_values,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    symbol=marker_symbol,
                    color=plddt_values,
                    colorscale=af_colorscale,
                    cmin=50,
                    cmax=100,
                    showscale=col == n_cols,  # Only show colorbar for rightmost column
                    colorbar=dict(
                        title=dict(
                            text='pLDDT',
                            font=dict(size=12)
                        ),
                        x=1.05,  # Moved right
                        xanchor='left',
                        yanchor='middle',
                        len=0.5,
                        thickness=15,
                        ticktext=['Very low (<50)', 'Low (50-70)', 
                                 'Confident (70-90)', 'Very high (>90)'],
                        tickvals=[45, 60, 80, 95],
                        tickmode='array',
                        tickfont=dict(size=10)  # Smaller font for tick labels
                    ),
                    opacity=0.7
                ),
                text=[
                    f"Position: {position}<br>"
                    f"Model: {r.model_name}<br>"
                    f"RMSD Ref1: {r.rmsd_ref1:.2f} Å<br>"
                    f"RMSD Ref2: {r.rmsd_ref2:.2f} Å<br>"
                    f"pLDDT: {r.plddt:.1f}"
                    for r in results if r.plddt is not None
                ],
                hoverinfo='text',
                name=f"{clean_name} (mean pLDDT: {mean_plddt:.1f})",
                showlegend=col == 1  # Only show legend for first column
            ),
            row=row,
            col=col
        ) 