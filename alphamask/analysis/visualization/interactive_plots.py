"""
Interactive plotting functionality using Plotly.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from ..rmsd import RMSDResult
from ..storage import Storage

logger = logging.getLogger(__name__)

def create_interactive_violin(results: Dict[str, Any]) -> go.Figure:
    """
    Create interactive violin plots using plotly.
    
    Args:
        results: Dictionary containing RMSD analysis results
        
    Returns:
        Plotly figure object
    """
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

def create_interactive_plot(
    storage: Storage,
    position: int,
    output_dir: Optional[Path] = None,
    format: str = "pdf",
    show: bool = False
) -> List[go.Figure]:
    """
    Create interactive Plotly scatter plots of RMSDs colored by pLDDT.
    
    Args:
        storage: H5 storage instance
        position: Position number to analyze
        output_dir: Directory to save plots
        format: Output format (pdf, png)
        show: Whether to display plots
        
    Returns:
        List of Plotly figure objects
    """
    figures = []
    
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
    
    # Create scatter plot
    for model in sorted(recycle_data.keys()):
        model_data = recycle_data[model]
        
        # Extract and flatten arrays
        plddt_values = []
        rmsd1_values = []
        rmsd2_values = []
        hover_texts = []
        
        for recycle, data in sorted(model_data.items()):
            if data.plddt is not None:
                plddt_array = np.array(data.plddt).flatten()
                rmsd1_array = np.array(data.rmsd_ref1).flatten()
                
                plddt_values.extend(plddt_array)
                rmsd1_values.extend(rmsd1_array)
                
                if data.rmsd_ref2 is not None:
                    rmsd2_array = np.array(data.rmsd_ref2).flatten()
                    rmsd2_values.extend(rmsd2_array)
                
                # Add hover text for each point
                hover_texts.extend([
                    f"Position: {position}<br>"
                    f"Model: {model}<br>"
                    f"Recycle: {recycle}<br>"
                    f"RMSD Ref1: {rmsd1:.2f} Å<br>"
                    f"RMSD Ref2: {rmsd2:.2f} Å<br>"
                    f"pLDDT: {plddt:.1f}"
                    for rmsd1, rmsd2, plddt in zip(
                        rmsd1_array,
                        np.array(data.rmsd_ref2).flatten() if data.rmsd_ref2 is not None else rmsd1_array,
                        plddt_array
                    )
                ])
        
        if not plddt_values:
            continue
        
        # Calculate marker size based on number of points
        marker_size = max(4, min(8, 12 - 0.02 * len(plddt_values)))  # Dynamic sizing
        
        # Calculate mean pLDDT for hover info
        mean_plddt = np.mean(plddt_values) if plddt_values else 0
        
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
                    symbol='circle',
                    color=plddt_values,
                    colorscale=af_colorscale,
                    cmin=50,
                    cmax=100,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text='pLDDT',
                            font=dict(size=12)
                        ),
                        x=1.05,
                        xanchor='left',
                        yanchor='middle',
                        len=0.5,
                        thickness=15,
                        ticktext=['Very low (<50)', 'Low (50-70)', 
                                 'Confident (70-90)', 'Very high (>90)'],
                        tickvals=[45, 60, 80, 95],
                        tickmode='array',
                        tickfont=dict(size=10)
                    ),
                    opacity=0.7
                ),
                text=hover_texts,
                hoverinfo='text',
                name=f"Model {model} (mean pLDDT: {mean_plddt:.1f})",
                showlegend=True
            )
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Position {position} - RMSD Comparison",
            y=0.95
        ),
        showlegend=True,
        height=800,
        width=1000,
        template="plotly_white",
        xaxis=dict(
            title="RMSD vs Reference 1 (Å)",
            range=[0, max(max(trace.x) for trace in fig.data) * 1.1],
            zeroline=True,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="RMSD vs Reference 2 (Å)" if has_rmsd2 else "RMSD vs Reference 1 (Å)",
            range=[0, max(max(trace.y) for trace in fig.data) * 1.1],
            zeroline=True,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.95,
            font=dict(size=10),
            bordercolor="Black",
            borderwidth=1,
            itemsizing='constant'
        )
    )
    
    # Add diagonal line
    max_rmsd = max(
        max(max(trace.x) for trace in fig.data),
        max(max(trace.y) for trace in fig.data)
    ) * 1.1
    
    fig.add_trace(
        go.Scatter(
            x=[0, max_rmsd],
            y=[0, max_rmsd],
            mode='lines',
            line=dict(color='black', dash='dash', width=1),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    
    figures.append(fig)
    
    if output_dir:
        output_path = output_dir / f"pos_{position}_interactive.{format}"
        fig.write_image(output_path, format=format)
        if not show:
            plt.close(fig)
    
    return figures 