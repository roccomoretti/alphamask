import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run_command(command: Union[str, List[str]], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command safely.
    
    Args:
        command: Command to run (string or list of strings)
        check: Whether to check for errors
        
    Returns:
        CompletedProcess instance
    
    Raises:
        subprocess.CalledProcessError: If command fails and check is True
    """
    try:
        if isinstance(command, str):
            command = command.split()
        result = subprocess.run(
            command,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Error output:\n{e.stderr}")
        if check:
            raise

def create_zip_archive(source_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a zip archive of a directory.
    
    Args:
        source_path: Path to directory to zip
        output_path: Path for output zip file (optional)
        
    Returns:
        Path to created zip file
    """
    source_path = Path(source_path)
    if output_path is None:
        output_path = source_path.with_suffix('.zip')
    else:
        output_path = Path(output_path)
        
    shutil.make_archive(
        str(output_path.with_suffix('')),
        'zip',
        str(source_path)
    )
    return output_path

def plot_msa_comparison(
    original_msa: np.ndarray,
    modified_msa: np.ndarray,
    title: str = "MSA Comparison",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot comparison between original and modified MSA.
    
    Args:
        original_msa: Original MSA array
        modified_msa: Modified MSA array
        title: Plot title
        save_path: Path to save plot
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Original MSA", "Modified MSA")
    )
    
    fig.add_trace(
        go.Heatmap(z=original_msa, colorscale="Viridis"),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=modified_msa, colorscale="Viridis"),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=title,
        height=600,
        showlegend=False
    )
    
    if save_path:
        fig.write_image(str(save_path))
    fig.show()

def plot_plddt_comparison(
    original_plddt: np.ndarray,
    modified_plddt: np.ndarray,
    positions: Optional[List[int]] = None,
    title: str = "pLDDT Comparison",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot comparison between original and modified pLDDT scores.
    
    Args:
        original_plddt: Original pLDDT scores
        modified_plddt: Modified pLDDT scores
        positions: List of positions to highlight
        title: Plot title
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(original_plddt))
    plt.plot(x, original_plddt, label='Original', alpha=0.7)
    plt.plot(x, modified_plddt, label='Modified', alpha=0.7)
    
    if positions:
        for pos in positions:
            plt.axvline(x=pos, color='r', linestyle='--', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Residue Position')
    plt.ylabel('pLDDT Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_changes(
    original_plddt: np.ndarray,
    modified_plddt: np.ndarray,
    positions: List[int],
    window_size: int = 5
) -> Dict[str, Any]:
    """
    Analyze changes in pLDDT scores around modified positions.
    
    Args:
        original_plddt: Original pLDDT scores
        modified_plddt: Modified pLDDT scores
        positions: Modified positions
        window_size: Size of window around positions to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        "global_mean_change": float(np.mean(modified_plddt - original_plddt)),
        "global_std_change": float(np.std(modified_plddt - original_plddt)),
        "position_changes": {},
        "window_changes": {}
    }
    
    for pos in positions:
        # Direct position change
        pos_change = float(modified_plddt[pos] - original_plddt[pos])
        results["position_changes"][pos] = pos_change
        
        # Window analysis
        start = max(0, pos - window_size)
        end = min(len(original_plddt), pos + window_size + 1)
        
        window_orig = original_plddt[start:end]
        window_mod = modified_plddt[start:end]
        
        results["window_changes"][pos] = {
            "mean_change": float(np.mean(window_mod - window_orig)),
            "std_change": float(np.std(window_mod - window_orig)),
            "max_change": float(np.max(np.abs(window_mod - window_orig))),
            "window_range": (int(start), int(end))
        }
    
    return results

def save_analysis_report(
    analysis: Dict[str, Any],
    output_path: Union[str, Path],
    include_plots: bool = True
) -> None:
    """
    Save analysis results to a report file.
    
    Args:
        analysis: Analysis results dictionary
        output_path: Path to save report
        include_plots: Whether to include ASCII plots in report
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("AlphaMask Analysis Report\n")
        f.write("=======================\n\n")
        
        f.write(f"Global Changes:\n")
        f.write(f"- Mean change: {analysis['global_mean_change']:.3f}\n")
        f.write(f"- Std change: {analysis['global_std_change']:.3f}\n\n")
        
        f.write("Position-specific Changes:\n")
        for pos, change in analysis["position_changes"].items():
            f.write(f"Position {pos}: {change:.3f}\n")
        f.write("\n")
        
        f.write("Window Analysis:\n")
        for pos, window_data in analysis["window_changes"].items():
            f.write(f"Window around position {pos}:\n")
            f.write(f"- Range: {window_data['window_range']}\n")
            f.write(f"- Mean change: {window_data['mean_change']:.3f}\n")
            f.write(f"- Std change: {window_data['std_change']:.3f}\n")
            f.write(f"- Max absolute change: {window_data['max_change']:.3f}\n\n") 

def plot_pae_comparison(
    original_pae: np.ndarray,
    modified_pae: np.ndarray,
    positions: Optional[List[int]] = None,
    title: str = "PAE Comparison",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot comparison between original and modified PAE matrices.
    
    Args:
        original_pae: Original PAE matrix
        modified_pae: Modified PAE matrix
        positions: List of positions to highlight
        title: Plot title
        save_path: Path to save plot
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Original PAE", "Modified PAE", "Difference"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    # Original PAE
    fig.add_trace(
        go.Heatmap(
            z=original_pae,
            colorscale="Viridis",
            showscale=True,
            name="Original"
        ),
        row=1, col=1
    )
    
    # Modified PAE
    fig.add_trace(
        go.Heatmap(
            z=modified_pae,
            colorscale="Viridis",
            showscale=True,
            name="Modified"
        ),
        row=1, col=2
    )
    
    # Difference
    diff = modified_pae - original_pae
    fig.add_trace(
        go.Heatmap(
            z=diff,
            colorscale="RdBu",
            zmid=0,
            showscale=True,
            name="Difference"
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        height=600,
        width=1800,
        showlegend=False
    )
    
    if positions:
        for pos in positions:
            for i in range(3):
                fig.add_shape(
                    type="line",
                    x0=pos, y0=0,
                    x1=pos, y1=len(original_pae),
                    line=dict(color="red", width=1, dash="dash"),
                    row=1, col=i+1
                )
                fig.add_shape(
                    type="line",
                    x0=0, y0=pos,
                    x1=len(original_pae), y1=pos,
                    line=dict(color="red", width=1, dash="dash"),
                    row=1, col=i+1
                )
    
    if save_path:
        fig.write_image(str(save_path))
    fig.show()

def analyze_pae_changes(
    original_pae: np.ndarray,
    modified_pae: np.ndarray,
    positions: List[int],
    window_size: int = 5
) -> Dict[str, Any]:
    """
    Analyze changes in PAE matrices around modified positions.
    
    Args:
        original_pae: Original PAE matrix
        modified_pae: Modified PAE matrix
        positions: Modified positions
        window_size: Size of window around positions to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        "global_mean_change": float(np.mean(modified_pae - original_pae)),
        "global_std_change": float(np.std(modified_pae - original_pae)),
        "position_changes": {},
        "interaction_changes": {}
    }
    
    for pos in positions:
        # Row changes (interactions from this position)
        row_change = modified_pae[pos] - original_pae[pos]
        # Column changes (interactions to this position)
        col_change = modified_pae[:, pos] - original_pae[:, pos]
        
        # Window analysis
        start = max(0, pos - window_size)
        end = min(len(original_pae), pos + window_size + 1)
        
        window_orig = original_pae[start:end, start:end]
        window_mod = modified_pae[start:end, start:end]
        
        results["position_changes"][pos] = {
            "mean_outgoing_change": float(np.mean(row_change)),
            "mean_incoming_change": float(np.mean(col_change)),
            "max_outgoing_change": float(np.max(np.abs(row_change))),
            "max_incoming_change": float(np.max(np.abs(col_change))),
        }
        
        results["interaction_changes"][pos] = {
            "window_mean_change": float(np.mean(window_mod - window_orig)),
            "window_std_change": float(np.std(window_mod - window_orig)),
            "window_max_change": float(np.max(np.abs(window_mod - window_orig))),
            "window_range": (int(start), int(end))
        }
    
    return results

def plot_structure_comparison(
    original_pdb: str,
    modified_pdb: str,
    positions: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create PyMOL visualization script for structure comparison.
    
    Args:
        original_pdb: Path to original PDB file
        modified_pdb: Path to modified PDB file
        positions: List of positions to highlight
        save_path: Path to save PyMOL script
    """
    script = f"""
from pymol import cmd
import os

# Load structures
cmd.load("{original_pdb}", "original")
cmd.load("{modified_pdb}", "modified")

# Align structures
cmd.align("modified", "original")

# Color schemes
cmd.color("lightblue", "original")
cmd.color("salmon", "modified")

# Show as cartoon with transparency
cmd.show("cartoon")
cmd.set("cartoon_transparency", 0.5)

# Show both structures
cmd.enable("original")
cmd.enable("modified")
"""

    if positions:
        script += """
# Highlight modified positions
cmd.select("modified_pos", f"resi {'+'.join(map(str, positions))}")
cmd.show("sticks", "modified_pos")
cmd.color("red", "modified_pos")
"""

    script += """
# Set view and scene
cmd.zoom()
cmd.center()
cmd.set("ray_shadows", 0)
"""

    if save_path:
        script += f"""
# Save image
cmd.png("{save_path}", ray=1, width=1200, height=1200, dpi=300)
"""

    script_path = Path(save_path).parent / "visualize.py" if save_path else Path("visualize.py")
    with open(script_path, "w") as f:
        f.write(script)
    
    print(f"PyMOL visualization script saved to: {script_path}")
    print("Run with: pymol -qc visualize.py")