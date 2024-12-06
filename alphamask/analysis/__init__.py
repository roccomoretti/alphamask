"""
Analysis module for AlphaMask package.

This module provides functionality for analyzing protein structure predictions,
including RMSD calculations, statistical analysis, and visualization.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import plotly.io as pio
from IPython.display import display

from .rmsd import RMSDConfig, RMSDResult, RMSDCalculator
from .statistics import RMSDStatistics, RMSDAnalyzer
from .visualization import PlotConfig, RMSDVisualizer

# Configure plotly to render in notebooks
pio.renderers.default = 'notebook'

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for RMSD analysis pipeline."""
    rmsd_config: RMSDConfig
    plot_config: Optional[PlotConfig] = None
    output_dir: Optional[Union[str, Path]] = None
    save_plots: bool = True
    show_plots: bool = True
    include_control: bool = True

class RMSDAnalysis:
    """
    Main class for running RMSD analysis pipeline.
    
    Attributes:
        config (AnalysisConfig): Analysis configuration
        calculator (RMSDCalculator): RMSD calculator instance
        analyzer (RMSDAnalyzer): Statistical analyzer instance
        visualizer (RMSDVisualizer): Visualization handler instance
    """
    
    def __init__(self, config: AnalysisConfig):
        """Initialize analysis pipeline."""
        self.config = config
        self.calculator = RMSDCalculator(config.rmsd_config)
        self.analyzer = RMSDAnalyzer()
        self.visualizer = RMSDVisualizer(config.plot_config)
        
        if self.config.output_dir:
            self.config.output_dir = Path(self.config.output_dir)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
    def run_analysis(
        self,
        ref_pdb1: Union[str, Path],
        model_dirs: Dict[str, Union[str, Path]],
        ref_pdb2: Optional[Union[str, Path]] = None,
        control_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Run complete RMSD analysis pipeline.
        
        Args:
            ref_pdb1: Path to first reference PDB
            model_dirs: Dictionary mapping structure names to model directories
            ref_pdb2: Optional path to second reference PDB
            control_dir: Optional path to control structure directory
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting RMSD analysis pipeline")
        
        # Extract reference coordinates
        ref_coords1 = self.calculator.extract_coordinates(ref_pdb1)
        ref_coords2 = None
        if ref_pdb2:
            ref_coords2 = self.calculator.extract_coordinates(ref_pdb2)
            
        # Process control structures if requested
        if control_dir:
            if self.config.include_control:
                logger.info("Processing control structures from: %s", control_dir)
                control_results = self.calculator.process_models(
                    control_dir,
                    ref_coords1,
                    ref_coords2
                )
            else:
                logger.info("Control directory provided but include_control=False, skipping control")
                control_results = None
        else:
            logger.info("No control directory provided")
            control_results = None
            
        # Add debug logging after processing control
        if control_results:
            logger.info("Successfully processed %d control structures", len(control_results))
            
        # Process model structures
        results_dict = {}
        for name, model_dir in model_dirs.items():
            logger.info(f"Processing {name} structures")
            results_dict[name] = self.calculator.process_models(
                model_dir,
                ref_coords1,
                ref_coords2
            )
            
        # Calculate statistics
        logger.info("Calculating statistics")
        stats_list = []
        
        if control_results:
            stats_list.append(
                self.analyzer.calculate_statistics(control_results, "Control", "ref1")
            )
            if ref_coords2 is not None:
                stats_list.append(
                    self.analyzer.calculate_statistics(control_results, "Control", "ref2")
                )
                
        for name, results in results_dict.items():
            stats_list.append(
                self.analyzer.calculate_statistics(results, name, "ref1")
            )
            if ref_coords2 is not None:
                stats_list.append(
                    self.analyzer.calculate_statistics(results, name, "ref2")
                )
                
        summary_df = self.analyzer.create_summary_dataframe(stats_list)
        detailed_df = self.analyzer.create_detailed_dataframe(results_dict, control_results)
        
        # Create visualizations
        if self.config.save_plots or self.config.show_plots:
            logger.info("Creating visualizations")
            
            violin_fig = self.visualizer.plot_violin_distributions(
                results_dict,
                control_results,
                show=self.config.show_plots
            )
            
            landscape_fig = self.visualizer.plot_multiple_landscapes(
                results_dict,
                control_results,
                show=self.config.show_plots
            )
            
            interactive_fig = self.visualizer.create_interactive_plot(
                results_dict,
                control_results
            )
            
            # Display the interactive plot immediately if show_plots is True
            if self.config.show_plots:
                display(interactive_fig)
            
            # Save plots if requested
            if self.config.save_plots and self.config.output_dir:
                logger.info("Saving plots")
                plots_dir = self.config.output_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                violin_fig.savefig(plots_dir / "violin_distributions.png", dpi=300)
                landscape_fig.savefig(plots_dir / "landscapes.png", dpi=300)
                interactive_fig.write_html(str(plots_dir / "interactive_plot.html"))
                
        # Save results
        if self.config.output_dir:
            logger.info("Saving analysis results")
            self.analyzer.save_results(
                detailed_df,
                summary_df,
                self.config.output_dir / "data"
            )
            
        return {
            "control_results": control_results,
            "model_results": results_dict,
            "statistics": {
                "summary": summary_df,
                "detailed": detailed_df
            },
            "plots": {
                "violin": violin_fig,
                "landscape": landscape_fig,
                "interactive": interactive_fig
            } if (self.config.save_plots or self.config.show_plots) else None
        }

# Expose main classes and types
__all__ = [
    "RMSDConfig",
    "RMSDResult",
    "RMSDCalculator",
    "RMSDStatistics",
    "RMSDAnalyzer",
    "PlotConfig",
    "RMSDVisualizer",
    "AnalysisConfig",
    "RMSDAnalysis"
] 