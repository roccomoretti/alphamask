"""
Main RMSD visualization class.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

import matplotlib.pyplot as plt
import numpy as np

from .base_visualizer import BaseVisualizer
from .violin_plots import create_violin_plots
from .landscape_plots import (
    create_rmsd_landscape,
    create_combined_landscape,
    create_combined_landscape_breakdown,
    apriori_create_summary_landscape
)
from .interactive_plots import create_interactive_plot
from .recycle_plots import (
    create_recycle_landscapes,
    create_recycle_progression,
    create_recycle_summary,
    create_model_comparison,
    create_cumulative_landscapes,
    create_summary_collages,
    create_model_comparisons
)
from ..storage import Storage

logger = logging.getLogger(__name__)

class RMSDVisualizer(BaseVisualizer):
    """
    Class for visualizing RMSD data from AlphaMask runs.
    """
    
    def __init__(self, storage: Optional[Storage] = None):
        """Initialize visualizer.
        
        Args:
            storage: Storage instance to use for data access
        """
        super().__init__()
        self.storage = storage
    
    def create_plots(
        self,
        storage: Storage,
        position: int,
        output_dir: Optional[Path] = None,
        format: str = "pdf",
        show: bool = False,
        interactive: bool = False,
        recycle: bool = False,
        max_rmsd: float = 0
    ) -> Dict[str, plt.Figure]:
        """
        Create all available plots for a given position.
        
        Args:
            storage: Storage instance to use for data access
            position: Position number to analyze
            output_dir: Directory to save plots
            format: Output format (pdf, png)
            show: Whether to display plots
            interactive: Whether to create interactive plots
            
        Returns:
            Dictionary of generated figures
        """
        self.storage = storage
        figures = {}
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create rmsd landscape plots
        landscape_figs = self._create_landscape_plots(
            storage=storage,
            position=position,
            output_dir=output_dir,
            format=format,
            show=show,
            max_rmsd=max_rmsd
        )
        figures.update(landscape_figs)
        logger.info(f"Landscape plots created for position {position}")

        
        # Create violin plots
        violin_figs = create_violin_plots(
            storage=storage,
            position=position,
            output_dir=output_dir,
            format=format,
            show=show,
            max_rmsd=max_rmsd
        )
        figures.update(violin_figs)
        logger.info(f"Violin plots created for position {position}")
        
        # Create landscape plots
        landscape_figs = self._create_landscape_plots(
            storage=storage,
            position=position,
            output_dir=output_dir,
            format=format,
            show=show,
            max_rmsd=max_rmsd
        )
        figures.update(landscape_figs)
        logger.info(f"Landscape plots created for position {position}")
        
        if recycle:
            # Create recycle plots
            recycle_figs = create_recycle_landscapes(
                storage=storage,
                position=position,
                output_dir=output_dir,
                format=format,
                show=show,
                max_rmsd=max_rmsd
            )
            figures.update(recycle_figs)
            logger.info(f"Recycle plots created for position {position}")
        
            # Create recycle progression plots
            progression_figs = create_recycle_progression(
                storage=storage,
                position=position,
                output_dir=output_dir,
                format=format,
                show=show,
                max_rmsd=max_rmsd
            )
            figures.update(progression_figs)
            logger.info(f"Recycle progression plots created for position {position}")
        
            # Create recycle summary plot
            summary_fig = create_recycle_summary(
                storage=storage,
                position=position,
                output_dir=output_dir,
                format=format,
                show=show,
                max_rmsd=max_rmsd
            )
            figures['recycle_summary'] = summary_fig
            logger.info(f"Recycle summary plot created for position {position}")
        
            # Create model comparison plots
            comparison_figs = create_model_comparison(
                storage=storage,
                position=position,
                output_dir=output_dir,
                format=format,
                show=show,
                max_rmsd=max_rmsd
            )
            figures.update(comparison_figs)
            logger.info(f"Model comparison plots created for position {position}")
            
            # Create cumulative landscape plots
            cumulative_figs = create_cumulative_landscapes(
                storage=storage,
                position=position,
                output_dir=output_dir,
                format=format,
                show=show,
                max_rmsd=max_rmsd
            )
            figures.update(cumulative_figs)
            logger.info(f"Cumulative landscape plots created for position {position}")
        
            # Create summary collages
            collage_figs = create_summary_collages(
                storage=storage,
                position=position,
                output_dir=output_dir,
                format=format,
                show=show,
                max_rmsd=max_rmsd
            )
            figures.update(collage_figs)
            logger.info(f"Summary collages created for position {position}")
            
            # Create detailed model comparisons
            model_comparison_figs = create_model_comparisons(
                storage=storage,
                position=position,
                output_dir=output_dir,
                format=format,
                show=show,
                max_rmsd=max_rmsd
            )
            figures.update(model_comparison_figs)
            logger.info(f"Detailed model comparisons created for position {position}")
            
            # Create interactive plots if requested
            if interactive:
                # Get all recycle data
                recycle_data = storage.get_all_recycle_data(position)
                if recycle_data:
                    # Create results dictionary with proper key format
                    results_dict = {}
                    for model in sorted(recycle_data.keys()):
                        for recycle, data in recycle_data[model].items():
                            key = f"pos_{position}_{model}"
                            if key not in results_dict:
                                results_dict[key] = []
                            results_dict[key].append(data)
                    # Not working for the moment TypeError: create_interactive_plot() got an unexpected keyword argument 'results_dict'
                    # interactive_figs = create_interactive_plot(
                    #     results_dict=results_dict,
                    #     control_results=None,
                    #     max_rmsd=None
                    # )
                    # if interactive_figs:
                    #     figures['interactive'] = interactive_figs
                    #     logger.info(f"Interactive plots created for position {position}")
        return figures
    
    def _create_landscape_plots(
        self,
        storage: Storage,
        position: int,
        output_dir: Optional[Path] = None,
        format: str = "pdf",
        show: bool = False,
        max_rmsd: float = 0
    ) -> Dict[str, plt.Figure]:
        """
        Create RMSD landscape plots for a given position.
        
        Args:
            storage: Storage instance to use for data access
            position: Position number to analyze
            output_dir: Directory to save plots
            format: Output format (pdf, png)
            show: Whether to display plots
            max_rmsd: Maximum RMSD to use for plotting
            
        Returns:
            Dictionary of generated figures
        """
        figures = {}
        
        # Get all recycle data
        recycle_data = storage.get_all_recycle_data(position)
        if not recycle_data:
            logger.warning(f"No data found for position {position}")
            return figures
        
        # Combine all RMSD data
        rmsd_ref1_values = []
        rmsd_ref2_values = []
        
        for model in sorted(recycle_data.keys()):
            for recycle in sorted(recycle_data[model].keys()):
                data = recycle_data[model][recycle]
                if data.rmsd_ref1 is not None:
                    rmsd_ref1_values.extend(data.rmsd_ref1.flatten())
                    if data.rmsd_ref2 is not None:
                        rmsd_ref2_values.extend(data.rmsd_ref2.flatten())
        
        if not rmsd_ref1_values:
            logger.warning(f"No RMSD data found for position {position}")
            return figures
        
        # Create landscape plot
        fig = create_rmsd_landscape(
            rmsd_ref1=np.array(rmsd_ref1_values),
            rmsd_ref2=np.array(rmsd_ref2_values) if rmsd_ref2_values else None,
            title=f"Position {position}",
            show=False,
            max_rmsd=max_rmsd
        )
        figures['landscape'] = fig
        
        if output_dir:
            fig.savefig(output_dir / f"pos_{position}_landscape.{format}",
                       format=format, bbox_inches='tight', dpi=300)
            if not show:
                plt.close(fig)
        
        return figures 
    
    def create_combined_landscape(
        self,
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
        return create_combined_landscape(
            storage=storage,
            output_dir=output_dir,
            format=format,
            show=show,
            max_rmsd=max_rmsd
        ) 
    
    def create_combined_landscape_breakdown(
        self,
        storage: Storage,
        output_dir: Optional[Path] = None,
        format: str = "pdf",
        show: bool = False,
        max_rmsd: Optional[float] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create breakdown plots of combined RMSD landscapes by model and recycle.
        
        Args:
            storage: Storage instance to use for data access
            output_dir: Directory to save plots
            format: Output format (pdf, png)
            show: Whether to display plots
            max_rmsd: Optional maximum RMSD value for plot scaling
            
        Returns:
            Dictionary of generated figures
        """
        self.storage = storage
        return create_combined_landscape_breakdown(
            storage=storage,
            output_dir=output_dir,
            format=format,
            show=show,
            max_rmsd=max_rmsd
        ) 
        
        
    def apriori_create_summary_landscape(
        self,
        storages: Dict[str, Storage],
        output_dir: Optional[Path] = None,
        format: str = "pdf",
        show: bool = False,
        max_rmsd: Optional[float] = None
    ) -> plt.Figure:
        """
        Create a summary landscape plot comparing all positions.
        
        Args:
            storages: Dictionary mapping condition names to their Storage objects
            output_dir: Directory to save plots
            format: Output format (pdf, png)
            show: Whether to display plots
            max_rmsd: Optional maximum RMSD value for plot scaling
            
        Returns:
            Generated figure
        """
        return apriori_create_summary_landscape(
            storages=storages,
            output_dir=output_dir,
            format=format,
            show=show,
            max_rmsd=max_rmsd
        )