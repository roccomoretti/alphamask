"""Analysis module for protein structure predictions."""

from typing import Dict, Optional, Union, Any
from pathlib import Path
import logging
import json
from dataclasses import dataclass
import concurrent.futures
from datetime import datetime
import numpy as np
import traceback
from rich.console import Console
from rich.status import Status
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.spinner import Spinner
from contextlib import contextmanager
import time

from .config import AnalysisConfig, create_analysis_config
from .rmsd import RMSDCalculator, RMSDConfig, RMSDResult
from .statistics import RMSDAnalyzer
from .visualization import RMSDVisualizer
from ..utils.compression import CompressedPredictionReader

logger = logging.getLogger(__name__)

class SimpleSpinner:
    """A simple spinner that returns string characters."""
    def __init__(self):
        self.chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current = 0
        
    def __str__(self) -> str:
        char = self.chars[self.current]
        self.current = (self.current + 1) % len(self.chars)
        return char

@contextmanager
def progress_status(console: Console, total: int):
    """Custom context manager for progress status."""
    status = Status("", console=console)
    try:
        with status:
            yield status
    finally:
        status.stop()

@dataclass
class RMSDAnalysis:
    """Main class for RMSD analysis pipeline."""
    
    def __init__(
        self,
        config: AnalysisConfig,
        output_dir: Union[str, Path],
        save_plots: bool = True,
        plot_format: str = "pdf",
        force: bool = False,
        parallel: int = 1,
        overwrite: bool = False
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.save_plots = save_plots
        self.plot_format = plot_format
        self.force = force
        self.parallel = parallel
        self.overwrite = overwrite
        
        # Initialize RMSDCalculator with RMSDConfig
        rmsd_config = RMSDConfig(
            atom_selection=config.atom_selection,
            start_residue=config.regions[0].start if config.regions else None,
            end_residue=config.regions[0].end if config.regions else None,
            use_region=bool(config.regions)
        )
        self.calculator = RMSDCalculator(rmsd_config)
        
        # Load reference structures
        self.ref_coords1 = self._load_reference(config.references['ref1'].path)
        self.ref_coords2 = None
        if 'ref2' in config.references:
            self.ref_coords2 = self._load_reference(config.references['ref2'].path)
        
        # Initialize other components
        self.analyzer = RMSDAnalyzer()
        self.visualizer = RMSDVisualizer()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
    def _load_reference(self, pdb_path: Union[str, Path]) -> np.ndarray:
        """Load reference coordinates from PDB file."""
        return self.calculator.extract_coordinates(pdb_path)
    
    def run_analysis(
        self,
        exp_dir: Union[str, Path],
        incremental: bool = False,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Run analysis pipeline on experiment directory."""
        exp_dir = Path(exp_dir)
        console = Console()
        
        # Get list of compressed files
        compressed_dir = exp_dir / "out" / "compressed"
        if not compressed_dir.exists():
            logger.warning(f"No compressed predictions found in {exp_dir}")
            return {}
            
        comp_files = list(compressed_dir.glob("*_all_atoms.h5"))
        if not comp_files:
            logger.warning(f"No compressed prediction files found in {compressed_dir}")
            return {}
            
        logger.info(f"Found {len(comp_files)} compressed prediction files")
        
        results = {}
        
        with progress_status(console, len(comp_files)) as status:
            # Process each file
            for idx, comp_file in enumerate(comp_files, 1):
                status.update(
                    f"[cyan]Processing {comp_file.name} ({idx}/{len(comp_files)})"
                )
                
                try:
                    file_results = self._process_compressed_file(comp_file)
                    if file_results:
                        results.update(file_results)
                except Exception as e:
                    logger.error(f"Failed to process {comp_file}: {str(e)}")
            
            # Final tasks
            if results:
                status.update("[yellow]Aggregating results...")
                
                if self.save_plots:
                    status.update("[green]Generating plots...")
                    self.visualizer.plot_multiple_landscapes(
                        results,
                        output_dir=self.output_dir / "plots",
                        format=self.plot_format,
                        overwrite=overwrite
                    )
                
                status.update("[blue]Saving results...")
                self._save_results(results)
                
                console.print("[bold green]Analysis complete!")
            else:
                console.print("[bold red]No valid results found")
        
        return results
    
    def _process_compressed_file(self, comp_file: Path) -> Dict[str, Any]:
        """Process a single compressed prediction file."""
        try:
            # Calculate RMSDs with overwrite flag
            rmsd_results = self.calculator.calculate_rmsds(
                reader=CompressedPredictionReader(comp_file),
                ref_coords1=self.ref_coords1,
                ref_coords2=self.ref_coords2,
                output_dir=self.output_dir / "data",
                overwrite=self.overwrite
            )
            
            if rmsd_results:
                return {comp_file.stem: rmsd_results}
            return {}
            
        except Exception as e:
            logger.error(f"Failed to process {comp_file}: {str(e)}")
            return {}
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple files."""
        return self.analyzer.aggregate_results(results)
    
    def _generate_plots(self, results: Dict[str, Any]) -> None:
        """Generate visualization plots."""
        plots_dir = self.output_dir / "plots"
        self.visualizer.create_plots(
            results,
            plots_dir,
            format=self.plot_format
        )
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results."""
        # Convert RMSDResult objects to dictionaries
        serializable_results = {}
        for file_name, result_list in results.items():
            serializable_results[file_name] = [
                {
                    'rmsd_ref1': float(r.rmsd_ref1),
                    'rmsd_ref2': float(r.rmsd_ref2) if r.rmsd_ref2 is not None else None,
                    'model_name': r.model_name,
                    'plddt': float(r.plddt) if r.plddt is not None else None
                }
                for r in result_list
            ]
        
        # Save to JSON file
        results_file = self.output_dir / "data" / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _load_cached_results(self) -> Dict[str, Any]:
        """Load cached analysis results."""
        results_file = self.output_dir / "data" / "analysis_results.json"
        with open(results_file) as f:
            json_results = json.load(f)
        
        # Convert back to RMSDResult objects
        results = {}
        for file_name, result_list in json_results.items():
            results[file_name] = [
                RMSDResult(
                    rmsd_ref1=r['rmsd_ref1'],
                    rmsd_ref2=r['rmsd_ref2'],
                    model_name=r['model_name'],
                    plddt=r['plddt']
                )
                for r in result_list
            ]
        return results
    
    def _is_analyzed(self, comp_file: Path) -> bool:
        """Check if a compressed file has already been analyzed."""
        results_file = self.output_dir / "data" / "analysis_results.json"
        if not results_file.exists():
            return False
            
        with open(results_file) as f:
            results = json.load(f)
            return comp_file.stem in results 