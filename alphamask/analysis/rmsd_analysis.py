"""RMSD analysis pipeline implementation."""

from typing import Dict, Optional, Union, Any
from pathlib import Path
import logging
import traceback
import numpy as np
import h5py
import matplotlib.pyplot as plt
from dataclasses import dataclass
import re

from .config import AnalysisConfig
from .rmsd import RMSDCalculator, RMSDConfig, RMSDResult
from .statistics import RMSDAnalyzer
from .visualization.rmsd_visualizer import RMSDVisualizer
from .storage import Storage
from ..utils.compression import CompressedPredictionReader
from .collective_variables import CVCalculator, CVConfig, CVResult

logger = logging.getLogger(__name__)


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
        
        # Initialize storage with in-memory mode
        self.storage = Storage(
            self.output_dir / "analysis.h5",
            in_memory=True
        )
        
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
        
        # Initialize CV calculator if configured
        self.cv_calculator = None
        if hasattr(config, 'calculate_cvs') and config.calculate_cvs:
            cv_config = CVConfig(
                calculate_cv1=True,
                calculate_cv2=True,
                cv1_residues=(151, 152, 153, 154),
                cv2_residues=(41, 58, 156)
            )
            self.cv_calculator = CVCalculator(cv_config)

        
    def _load_reference(self, pdb_path: Union[str, Path]) -> np.ndarray:
        """Load reference coordinates from PDB file."""
        return self.calculator.extract_coordinates(pdb_path)
    
    def _extract_info_from_name(self, name: str) -> Dict[str, int]:
        """Extract position, model, recycle, and seed from name."""
        # Try WT pattern first
        wt_pattern = r"WT_pos_(\d+).*_model_(\d+).*_r(\d+)_seed_(\d+)_.*"
        wt_match = re.match(wt_pattern, name)
        if wt_match:
            return {
                'position': int(wt_match.group(1)),
                'model': int(wt_match.group(2)),
                'recycle': int(wt_match.group(3)),
                'seed': int(wt_match.group(4))
            }
            
        # Try mutant pattern
        mut_pattern = r".*_pos_(\d+).*_model_(\d+).*_r(\d+)_seed_(\d+).*"
        mut_match = re.match(mut_pattern, name)
        if mut_match:
            return {
                'position': int(mut_match.group(1)),
                'model': int(mut_match.group(2)),
                'recycle': int(mut_match.group(3)),
                'seed': int(mut_match.group(4))
            }
            
        # Try apriori pattern
        apriori_pattern = r".*_model_(\d+)_ptm_r(\d+)_seed_(\d+)(?:_mask_(\d+))?(?:_mut_[A-Z]\d+[A-Z])?(?:_id_X)?"
        apriori_match = re.match(apriori_pattern, name)
        if apriori_match:
            position = int(apriori_match.group(4)) if apriori_match.group(4) else 89  # Default to 89 if not specified
            return {
                'position': position,
                'model': int(apriori_match.group(1)),
                'recycle': int(apriori_match.group(2)),
                'seed': int(apriori_match.group(3))
            }
            
        raise ValueError(f"Invalid model name format: {name}")
    
    def run_analysis(
        self,
        exp_dir: Path,
        incremental: bool = False,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Run RMSD analysis on experiment directory."""
        try:
            # Check if analysis file exists and is not empty
            h5_file = self.output_dir / "analysis.h5"
            if incremental and h5_file.exists() and h5_file.stat().st_size > 0:
                logger.info("Analysis file exists and is not empty, skipping RMSD calculations")
                # Load existing results from H5 file
                self.storage._load_to_memory()
                # Return True to indicate we have valid data for plotting
                return True
            
            # Quick check if we have all positions
            if not overwrite and h5_file.exists():
                # Get a sample compressed file to check what positions we need
                sample_file = next(exp_dir.glob("*_all_atoms.h5"), None)
                if sample_file:
                    needed_positions = set()
                    reader = CompressedPredictionReader(sample_file)
                    try:
                        predictions = reader.get_predictions()
                        # Double check this for loop as it might not be needed
                        for pred in predictions:
                            try:
                                info = self._extract_info_from_name(pred.name)
                                needed_positions.add(info['position'])
                            except ValueError:
                                continue
                        
                        # Check if all needed positions exist in storage
                        if needed_positions and needed_positions.issubset(self.storage._memory_store['metadata']['positions']):
                            logger.info("All required positions already exist in storage, skipping analysis")
                            return True
                    except Exception as e:
                        logger.warning(f"Failed to check positions in {sample_file}: {e}")
            
            # Process each compressed file
            results = {}
            for comp_file in exp_dir.glob("*_all_atoms.h5"):
                try:
                    file_results = self._process_compressed_file(comp_file)
                    if file_results:
                        results.update(file_results)
                except Exception as e:
                    logger.error(f"Failed to process {comp_file}: {str(e)}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(traceback.format_exc())
            
            # Save results to H5 file
            if results:
                self.storage.save()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            return False

    def _process_compressed_file(self, comp_file: Path) -> Dict[str, Any]:
        """Process a single compressed prediction file."""
        try:
            # First, check if we need to process this file by looking at a sample model name
            reader = CompressedPredictionReader(comp_file)
            predictions = reader.get_predictions()
            if not predictions:
                logger.warning(f"No predictions found in {comp_file}")
                return {}
                
            # Log number of predictions found
            logger.debug(f"Found {len(predictions)} predictions in {comp_file}")
            # Log first few predictions to check format
            for i, pred in enumerate(predictions[:3]):
                logger.debug(f"Prediction {i}: name={pred.name}")

            # Check first prediction to see if we need to process this file
            try:
                info = self._extract_info_from_name(predictions[0].name)
                logger.debug(f"First prediction info: {info}")
                
                # Check if we already have data for this position/model/recycle
                if (info['position'] in self.storage._memory_store['positions'] and
                    info['model'] in self.storage._memory_store['positions'][info['position']] and
                    info['recycle'] in self.storage._memory_store['positions'][info['position']][info['model']]):
                    logger.debug(f"Skipping {comp_file} - data already exists for position {info['position']}, model {info['model']}, recycle {info['recycle']}")
                    return {}
            except ValueError:
                logger.warning(f"Invalid model name format in {comp_file}: {predictions[0].name}")
                return {}

            # If we get here, we need to process this file
            # Calculate RMSDs
            rmsd_results = self.calculator.calculate_rmsds(
                reader=CompressedPredictionReader(comp_file),
                ref_coords1=self.ref_coords1,
                ref_coords2=self.ref_coords2
            )
            
            # Log number of RMSD results
            logger.debug(f"Calculated {len(rmsd_results)} RMSD results from {comp_file}")
            # Log first few results
            for i, result in enumerate(rmsd_results[:3]):
                logger.debug(f"RMSD Result {i}: model_name={result.model_name}")
            
            # Prepare batch data
            batch_data = []
            model_seed_counts = {}  # Track seeds per model/recycle
            
            for result in rmsd_results:
                if not result.model_name:
                    logger.warning("Skipping result with no model name")
                    continue
                    
                try:
                    info = self._extract_info_from_name(result.model_name)
                    
                    key = (info['model'], info['recycle'])
                    if key not in model_seed_counts:
                        model_seed_counts[key] = set()
                    model_seed_counts[key].add(info['seed'])
                    
                    batch_data.append((
                        info['position'],
                        info['model'],
                        info['recycle'],
                        info['seed'],
                        result.rmsd_ref1,
                        result.rmsd_ref2,
                        result.plddt_array
                    ))
                except ValueError as e:
                    logger.warning(f"Skipping invalid model name {result.model_name}: {e}")
                    continue
            
            # Log seed counts per model/recycle before storage
            for (model, recycle), seeds in model_seed_counts.items():
                logger.debug(f"Before storage - Model {model}, Recycle {recycle}: {len(seeds)} seeds - {sorted(seeds)}")
            
            # Log batch data structure
            if batch_data:
                positions = sorted(set(pos for pos, *_ in batch_data))
                models = sorted(set(model for _, model, *_ in batch_data))
                recycles = sorted(set(recycle for _, _, recycle, *_ in batch_data))
                logger.debug(f"Batch data structure:")
                logger.debug(f"  Positions: {positions}")
                logger.debug(f"  Models: {models}")
                logger.debug(f"  Recycles: {recycles}")
                logger.debug(f"  Total batch entries: {len(batch_data)}")
            
            # Store batch data
            if batch_data:
                self.storage.store_rmsd_results_batch(batch_data)
                results = {comp_file.stem: rmsd_results}
            else:
                results = {}
            
            # Calculate CVs if configured
            cv_results = []
            if self.cv_calculator is not None:
                for pred in predictions:
                    cv_result = self.cv_calculator.calculate_cvs(pred)
                    cv_results.append(cv_result)
            
            # Store CV results in storage
            if cv_results:
                # Add CV results to storage logic here
                pass
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process {comp_file}: {str(e)}")
            return {} 

    def save_to_disk(self):
        """Explicitly save all data to disk."""
        logger.info("Saving all results to disk...")
        self.storage.save_to_disk() 