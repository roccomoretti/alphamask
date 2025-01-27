"""Utilities for working with compressed prediction data."""

from typing import Dict, List, Optional, Union, Any
import numpy as np
import h5py
from pathlib import Path
import logging
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from dataclasses import dataclass
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import traceback
import time

logger = logging.getLogger('alphamask.utils.compression')

@dataclass
class CompressedPrediction:
    """Container for compressed prediction data."""
    atom_positions: np.ndarray
    name: str
    plddt: Optional[np.ndarray] = None
    seed: Optional[str] = None
    model: Optional[str] = None
    recycle: Optional[int] = None

class CompressedPredictionReader:
    """Reads compressed AlphaFold prediction data."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self._predictions = None  # Cache predictions
        
    def get_predictions(self) -> List[CompressedPrediction]:
        """Get list of predictions from compressed file."""
        if self._predictions is not None:
            return self._predictions
            
        predictions = []
        start_time = time.time()
        logger.debug(f"Opening HDF5 file: {self.file_path}")
        
        try:
            # Increase chunk cache while preserving existing SWMR & libver settings
            with h5py.File(self.file_path, 'r', libver='latest', swmr=True,
                           rdcc_nbytes=1024*1024*128,  # 128 MB chunk cache
                           rdcc_nslots=1_000_000) as f:
                open_time = time.time()
                logger.debug(f"File opened in {open_time - start_time:.2f}s")
                
                if 'data' not in f:
                    logger.warning("No 'data' group found in HDF5 file")
                    return predictions
                
                data_group = f['data']
                n_predictions = len(data_group.keys())
                logger.debug(f"Found {n_predictions} prediction groups")
                predictions = [None] * n_predictions
                current_idx = 0
                
                # Comment out the detailed debug-logging for each dataset's shape
                # logger.debug("File structure:")
                # for key in data_group.keys():
                #     logger.debug(f"  - {key}")
                #     if current_idx == 0:
                #         group = data_group[key]
                #         logger.debug("First prediction details:")
                #         for subkey in group.keys():
                #             logger.debug(f"    - {subkey}: {group[subkey].shape}")
                
                for pred_name in data_group.keys():
                    read_start = time.time()
                    try:
                        pred_group = data_group[pred_name]
                        
                        if 'atom_positions' in pred_group and 'plddt' in pred_group:
                            # Read data in one go
                            coords = pred_group['atom_positions'][:]
                            plddt = pred_group['plddt'][:]
                            
                            # Parse prediction info
                            parts = pred_name.split('_')
                            seed = next((p.replace('seed_', '') for p in parts if p.startswith('seed_')), "1")
                            model = next((p.replace('model_', '') for p in parts if p.startswith('model_')), "1")
                            recycle = int(next((p.replace('r', '') for p in parts if p.startswith('r')), "0"))
                            
                            predictions[current_idx] = CompressedPrediction(
                                atom_positions=coords,
                                plddt=plddt,
                                name=pred_name,
                                seed=seed,
                                model=model,
                                recycle=recycle
                            )
                            current_idx += 1
                            
                            read_time = time.time() - read_start
                            # if current_idx % 5 == 0:
                            #     logger.debug(f"Read prediction {current_idx}/{n_predictions} in {read_time:.2f}s")
                                
                    except Exception as e:
                        logger.warning(f"Failed to read prediction {pred_name}: {str(e)}")
                        continue
                
                # Remove any unused slots
                predictions = [p for p in predictions if p is not None]
                
                end_time = time.time()
                logger.debug(
                    f"Completed reading {len(predictions)} predictions in {end_time - start_time:.2f}s\n"
                    f"  - File open: {open_time - start_time:.2f}s\n"
                    f"  - Average read time: {(end_time - open_time) / len(predictions):.3f}s per prediction"
                )
                    
        except Exception as e:
            logger.error(f"Failed to read predictions from {self.file_path}: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
        
        self._predictions = predictions
        return predictions
    
    def get_best_prediction(self) -> Optional[CompressedPrediction]:
        """Get the best prediction based on mean pLDDT."""
        try:
            with h5py.File(self.file_path, 'r', libver='latest', swmr=True) as f:
                if 'best' not in f:
                    return None
                    
                best_grp = f['best']
                return CompressedPrediction(
                    atom_positions=best_grp['atom_positions'][:],
                    plddt=best_grp['plddt'][:],
                    name="best",
                    seed="best",
                    model="best",
                    recycle=-1
                )
        except Exception as e:
            logger.error(f"Failed to get best prediction: {str(e)}")
            return None
    
    def extract_coordinates(
        self,
        prediction: CompressedPrediction,
        atom_selection: str = "CA"
    ) -> np.ndarray:
        """Extract coordinates based on atom selection."""
        if atom_selection == "CA":
            return prediction.atom_positions[:, 1]  # CA atoms
        elif atom_selection == "backbone":
            return prediction.atom_positions[:, :4]  # N, CA, C, O
        else:  # all atoms
            return prediction.atom_positions
    
    def get_mean_plddt(self, prediction: CompressedPrediction) -> float:
        """Get mean pLDDT score for prediction."""
        return float(np.mean(prediction.plddt)) 
    
    def print_structure(self) -> None:
        """Print the HDF5 file structure for debugging."""
        try:
            with h5py.File(self.file_path, 'r', libver='latest', swmr=True) as f:
                logger.debug(f"HDF5 file structure for {self.file_path}:")
                self._print_structure(f)
        except Exception as e:
            logger.error(f"Failed to print structure: {str(e)}")
            
    def _print_structure(self, obj, level=0):
        """Recursively print the HDF5 file structure."""
        indent = '  ' * level
        for key in obj.keys():
            try:
                item = obj[key]
                logger.debug(f"{indent}{key}:")
                if isinstance(item, h5py.Group):
                    self._print_structure(item, level + 1)
                else:
                    logger.debug(f"{indent}  Shape: {item.shape}, Type: {item.dtype}")
            except Exception as e:
                logger.debug(f"{indent}  Error accessing {key}: {str(e)}") 