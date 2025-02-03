import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RecycleRMSDResult:
    """Container for recycle-specific RMSD results."""
    rmsd_ref1: Optional[np.ndarray] = None
    rmsd_ref2: Optional[np.ndarray] = None
    plddt: Optional[np.ndarray] = None
    seeds: Optional[np.ndarray] = None
    n_residues: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecycleRMSDResult':
        """Create instance from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

@dataclass
class RecycleData:
    """Container for recycle-specific data."""
    rmsd_ref1: np.ndarray
    rmsd_ref2: Optional[np.ndarray] = None
    plddt: Optional[np.ndarray] = None
    seeds: Optional[np.ndarray] = None

class Storage:
    """Storage class for RMSD analysis results."""
    
    def __init__(self, file_path: Union[str, Path], in_memory: bool = True):
        """Initialize storage.
        
        Args:
            file_path: Path to HDF5 file
            in_memory: If True, keep all data in memory and only write at the end
        """
        self.file_path = Path(file_path)
        self.in_memory = in_memory
        
        # Initialize all required attributes
        self.data = {}
        self.positions = set()
        
        self._memory_store = {
            'metadata': {
                'positions': set(),
                'max_models': 0,
                'max_recycles': 0
            },
            'positions': {}
        }
        self._modified = False
        self._load_to_memory()
    
    def _load_to_memory(self):
        """Load data from disk to memory."""
        try:
            # Create parent directories if they don't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing file
            try:
                with h5py.File(self.file_path, 'r') as f:
                    logger.debug(f"Loading data from {self.file_path}")
                    
                    # Load all positions at once
                    positions = []
                    for key in f.keys():
                        if key == 'metadata':
                            continue
                            
                        # Handle both direct position numbers and 'pos_X' format
                        pos_str = key.split('_')[-1] if '_' in key else key
                        try:
                            pos_int = int(pos_str)
                            positions.append(pos_int)
                            
                            # Initialize data structures
                            self._memory_store['metadata']['positions'].add(pos_int)
                            self._memory_store['positions'][pos_int] = {}
                            self.positions.add(pos_int)
                            self.data[pos_int] = {}
                            
                        except ValueError:
                            continue
                    
                    # Bulk load all position data
                    for pos_int in positions:
                        pos_group = f[f'pos_{pos_int}']
                        
                        for model_key in pos_group.keys():
                            model_int = int(model_key.split('_')[1])
                            self._memory_store['metadata']['max_models'] = max(
                                self._memory_store['metadata']['max_models'], 
                                model_int
                            )
                            self._memory_store['positions'][pos_int][model_int] = {}
                            
                            for recycle_key in pos_group[model_key].keys():
                                recycle_int = int(recycle_key)
                                self._memory_store['metadata']['max_recycles'] = max(
                                    self._memory_store['metadata']['max_recycles'], 
                                    recycle_int
                                )
                                
                                # Initialize and load data in one go
                                recycle_data = {
                                    'rmsd_ref1': [],
                                    'rmsd_ref2': [],
                                    'plddt': [],
                                    'seeds': []
                                }
                                
                                recycle_group = pos_group[model_key][recycle_key]
                                if 'rmsd_ref1' in recycle_group:
                                    recycle_data['rmsd_ref1'].extend(recycle_group['rmsd_ref1'][()])
                                if 'rmsd_ref2' in recycle_group:
                                    recycle_data['rmsd_ref2'].extend(recycle_group['rmsd_ref2'][()])
                                if 'plddt' in recycle_group:
                                    recycle_data['plddt'].extend(recycle_group['plddt'][()])
                                if 'seeds' in recycle_group:
                                    recycle_data['seeds'].extend(recycle_group['seeds'][()])
                                    
                                self._memory_store['positions'][pos_int][model_int][recycle_int] = recycle_data
                    
                    logger.debug(f"Loaded positions: {sorted(self.positions)}")
                    
            except (OSError, IOError) as e:
                if not isinstance(e, FileNotFoundError):
                    logger.warning(f"Failed to load existing file: {e}")
                self.initialize()
                
        except Exception as e:
            logger.error(f"Failed to load/initialize storage: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            raise
    
    def _save_to_disk(self):
        """Save data to disk."""
        try:
            # Ensure parent directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(self.file_path, 'w') as f:
                # Save metadata
                metadata = f.create_group('metadata')
                metadata.attrs['max_models'] = self._memory_store['metadata']['max_models']
                metadata.attrs['max_recycles'] = self._memory_store['metadata']['max_recycles']
                
                # Save position data
                for position, pos_data in self.data.items():
                    pos_group = f.create_group(f'pos_{position}')
                    
                    for model, model_data in pos_data.items():
                        model_group = pos_group.create_group(f'model_{model}')
                        
                        for recycle, result in model_data.items():
                            recycle_group = model_group.create_group(str(recycle))
                            result_dict = result.to_dict()
                            
                            for key, value in result_dict.items():
                                if value is not None:
                                    recycle_group.create_dataset(key, data=value)
                                    
            logger.debug(f"Successfully saved data for {len(self.positions)} positions")
                                    
        except Exception as e:
            logger.error(f"Failed to save to disk: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            raise

    def initialize(self):
        """Initialize empty storage."""
        logger.debug("Initializing empty storage")
        self.data = {}
        self.positions = set()
        self._memory_store = {
            'metadata': {
                'positions': set(),
                'max_models': 0,
                'max_recycles': 0
            },
            'positions': {}
        }
        self._modified = False
        
        # Create a properly structured empty file
        with h5py.File(self.file_path, 'w') as f:
            # Create metadata group
            metadata = f.create_group('metadata')
            metadata.attrs['max_models'] = 0
            metadata.attrs['max_recycles'] = 0
            metadata.attrs['created'] = str(datetime.now())
            logger.debug(f"Initialized empty HDF5 file at {self.file_path}")

    def store_rmsd_results_batch(
        self,
        results: List[Tuple[int, int, int, int, float, Optional[float], np.ndarray]],
        overwrite: bool = False
    ) -> None:
        """Store multiple RMSD results in batch using in-memory storage."""
        if not results:
            return

        # Quick check if all positions in batch already exist
        if not overwrite:
            all_positions = {pos for pos, _, _, _, _, _, _ in results}
            existing_positions = set()
            for pos in all_positions:
                if pos in self._memory_store['positions']:
                    existing_positions.add(pos)
            
            if len(existing_positions) == len(all_positions):
                logger.debug(f"Skipping batch - all positions exist: {sorted(existing_positions)}")
                return

        # Track unique positions in this batch
        positions_in_batch = set()
        skipped_positions = set()
        logger.debug(f"Storing {len(results)} results in batch")

        # Track seeds per position/model/recycle
        seed_counts = {}

        # Group results by position/model/recycle for batch processing
        grouped_results = {}
        for pos, model, recycle, seed, rmsd1, rmsd2, plddt_array in results:
            key = (pos, model, recycle)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append((seed, rmsd1, rmsd2, plddt_array))

        # Process each result
        for (pos, model, recycle), entries in grouped_results.items():
            # Skip if position already has data and we're not overwriting
            if not overwrite and pos in self._memory_store['positions'] and \
               model in self._memory_store['positions'][pos] and \
               recycle in self._memory_store['positions'][pos][model]:
                skipped_positions.add(pos)
                continue

            positions_in_batch.add(pos)
            
            # Track seed counts
            key = (pos, model, recycle)
            if key not in seed_counts:
                seed_counts[key] = set()
            seed_counts[key].update(seed for seed, _, _, _ in entries)
            
            # Update metadata
            self._memory_store['metadata']['positions'].add(pos)
            self._memory_store['metadata']['max_models'] = max(
                self._memory_store['metadata']['max_models'], 
                model
            )
            self._memory_store['metadata']['max_recycles'] = max(
                self._memory_store['metadata']['max_recycles'], 
                recycle
            )
            
            # Initialize position data structure if needed
            if pos not in self._memory_store['positions']:
                self._memory_store['positions'][pos] = {}
            if model not in self._memory_store['positions'][pos]:
                self._memory_store['positions'][pos][model] = {}
            if recycle not in self._memory_store['positions'][pos][model]:
                self._memory_store['positions'][pos][model][recycle] = {
                    'rmsd_ref1': [],
                    'rmsd_ref2': [],
                    'plddt': [],
                    'seeds': []
                }
            
            # Store the data
            recycle_data = self._memory_store['positions'][pos][model][recycle]
            for seed, rmsd1, rmsd2, plddt_array in entries:
                recycle_data['rmsd_ref1'].append(rmsd1)
                if rmsd2 is not None:
                    recycle_data['rmsd_ref2'].append(rmsd2)
                if plddt_array is not None:
                    recycle_data['plddt'].append(plddt_array)
                recycle_data['seeds'].append(seed)
        
        # Log detailed seed counts
        for (pos, model, recycle), seeds in seed_counts.items():
            #   logger.debug(f"Position {pos}, Model {model}, Recycle {recycle}: {len(seeds)} unique seeds - {sorted(seeds)}")
            pass

        if positions_in_batch:
            logger.debug(f"Added results to positions: {sorted(positions_in_batch)}")
        if skipped_positions:
            logger.debug(f"Skipped existing positions: {sorted(skipped_positions)}")
        self._modified = bool(positions_in_batch)

    def save_to_disk(self) -> None:
        """Save all data to disk."""
        if not self._modified:
            return
            
        logger.info("Saving all results to disk...")
        try:
            with h5py.File(self.file_path, 'w') as f:
                # Save metadata
                metadata = f.create_group('metadata')
                metadata.attrs['max_models'] = self._memory_store['metadata']['max_models']
                metadata.attrs['max_recycles'] = self._memory_store['metadata']['max_recycles']
                metadata.attrs['positions'] = sorted(list(self._memory_store['metadata']['positions']))
                
                # Save position data
                for pos in sorted(self._memory_store['positions'].keys()):
                    pos_group = f.create_group(f'pos_{pos}')
                    
                    for model in sorted(self._memory_store['positions'][pos].keys()):
                        model_group = pos_group.create_group(f'model_{model}')
                        
                        for recycle in sorted(self._memory_store['positions'][pos][model].keys()):
                            recycle_group = model_group.create_group(str(recycle))
                            recycle_data = self._memory_store['positions'][pos][model][recycle]
                            
                            # Convert lists to numpy arrays and save
                            recycle_group.create_dataset('rmsd_ref1', data=np.array(recycle_data['rmsd_ref1']))
                            if recycle_data['rmsd_ref2']:
                                recycle_group.create_dataset('rmsd_ref2', data=np.array(recycle_data['rmsd_ref2']))
                            if recycle_data['plddt']:
                                recycle_group.create_dataset('plddt', data=np.array(recycle_data['plddt']))
                            recycle_group.create_dataset('seeds', data=np.array(recycle_data['seeds']))
                
            logger.info(f"Successfully saved data for {len(self._memory_store['positions'])} positions")
            self._modified = False
            
        except Exception as e:
            logger.error(f"Failed to save to disk: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            raise

    def save(self) -> None:
        """Alias for save_to_disk for compatibility."""
        self.save_to_disk()

    def get_rmsd_results(
        self,
        position: int,
        model: int,
        recycle: int,
        seed: int
    ) -> Optional[Dict[str, float]]:
        """Retrieve RMSD results for a specific configuration."""
        with h5py.File(self.file_path, 'r') as f:
            base_path = f'positions/pos_{position}/models/model_{model}/recycles/r{recycle}/seeds/s{seed:03d}'
            
            try:
                results = {
                    'rmsd_ref1': float(f[f'{base_path}/rmsd_ref1'][()]),
                    'rmsd_ref2': float(f[f'{base_path}/rmsd_ref2'][()]) if f'{base_path}/rmsd_ref2' in f else None,
                    'plddt': float(f[f'{base_path}/plddt'][()]) if f'{base_path}/plddt' in f else None
                }
                return results
            except KeyError:
                return None

    def get_position_summary(self, position: int) -> Dict[str, Any]:
        """Get summary statistics for a position."""
        with h5py.File(self.file_path, 'r') as f:
            summary = {
                'position': position,
                'models': {},
                'overall': {
                    'rmsd_ref1': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                    'rmsd_ref2': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                    'plddt': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                    'total_predictions': 0
                }
            }
            
            base_path = f'positions/pos_{position}'
            if base_path not in f:
                logger.warning(f"No data found for position {position}")
                return summary
            
            all_rmsd1 = []
            all_rmsd2 = []
            all_plddt = []
            
            # Collect data for each model and recycle
            for model in range(1, f['metadata'].attrs['max_models'] + 1):
                model_path = f'{base_path}/models/model_{model}'
                if model_path not in f:
                    continue
                    
                model_summary = {
                    'recycles': {},
                    'total_predictions': 0
                }
                
                for recycle in range(f['metadata'].attrs['max_recycles'] + 1):
                    recycle_path = f'{model_path}/recycles/r{recycle}'
                    if recycle_path not in f:
                        continue
                        
                    # Get data for this recycle
                    rmsd1 = f[f'{recycle_path}/rmsd_ref1'][:]
                    rmsd2 = f[f'{recycle_path}/rmsd_ref2'][:] if f'{recycle_path}/rmsd_ref2' in f else None
                    plddt = f[f'{recycle_path}/plddt'][:]
                    
                    # Add to overall collections
                    all_rmsd1.extend(rmsd1)
                    if rmsd2 is not None:
                        all_rmsd2.extend(rmsd2)
                    all_plddt.extend(plddt)
                    
                    # Calculate recycle statistics
                    recycle_stats = {
                        'rmsd_ref1': {
                            'mean': float(np.mean(rmsd1)),
                            'std': float(np.std(rmsd1)),
                            'min': float(np.min(rmsd1)),
                            'max': float(np.max(rmsd1))
                        },
                        'plddt': {
                            'mean': float(np.mean(plddt)),
                            'std': float(np.std(plddt)),
                            'min': float(np.min(plddt)),
                            'max': float(np.max(plddt))
                        },
                        'n_predictions': len(rmsd1)
                    }
                    
                    if rmsd2 is not None:
                        recycle_stats['rmsd_ref2'] = {
                            'mean': float(np.mean(rmsd2)),
                            'std': float(np.std(rmsd2)),
                            'min': float(np.min(rmsd2)),
                            'max': float(np.max(rmsd2))
                        }
                    
                    model_summary['recycles'][recycle] = recycle_stats
                    model_summary['total_predictions'] += len(rmsd1)
                
                summary['models'][model] = model_summary
            
            # Calculate overall statistics
            if all_rmsd1:
                summary['overall']['rmsd_ref1'] = {
                    'mean': float(np.mean(all_rmsd1)),
                    'std': float(np.std(all_rmsd1)),
                    'min': float(np.min(all_rmsd1)),
                    'max': float(np.max(all_rmsd1))
                }
                
            if all_rmsd2:
                summary['overall']['rmsd_ref2'] = {
                    'mean': float(np.mean(all_rmsd2)),
                    'std': float(np.std(all_rmsd2)),
                    'min': float(np.min(all_rmsd2)),
                    'max': float(np.max(all_rmsd2))
                }
                
            if all_plddt:
                summary['overall']['plddt'] = {
                    'mean': float(np.mean(all_plddt)),
                    'std': float(np.std(all_plddt)),
                    'min': float(np.min(all_plddt)),
                    'max': float(np.max(all_plddt))
                }
                
            summary['overall']['total_predictions'] = len(all_rmsd1)
            
            return summary

    def get_all_recycle_data(
        self,
        position: int,
        model: Optional[int] = None,
        chunk_size: int = 1000
    ) -> Dict[int, Dict[int, RecycleData]]:
        """Get all recycle data with memory-efficient chunking."""
        with h5py.File(self.file_path, 'r') as f:
            results = {}
            
            # Get position group
            pos_path = f'pos_{position}'
            if pos_path not in f:
                return results
                
            pos_group = f[pos_path]
            
            # Handle model selection
            if model is not None:
                model_keys = [f'model_{model}']
            else:
                model_keys = [k for k in pos_group.keys() if k.startswith('model_')]
            
            for model_key in model_keys:
                if model_key not in pos_group:
                    continue
                    
                model_num = int(model_key.split('_')[1])
                model_group = pos_group[model_key]
                model_data = {}
                
                # Get all recycle numbers (they are direct keys in model group)
                recycle_keys = sorted([int(k) for k in model_group.keys()])
                
                for recycle in recycle_keys:
                    recycle_group = model_group[str(recycle)]
                    
                    # Read rmsd1 data
                    if 'rmsd_ref1' in recycle_group:
                        rmsd1_dset = recycle_group['rmsd_ref1']
                        if len(rmsd1_dset) > chunk_size:
                            rmsd1 = np.concatenate([rmsd1_dset[i:i+chunk_size] for i in range(0, len(rmsd1_dset), chunk_size)])
                        else:
                            rmsd1 = rmsd1_dset[:]
                    else:
                        continue  # Skip if no rmsd1 data
                    
                    # Handle rmsd2 data
                    rmsd2 = None
                    if 'rmsd_ref2' in recycle_group:
                        rmsd2_dset = recycle_group['rmsd_ref2']
                        if len(rmsd2_dset) > chunk_size:
                            rmsd2 = np.concatenate([rmsd2_dset[i:i+chunk_size] for i in range(0, len(rmsd2_dset), chunk_size)])
                        else:
                            rmsd2 = rmsd2_dset[:]
                    
                    # Handle plddt data
                    plddt = None
                    if 'plddt' in recycle_group:
                        plddt_dset = recycle_group['plddt']
                        if len(plddt_dset) > chunk_size:
                            plddt = np.concatenate([plddt_dset[i:i+chunk_size] for i in range(0, len(plddt_dset), chunk_size)])
                        else:
                            plddt = plddt_dset[:]
                    
                    # Handle seeds data
                    seeds = None
                    if 'seeds' in recycle_group:
                        seeds_dset = recycle_group['seeds']
                        if len(seeds_dset) > chunk_size:
                            seeds = np.concatenate([seeds_dset[i:i+chunk_size] for i in range(0, len(seeds_dset), chunk_size)])
                        else:
                            seeds = seeds_dset[:]
                    
                    model_data[recycle] = RecycleData(rmsd1, rmsd2, plddt, seeds)
                
                if model_data:
                    results[model_num] = model_data
            
            return results

    def get_positions(self) -> List[int]:
        """Get list of all positions stored in the analysis.
        
        Returns:
            List of position numbers in sorted order.
        """
        try:
            with h5py.File(self.file_path, 'r') as f:
                # Debug: Print file structure
                # logger.debug(f"HDF5 file structure for {self.file_path}:")
                def print_structure(name, obj):
                    # logger.debug(f"  {name}: {type(obj)}")
                    if isinstance(obj, h5py.Group):
                        for key in obj.keys():
                            # logger.debug(f"    - {key}")
                            pass
                f.visititems(print_structure)
                
                # Get all position keys and convert to integers
                # Look for positions under the root level
                positions = []
                for key in f.keys():
                    try:
                        # Check if the key represents a position (should be numeric)
                        if key.isdigit() or (key.startswith('-') and key[1:].isdigit()):
                            positions.append(int(key))
                        # Also check for 'pos_X' format
                        elif key.startswith('pos_'):
                            pos_num = key.split('_')[1]
                            if pos_num.isdigit():
                                positions.append(int(pos_num))
                    except ValueError:
                        logger.debug(f"Skipping non-position key: {key}")
                        continue
                
                # logger.debug(f"Found positions in HDF5: {positions}")
                return sorted(positions)
                
        except Exception as e:
            logger.warning(f"Failed to read positions from HDF5 file: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            # Fall back to memory store or positions set
            positions = self._memory_store['metadata']['positions'] if hasattr(self, '_memory_store') else self.positions
            logger.debug(f"Using fallback positions: {positions}")
            return sorted(list(positions))