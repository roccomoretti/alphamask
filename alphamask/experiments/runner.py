from typing import List, Optional
from pathlib import Path
import logging
import yaml

from .base import ProteinSystem
from .masking import IterativeMaskingExperiment, AprioriMaskingExperiment
from .mutations import MutationExperiment, DoubleMutationExperiment
from .frustra import FrustraMaskingExperiment
from ..utils.slurm import SlurmJobConfig

logger = logging.getLogger(__name__)

def load_protein_config(config_path: str = "config/proteins.yaml") -> dict:
    """Load protein configurations from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_protein_config(config: dict) -> None:
    """Validate protein configuration"""
    required_fields = {
        # 'her2': ['sequence', 'mutations', 'known_positions'],
        # 'rfah': ['sequence', 'mutations', 'known_positions', 'frustra_positions'],
        'i89': ['sequence', 'known_positions', 'frustra_positions']
    }
    
    for protein, fields in required_fields.items():
        if protein not in config:
            raise ValueError(f"Missing configuration for {protein}")
            
        protein_config = config[protein]
        for field in fields:
            if field not in protein_config:
                raise ValueError(f"Missing {field} in {protein} configuration")

def run_her2_experiments(slurm_config: Optional[SlurmJobConfig] = None, config: Optional[dict] = None, base_dir: Optional[Path] = None):
    """Run all experiments for HER2"""
    if config is None:
        config = load_protein_config()
    
    her2_config = config['her2']
    her2 = ProteinSystem(
        name="HER2",
        sequence=her2_config['sequence'],
        mutations=her2_config['mutations'],
        known_positions=her2_config['known_positions'],
        parent_path=str(base_dir) if base_dir else "."
    )
    
    experiments = [
        IterativeMaskingExperiment(her2, slurm_config),
        AprioriMaskingExperiment(her2, slurm_config),
        DoubleMutationExperiment(her2, her2.mutations, slurm_config)
    ]
    
    for experiment in experiments:
        logger.info(f"Running {experiment.name} for HER2")
        if not experiment.run():
            logger.error(f"Failed to run {experiment.name} for HER2")

def run_rfah_experiments(slurm_config: Optional[SlurmJobConfig] = None, config: Optional[dict] = None, base_dir: Optional[Path] = None):
    """Run all experiments for RfaH"""
    if config is None:
        config = load_protein_config()
    
    rfah_config = config['rfah']
    rfah = ProteinSystem(
        name="RfaH",
        sequence=rfah_config['sequence'],
        mutations=rfah_config['mutations'],
        known_positions=rfah_config['known_positions'],
        frustra_positions=rfah_config['frustra_positions'],
        parent_path=str(base_dir) if base_dir else "."
    )
    
    experiments = [
        IterativeMaskingExperiment(rfah, slurm_config),
        AprioriMaskingExperiment(rfah, slurm_config),
        FrustraMaskingExperiment(rfah, slurm_config),
        MutationExperiment(rfah, ["K142N"], slurm_config),
        MutationExperiment(rfah, ["R146H"], slurm_config),
        DoubleMutationExperiment(rfah, rfah.mutations, slurm_config)
    ]
    
    for experiment in experiments:
        logger.info(f"Running {experiment.name} for RfaH")
        if not experiment.run():
            logger.error(f"Failed to run {experiment.name} for RfaH")

def run_i89_experiments(slurm_config: Optional[SlurmJobConfig] = None, config: Optional[dict] = None, base_dir: Optional[Path] = None):
    """Run all experiments for I89"""
    if config is None:
        config = load_protein_config()
    
    i89_config = config['i89']
    i89 = ProteinSystem(
        name="I89",
        sequence=i89_config['sequence'],
        known_positions=i89_config['known_positions'],
        frustra_positions=i89_config['frustra_positions'],
        parent_path=str(base_dir) if base_dir else "."
    )
    
    experiments = [
        IterativeMaskingExperiment(i89, slurm_config),
        AprioriMaskingExperiment(i89, slurm_config),
        FrustraMaskingExperiment(i89, slurm_config)
    ]
    
    for experiment in experiments:
        logger.info(f"Running {experiment.name} for I89")
        if not experiment.run():
            logger.error(f"Failed to run {experiment.name} for I89") 