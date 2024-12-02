from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class MaskingStrategy(Enum):
    MASK_POSITIONS = "mask_positions"
    UNMASK_POSITIONS = "unmask_positions"
    ITERATIVE_SINGLE = "iterative_single"
    ITERATIVE_SINGLE_MASK_MUTATE = "iterative_single_mask_mutate"


@dataclass
class ExperimentConfig:
    sequence: str
    jobname_prefix: str
    parent_path: str
    masking_strategy: MaskingStrategy
    positions: Optional[List[int]] = None
    num_recycles: int = 12
    num_seeds: int = 12
    msa_method: str = "custom_a3m"
    custom_a3m_path: str = ""
    mutations: Optional[List[str]] = None
    run_control: bool = True
    run_only_control: bool = False
    unified_memory: bool = False


@dataclass
class ExperimentPaths:
    vanilla_jobname: Optional[str] = None
    vanilla_path: Optional[str] = None
    mask_mutate_jobname: Optional[str] = None
    mask_mutate_path: Optional[str] = None
    mask_jobname: Optional[str] = None
    mask_path: Optional[str] = None
