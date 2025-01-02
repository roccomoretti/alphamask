from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger("alphamask.utils.partition")

@dataclass
class PartitionConfig:
    """Configuration for a SLURM partition"""
    name: str
    gpu_types: List[str]
    max_jobs: int = 1  # Default to one job per GPU type
    priority: int = 1
    sequence_length_range: Tuple[int, int] = (0, float('inf'))
    current_jobs: Dict[str, int] = None

    def __post_init__(self):
        if self.current_jobs is None:
            self.current_jobs = defaultdict(int)

    def can_accept_job(self, gpu_type: str) -> bool:
        """Check if partition can accept more jobs for specific GPU type"""
        return self.current_jobs[gpu_type] < self.max_jobs

    def is_suitable_for_sequence(self, seq_length: int) -> bool:
        """Check if partition is suitable for sequence length"""
        min_len, max_len = self.sequence_length_range
        return min_len <= seq_length <= max_len

class PartitionManager:
    """Manages job distribution across SLURM partitions"""
    
    def __init__(self, partitions: Optional[List[str]] = None, gpu_types: Optional[List[str]] = None):
        """
        Initialize partition manager with specified partitions and GPU types
        
        Args:
            partitions: List of partition names (e.g., ["clara", "paula"])
            gpu_types: List of GPU types (e.g., ["rtx2080ti", "v100", "a30"])
        """
        # Simple fixed assignments for different GPU types
        self.assignments = [
            ("clara", "v100"),
            ("paula", "a30"),
            ("clara", "rtx2080ti")
        ]
        self.current_index = 0
        self.partitions = {}
        
        # Initialize partition configs for tracking
        for partition, gpu_type in self.assignments:
            if partition not in self.partitions:
                self.partitions[partition] = PartitionConfig(
                    name=partition,
                    gpu_types=[gpu_type]
                )
            elif gpu_type not in self.partitions[partition].gpu_types:
                self.partitions[partition].gpu_types.append(gpu_type)
        
        logger.info("Initialized partition manager with configurations:")
        logger.info(self.get_usage_summary())

    def get_partition_for_sequence(self, sequence: str) -> Optional[Tuple[str, str]]:
        """
        Get the next partition and GPU type in sequence
        
        Returns:
            Tuple[str, str]: (partition_name, gpu_type)
        """
        partition_name, gpu_type = self.assignments[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.assignments)
        
        partition = self.partitions[partition_name]
        partition.current_jobs[gpu_type] += 1
        
        logger.info(f"Assigned sequence to {partition_name} partition using {gpu_type}")
        return partition_name, gpu_type

    def get_usage_summary(self) -> str:
        """Get a summary of current partition and GPU usage"""
        summary = ["Partition Usage:"]
        for partition_name, partition in self.partitions.items():
            summary.append(f"\n{partition_name}:")
            for gpu_type in partition.gpu_types:
                jobs = partition.current_jobs[gpu_type]
                summary.append(f"  {gpu_type}: {jobs}/{partition.max_jobs} jobs")
        return "\n".join(summary) 