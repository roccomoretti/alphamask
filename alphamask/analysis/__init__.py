"""Analysis module for protein structure predictions."""

import logging
from rich.console import Console
from rich.status import Status
from contextlib import contextmanager

from .config import AnalysisConfig, create_analysis_config
from .rmsd import RMSDCalculator, RMSDConfig, RMSDResult
from .statistics import RMSDAnalyzer
from .visualization.rmsd_visualizer import RMSDVisualizer
from .storage import Storage
from .rmsd_analysis import RMSDAnalysis

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

__all__ = [
    'RMSDAnalysis',
    'AnalysisConfig', 
    'create_analysis_config',
    'RMSDCalculator',
    'RMSDConfig',
    'RMSDResult',
    'RMSDAnalyzer',
    'RMSDVisualizer',
    'Storage',
    'SimpleSpinner',
    'progress_status'
]
