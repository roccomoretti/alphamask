"""AlphaMask: A tool for protein masking analysis"""

from .utils.logging import logger

# Import submodules
from .experiments import *
from .utils import *
from .cli import *

__version__ = "0.1.0"