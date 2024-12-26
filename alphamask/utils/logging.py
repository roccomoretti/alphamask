import logging
import sys
from typing import Optional
from pathlib import Path

def setup_package_logging():
    """Configure package-wide logging"""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create and return package logger
    logger = logging.getLogger("alphamask")
    logger.setLevel(logging.INFO)
    return logger

def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Set up and return a logger with the package configuration.
    
    Args:
        name: Optional name for the logger. If None, uses __name__
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    return logger

def setup_logging(log_dir: Optional[Path] = None, debug: bool = False, disable: bool = False) -> None:
    """Set up logging configuration for the application."""
    if disable:
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        return

    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.StreamHandler()]
    
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_dir / "alphamask.log")))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Initialize package logging
logger = setup_package_logging()
