import logging
from typing import Optional
import sys
from pathlib import Path

def is_notebook() -> bool:
    """Check if we are running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal IPython
            return False
        else:
            return False
    except NameError:  # Probably standard Python interpreter
        return False

class NotebookHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.is_notebook_env = is_notebook()
        if not self.is_notebook_env:
            # Create a stream handler for command line output
            self.stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            self.stream_handler.setFormatter(formatter)

    def emit(self, record):
        log_entry = self.format(record)
        if self.is_notebook_env:
            try:
                from IPython.display import display, HTML
                color = {
                    "DEBUG": "grey",
                    "INFO": "blue",
                    "WARNING": "orange",
                    "ERROR": "red",
                    "CRITICAL": "red",
                }.get(record.levelname, "black")
                display(HTML(f'<pre style="color: {color};">{log_entry}</pre>'))
            except ImportError:
                print(log_entry)
        else:
            # Use standard stream handler for command line
            self.stream_handler.emit(record)

def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Set up and return a logger that works in both notebook and command line environments.
    
    Args:
        name: Optional name for the logger. If None, uses __name__
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add our custom handler
    notebook_handler = NotebookHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    notebook_handler.setFormatter(formatter)
    logger.addHandler(notebook_handler)
    
    return logger

def setup_logging(log_dir: Optional[Path] = None, debug: bool = False, disable: bool = False) -> None:
    """Set up logging configuration for the application.
    
    Args:
        log_dir: Optional directory for log files. If provided, logs will be written to a file
                in this directory in addition to console output.
        debug: If True, sets logging level to DEBUG, otherwise INFO.
        disable: If True, disables all logging output.
    """
    if disable:
        # Disable all logging
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        return

    # Create log directory if it doesn't exist
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "alphamask.log"
    else:
        log_file = None
    
    # Set up handlers
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(str(log_file)))
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add configured handlers
    for handler in handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    # Log initial setup
    if not disable:
        logging.info("Logging system initialized")
        if debug:
            logging.debug("Debug logging enabled")
