import logging
from typing import Optional
import sys

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
