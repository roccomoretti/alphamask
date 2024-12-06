import logging
from IPython.display import display, HTML


class NotebookHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        color = {
            "DEBUG": "grey",
            "INFO": "blue",
            "WARNING": "orange",
            "ERROR": "red",
            "CRITICAL": "red",
        }.get(record.levelname, "black")
        display(HTML(f'<pre style="color: {color};">{log_entry}</pre>'))


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    notebook_handler = NotebookHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    notebook_handler.setFormatter(formatter)
    logger.addHandler(notebook_handler)
    return logger
