"""Help command implementation."""

import textwrap
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from .utils import console
from ..utils.constants import HELP_TEXTS

def help_cmd(args):
    """Display help information for AlphaMask commands.

    This command shows detailed help information about AlphaMask commands
    and configuration. It can display general help or specific help for
    a particular topic.

    Args:
        args: Namespace object from argparse containing:
            - topic (str, optional): Specific topic to get help on
                Choices: ["setup", "run", "config"]
            - debug (bool): Enable debug logging
            - quiet (bool): Disable logging output
            - log_file (str, optional): Path to log file
    """
    if args.topic:
        if args.topic in HELP_TEXTS:
            console.print(Markdown(textwrap.dedent(HELP_TEXTS[args.topic])))
        else:
            console.print(f"No help available for topic: {args.topic}")
    else:
        console.print(Panel.fit(
            "AlphaMask: Protein Masking Analysis Tool\n\n"
            "Available commands:\n"
            "  setup  - Set up experiment directories\n"
            "  run    - Run protein experiments\n"
            "  help   - Show this help message\n\n"
            "For detailed help on a command:\n"
            "  alphamask help [command]\n\n"
            "For configuration help:\n"
            "  alphamask help config",
            title="AlphaMask Help"
        ))
