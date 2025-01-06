"""Shared utilities for CLI commands."""

import logging
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)
console = Console()

def show_summary(success: bool, title: str, details: Dict[str, Any]) -> None:
    """Show command execution summary."""
    status = "✅ Success" if success else "❌ Failed"
    
    # Format details
    detail_lines = [f"{k}: {v}" for k, v in details.items()]
    
    console.print(Panel.fit(
        f"{status}\n\n" + "\n".join(detail_lines),
        title=title,
        border_style="green" if success else "red"
    ))

def ensure_directory(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
