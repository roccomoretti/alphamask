"""Extract PDBs from compressed storage."""

import logging
import traceback
from pathlib import Path
import os
import textwrap
import yaml
import json
import pandas as pd
from typing import Dict
from datetime import datetime, timedelta

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.style import Style
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Group

from .generic import show_summary

logger = logging.getLogger(__name__)

console = Console()

def extract_pdbs_cmd(args):
    """Extract PDBs from compressed storage."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting PDBs...", total=None)
            
            # Load and parse YAML config
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            # Get proteins to process
            proteins = args.proteins if args.proteins else list(config.get("proteins", {}).keys())
            
            if not proteins:
                raise ValueError("No proteins found in config file")
            
            from alphamask.analysis.compressed import extract_pdbs_from_experiments
            
            success = extract_pdbs_from_experiments(
                config=config,
                protein_ids=proteins,
                models=args.models,
                seeds=args.seeds,
                recycles=args.recycles,
                best_only=args.best_only
            )
            
            progress.update(task, completed=True)
            
            show_summary(
                success=success,
                title="PDB Extraction Complete",
                details={
                    "Config File": args.config,
                    "Proteins": ", ".join(proteins),
                    "Models": ", ".join(args.models) if args.models else "All",
                    "Seeds": ", ".join(args.seeds) if args.seeds else "All",
                    "Recycles": ", ".join(args.recycles) if args.recycles else "All",
                    "Best Only": "Yes" if args.best_only else "No"
                }
            )
            
    except Exception as e:
        logger.error(f"PDB extraction failed: {str(e)}")
        show_summary(
            success=False,
            title="PDB Extraction Failed",
            details={
                "Error": str(e),
                "Config File": args.config
            }
        )
        raise
