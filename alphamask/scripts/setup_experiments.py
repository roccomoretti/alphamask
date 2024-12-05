#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

def setup_experiment_directories(base_path: str):
    """Set up experiment directory structure"""
    base_dir = Path(base_path)
    
    # Create main directories
    dirs = [
        "config",
        "logs",
        "results/her2",
        "results/rfah",
        "results/i89",
        "scripts"
    ]
    
    for dir_path in dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Copy config template
    pkg_config = Path(__file__).parent.parent / "config" / "proteins.yaml"
    if pkg_config.exists():
        shutil.copy(pkg_config, base_dir / "config" / "proteins.yaml")

def main():
    parser = argparse.ArgumentParser(description="Set up AlphaMask experiment directories")
    parser.add_argument(
        "--path", 
        type=str, 
        default=".",
        help="Base path for experiment setup"
    )
    
    args = parser.parse_args()
    setup_experiment_directories(args.path)
    print(f"Experiment directories set up in {args.path}")

if __name__ == "__main__":
    main() 