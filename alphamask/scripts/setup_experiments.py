#!/usr/bin/env python3
import argparse
import shutil
import sys
import os
from pathlib import Path

def setup_experiment_directories(base_path: str, setup_path: str, force: bool = False):
    """Set up experiment directory structure"""
    base_dir = Path(base_path).resolve()
    setup_base = Path(setup_path).resolve()
    
    # Check if directory already contains experiment files
    if base_dir.exists() and not force:
        existing_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if existing_dirs:
            print(f"Warning: {base_dir} already contains directories.")
            print("Use --force to overwrite existing setup.")
            sys.exit(1)
    
    # Create main directories
    dirs = [
        "config",
        "logs",
        "schema",
        #"results/her2",
        #"results/rfah",
        #"results/i89",
        "scripts",
        # Add protein-specific config and script directories
        #"HER2/controls/vanilla/configs",
        #"HER2/controls/vanilla/scripts",
        #"RfaH/controls/vanilla/configs",
        #"RfaH/controls/vanilla/scripts", 
        "I89/controls/vanilla/configs",
        "I89/controls/vanilla/scripts"
    ]
    
    print(f"Setting up experiment directories in {base_dir}")
    
    for dir_path in dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")
    
    # Create shared setup directory
    print(f"\nSetting up shared setup directory in {setup_base}")
    setup_base.mkdir(parents=True, exist_ok=True)
    print(f"Created setup directory: {setup_base}")
    
    # Copy config templates
    pkg_config = Path(__file__).parent.parent / "config"
    
    config_files = {
        "test.yaml": "config/test.yaml",
        "schema_validation.json": "schema/schema_validation.json"
    }
    
    for src_file, dst_path in config_files.items():
        src_path = pkg_config / src_file
        if src_path.exists():
            dst_full_path = base_dir / dst_path
            shutil.copy(src_path, dst_full_path)
            print(f"Copied {src_file} to {dst_full_path}")
        else:
            print(f"Warning: Could not find {src_file} in package config")

def main():
    parser = argparse.ArgumentParser(
        description="Set up AlphaMask experiment directories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default=".",
        help="Base path for experiment setup"
    )
    parser.add_argument(
        "--setup-path",
        type=str,
        default=os.path.expanduser("~/alphamask_setup"),
        help="Base path for setup files (default: ~/alphamask_setup)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force setup even if directories exist"
    )
    
    args = parser.parse_args()
    
    try:
        setup_experiment_directories(args.path, args.setup_path, args.force)
        print("\nExperiment directories setup complete!")
        print("\nNext steps:")
        print("1. Review and modify config/proteins.yaml as needed")
        print("2. Run experiments using run_experiments.py")
    except Exception as e:
        print(f"\nError setting up experiment directories: {str(e)}", file=sys.stderr)
        sys.exit(1)

def run():
    """Entry point for console script"""
    main()

if __name__ == "__main__":
    main() 