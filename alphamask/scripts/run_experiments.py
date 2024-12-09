#!/usr/bin/env python3
import argparse
import logging
import json
import os
from pathlib import Path

from alphamask.utils.slurm import SlurmJobConfig
from alphamask.experiments import (
    run_her2_experiments,
    run_rfah_experiments,
    run_i89_experiments,
    load_protein_config,
    validate_protein_config
)

def setup_logging(log_dir: Path):
    """Set up logging configuration"""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"experiments.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def generate_prediction_config(config: dict, protein_name: str) -> dict:
    """Generate prediction configuration for a protein"""
    protein_config = config[protein_name]
    return {
        "sequence": protein_config["sequence"],
        "mutations": protein_config.get("mutations", []),
        "known_positions": protein_config.get("known_positions", []),
        "frustra_positions": protein_config.get("frustra_positions", [])
    }

def main():
    parser = argparse.ArgumentParser(description="Run AlphaMask experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/proteins.yaml",
        help="Path to protein configuration file"
    )
    parser.add_argument(
        "--container", 
        type=str, 
        required=True,
        help="Path to Singularity container"
    )
    parser.add_argument(
        "--script", 
        type=str, 
        required=True,
        help="Path to prediction script"
    )
    parser.add_argument(
        "--schema", 
        type=str, 
        required=True,
        help="Path to JSON schema"
    )
    parser.add_argument(
        "--email", 
        type=str, 
        help="Email for job notifications"
    )
    parser.add_argument(
        "--partition", 
        type=str, 
        default="clara",
        help="SLURM partition"
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="rtx2080ti",
        help="GPU type to request (e.g., rtx2080ti)"
    )
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="Force local execution (don't use SLURM even if available)"
    )
    parser.add_argument(
        "--setup-base-path",
        type=str,
        default=os.path.expanduser("~/alphamask_setup"),
        help="Base path for setup files (default: ~/alphamask_setup)"
    )
    
    args = parser.parse_args()
    
    # Get base directory from config path
    base_dir = Path(args.config).parent.parent
    
    # Resolve schema path
    schema_path = Path(args.schema).resolve()
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
    
    # Set up logging
    setup_logging(base_dir / "logs")
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate configuration
        config = load_protein_config(args.config)
        validate_protein_config(config)
        
        # Configure SLURM/local settings
        slurm_config = SlurmJobConfig(
            time="24:00:00",
            memory="300000",
            cpus_per_task=1,
            gpu_type=args.gpu_type if not args.force_local else None,
            gpu_count=1,
            partition=args.partition if not args.force_local else None,
            email=args.email if not args.force_local else None,
            container_path=args.container,
            script_path=args.script,
            schema_path=str(schema_path),
            setup_base_path=args.setup_base_path
        )
        
        # Create results directory
        results_dir = base_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Generate and save prediction configs
        for protein_name in ["i89"]: #, "her2", "rfah"
            protein_dir = results_dir / protein_name
            protein_dir.mkdir(exist_ok=True)
            
            prediction_config = generate_prediction_config(config, protein_name)
            config_path = protein_dir / f"{protein_name}_config.json"
            
            with open(config_path, "w") as f:
                json.dump(prediction_config, f, indent=2)
        
        # Run experiments
        logger.info("Starting experiments in {} mode".format(
            "local" if args.force_local else "SLURM/local"
        ))
        
        logger.info("Starting I89 experiments")
        run_i89_experiments(slurm_config, config, base_dir)
        
        # logger.info("Starting HER2 experiments")
        # run_her2_experiments(slurm_config, config, base_dir)
        
        # logger.info("Starting RfaH experiments")
        # run_rfah_experiments(slurm_config, config, base_dir)
        
        logger.info("All experiments completed successfully")
        
    except Exception as e:
        logger.error(f"Error running experiments: {str(e)}", exc_info=True)
        raise

def run():
    """Entry point for console script"""
    main()

if __name__ == "__main__":
    main() 