#!/usr/bin/env python3
import argparse
import logging
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
        default="paula",
        help="SLURM partition"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(Path("logs"))
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate configuration
        config = load_protein_config(args.config)
        validate_protein_config(config)
        
        # Configure SLURM settings
        slurm_config = SlurmJobConfig(
            time="24:00:00",
            memory="300000",
            cpus_per_task=1,
            gpu_type="a30",
            gpu_count=1,
            partition=args.partition,
            email=args.email,
            container_path=args.container,
            script_path=args.script,
            schema_path=args.schema
        )
        
        # Run experiments
        logger.info("Starting HER2 experiments")
        run_her2_experiments(slurm_config, config)
        
        logger.info("Starting RfaH experiments")
        run_rfah_experiments(slurm_config, config)
        
        logger.info("Starting I89 experiments")
        run_i89_experiments(slurm_config, config)
        
        logger.info("All experiments completed successfully")
        
    except Exception as e:
        logger.error(f"Error running experiments: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 