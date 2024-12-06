#!/usr/bin/env python3
import argparse
import logging
import json
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

def generate_prediction_config(protein_config: dict, protein_name: str) -> dict:
    """Generate prediction configuration for a protein"""
    base_config = {
        "unified_memory": True,
        "parentPath": str(Path("results") / protein_name),
        "setupPath": str(Path("setup") / protein_name),
        "copies": 1,
        "msa_method": "mmseqs2",
        "pair_mode": "unpaired",
        "cov": 0,
        "id": 90,
        "qid": 0,
        "do_not_filter": False,
        "template_mode": "none",
        "pdb": "",
        "chain": "",
        "rm_template_seq": False,
        "propagate_to_copies": False,
        "do_not_align": False,
        "model_type": "monomer (ptm)",
        "rank_by": "plddt",
        "debug": False,
        "use_initial_guess": False,
        "num_msa": 512,
        "num_extra_msa": 1024,
        "use_cluster_profile": True,
        "model": "all",
        "num_recycles": 3,
        "recycle_early_stop_tolerance": 0.5,
        "select_best_across_recycles": True,
        "use_mlm": False,
        "use_dropout": True,
        "seed": 42,
        "num_seeds": 1,
        "show_images": True,
        "masking_mode": "off",
        "mask_msa": False,
        "mask_deletion_matrix": False,
        "cols": [],
        "cols_range": [],
        "mask_identity": "X",
        "overwrite": True
    }
    
    # Update with protein-specific settings
    protein_data = protein_config[protein_name]
    base_config.update({
        "sequence": protein_data["sequence"],
        "jobname": f"{protein_name}_prediction"
    })
    
    if "mutations" in protein_data:
        base_config["mutations"] = protein_data["mutations"]
    
    if "known_positions" in protein_data:
        base_config["cols"] = protein_data["known_positions"]
    
    return base_config

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
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="Force local execution (don't use SLURM even if available)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(Path("logs"))
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
            gpu_type="a30" if not args.force_local else None,
            gpu_count=1,
            partition=args.partition if not args.force_local else None,
            email=args.email if not args.force_local else None,
            container_path=args.container,
            script_path=args.script,
            schema_path=args.schema
        )
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate and save prediction configs
        for protein_name in ["i89", "her2", "rfah"]:
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
        run_i89_experiments(slurm_config, config)
        
        logger.info("Starting HER2 experiments")
        run_her2_experiments(slurm_config, config)
        
        logger.info("Starting RfaH experiments")
        run_rfah_experiments(slurm_config, config)
        
        logger.info("All experiments completed successfully")
        
    except Exception as e:
        logger.error(f"Error running experiments: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 