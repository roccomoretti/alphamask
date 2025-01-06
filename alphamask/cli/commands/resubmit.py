import os
import glob
import logging
import subprocess
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("alphamask.cli.resubmit")

def find_experiment_dirs(base_path: Path) -> list[Path]:
    """Find all experiment directories containing scripts and potential PDB outputs."""
    experiment_dirs = []
    
    # For iterative experiments
    for protein_dir in base_path.glob("*_*"):  # Match protein dirs like rfah_94a41
        if not protein_dir.is_dir():
            continue
            
        # Check iterative experiments
        iterative_dir = protein_dir / "iterative"
        if iterative_dir.is_dir():
            # Look for WT and mutation directories
            for exp_dir in iterative_dir.glob("*"):  # WT, mutations like I89N, etc.
                if exp_dir.is_dir() and (exp_dir / "scripts").is_dir():
                    experiment_dirs.append(exp_dir)
        
        # Check apriori experiments
        apriori_dir = protein_dir / "apriori"
        if apriori_dir.is_dir():
            # Look into each experiment directory
            for exp_dir in apriori_dir.glob("*"):  # e.g., kortemme_et_al_I89S
                if not exp_dir.is_dir():
                    continue
                # Look into condition directories
                for condition_dir in exp_dir.glob("*"):  # e.g., masked_mutated
                    if condition_dir.is_dir() and (condition_dir / "scripts").is_dir():
                        experiment_dirs.append(condition_dir)
    
    return experiment_dirs

def print_summary_stats(stats: dict, args):
    """Print a detailed summary of jobs found and their status."""
    total_jobs = sum(len(jobs) for jobs in stats.values())
    total_scripts = 0
    total_pdbs = 0
    
    logger.info("\nEXPERIMENT SUMMARY:")
    logger.info("=" * 50)
    
    # Collect all experiment directories for counting
    for exp_dir in find_experiment_dirs(Path(args.path)):
        scripts_dir = exp_dir / "scripts"
        pdbs_dir = exp_dir / "out" / "pdbs"
        
        if scripts_dir.is_dir():
            total_scripts += len(list(scripts_dir.glob("*.sh")))
        if pdbs_dir.is_dir():
            total_pdbs += len(list(pdbs_dir.glob("*_best.pdb")))
    
    logger.info(f"Total script files found: {total_scripts}")
    logger.info(f"Total PDB files found: {total_pdbs}")
    logger.info(f"Total jobs needing resubmission: {total_jobs}")
    
    if total_jobs > 0:
        logger.info("\nBreakdown by protein:")
        logger.info("-" * 30)
        
        for protein, jobs in sorted(stats.items()):
            # Group by experiment type
            by_type = defaultdict(list)
            for job in jobs:
                by_type[job['type']].append(job)
            
            logger.info(f"\n{protein}:")
            for exp_type, type_jobs in sorted(by_type.items()):
                logger.info(f"  {exp_type.title()}: {len(type_jobs)} jobs")
                
                # Group by directory to show pattern
                by_dir = defaultdict(list)
                for job in type_jobs:
                    dir_name = job['script'].parent.parent.name
                    by_dir[dir_name].append(job)
                
                for dir_name, dir_jobs in sorted(by_dir.items()):
                    logger.info(f"    {dir_name}: {len(dir_jobs)} jobs")
                    if args.debug:
                        for job in dir_jobs:
                            logger.info(f"      - {job['script'].name}")
    
    if args.dry_run:
        logger.info("\nThis was a dry run. Run without --dry-run to resubmit these jobs.")
    
    return total_jobs

def resubmit_incomplete(args):
    """
    Check for missing *_best.pdb's in Apriori or Iterative experiments
    and re-submit them to 'paula' partition with 'a30' GPU.
    """
    base_path = Path(args.path)
    logger.info(f"Checking for incomplete jobs under: {base_path}")
    
    stats = defaultdict(list)
    
    experiment_dirs = find_experiment_dirs(base_path)
    logger.info(f"Found {len(experiment_dirs)} experiment directories")
    
    for exp_dir in experiment_dirs:
        logger.info(f"\nChecking experiment directory: {exp_dir}")
            
        scripts_dir = exp_dir / "scripts"
        pdbs_dir = exp_dir / "out" / "pdbs"
        
        if not scripts_dir.is_dir():
            logger.info(f"No scripts directory found at {scripts_dir}")
            continue
        if not pdbs_dir.is_dir():
            logger.info(f"No PDBs directory found at {pdbs_dir}")
            continue
            
        path_parts = exp_dir.parts
        exp_type = "iterative" if "iterative" in path_parts else "apriori"
        
        # Find protein directory (contains the hash, e.g., rfah_94a41)
        for part in path_parts:
            if '_' in part and any(c.isdigit() for c in part):
                protein = part
                break
        else:
            logger.warning(f"Could not determine protein name from path: {exp_dir}")
            continue
            
        logger.info(f"Processing {exp_type} experiment for protein {protein}")
        
        script_files = sorted(scripts_dir.glob("*.sh"))
        logger.info(f"Found {len(script_files)} script files")
        
        # Retrieve all existing "*_best.pdb" files for logging
        existing_pdbs = set(f.name for f in pdbs_dir.glob("*_best.pdb"))
        logger.info(f"Found {len(existing_pdbs)} existing PDB files")
        
        for script_path in script_files:
            if script_path.stem.endswith('_resub'):
                continue  # Skip resubmission scripts
            
            logger.info(f"\nChecking script: {script_path.name}")
            
            if exp_type == "iterative":
                script_stem = script_path.stem
                if "_pos_" in script_stem:
                    pos = script_stem.split("_pos_")[-1]
                    mutation_dir = exp_dir.name
                    if mutation_dir == "WT":
                        expected_pattern = f"{script_stem}*_mask_{pos}_id_X_best.pdb"
                    else:
                        expected_pattern = f"{script_stem}*_mask_{pos}_mut_{mutation_dir}_id_X_best.pdb"
                else:
                    expected_pattern = f"{script_stem}*_best.pdb"
            else:
                # Simplify for Apriori: one script => one best PDB
                script_stem = script_path.stem
                expected_pattern = f"{script_stem}*_best.pdb"
            
            logger.debug(f"Looking for PDB matching pattern: {expected_pattern}")
            logger.debug(f"In directory: {pdbs_dir}")
            logger.debug("Existing PDBs:")
            for pdb in pdbs_dir.glob("*_best.pdb"):
                logger.debug(f"  {pdb.name}")
            
            matching_pdbs = list(pdbs_dir.glob(expected_pattern))
            if not matching_pdbs:
                logger.info(f"No matching PDB found for pattern {expected_pattern}")
                stats[protein].append({
                    'type': exp_type,
                    'script': script_path,
                    'pattern': expected_pattern
                })
                
                if not args.dry_run:
                    logger.warning(f"Missing a best.pdb for script {script_path.name}. Will re-submit.")
                    updated_script = force_script_partition(script_path, partition="paula", gpu_type="a30")
                    logger.info(f"Submitting {updated_script}")
                    try:
                        subprocess.run(["sbatch", updated_script], check=True)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Failed to submit {updated_script}: {str(e)}")
            else:
                logger.info(f"Found {len(matching_pdbs)} matching PDB files")

    return print_summary_stats(stats, args)

def force_script_partition(script_path: Path, partition: str, gpu_type: str) -> Path:
    """
    Read the original script, overwrite partition/gpu constraints with new ones, 
    and write to a new script.
    """
    script_text = script_path.read_text()
    
    new_text = []
    for line in script_text.splitlines():
        if line.strip().startswith("#SBATCH --partition"):
            line = f"#SBATCH --partition={partition}"
        elif line.strip().startswith("#SBATCH --gres=gpu:"):
            line = f"#SBATCH --gres=gpu:{gpu_type}:1"
        elif line.strip().startswith("#SBATCH --exclude"):
            # Skip any existing exclude directive
            continue
        elif line.strip().startswith("#SBATCH --time="):
            # Skip any existing time directive
            continue
        elif line.strip().startswith("#SBATCH --mem="):
            # Skip any existing memory directive
            continue
        new_text.append(line)
    
    # Add exclude directive, time limit, and memory limit after partition specification
    partition_index = next(i for i, line in enumerate(new_text) 
                         if line.strip().startswith("#SBATCH --partition"))
    new_text.insert(partition_index + 1, "#SBATCH --exclude=paula06")
    new_text.insert(partition_index + 2, "#SBATCH --time=02:00:00")  # 2 hours
    new_text.insert(partition_index + 3, "#SBATCH --mem=10000")      # 10GB
    
    updated_script_path = script_path.parent / f"{script_path.stem}_resub.sh"
    with open(updated_script_path, 'w') as fout:
        fout.write("\n".join(new_text) + "\n")
    
    # Make it executable just in case
    updated_script_path.chmod(0o755)
    return updated_script_path 