import yaml
from pathlib import Path
from datetime import datetime, timedelta
from rich.table import Table
from rich.console import Console
import logging
logger = logging.getLogger(__name__)

from typing import Dict, List, Tuple
import subprocess
from rich.layout import Layout

def get_protein_jobs(config_path):
    """Get job counts per protein"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    protein_jobs = {}
    for protein_id, protein_config in config.get('proteins', {}).items():
        total = 0
        sequence_length = len(protein_config['sequence'])
        
        # Count iterative masking jobs
        if protein_config.get('iterative_masking', {}).get('enabled', False):
            # Each position for WT
            total += sequence_length
            # Each position for each mutation
            mutations = protein_config['iterative_masking'].get('mutations', [])
            total += len(mutations) * sequence_length
        
        # Count apriori masking jobs
        if protein_config.get('apriori_masking', {}).get('enabled', False):
            experiments = protein_config['apriori_masking'].get('experiments', [])
            for exp in experiments:
                conditions = exp.get('conditions', [])
                total += len(conditions)
                
        protein_jobs[protein_id] = total
    
    return protein_jobs

def get_completed_jobs_by_protein(base_path: str, config_path: str) -> Dict[str, int]:
    """
    Get completed jobs count per protein by checking recursively for
    .h5 or .npz files in 'out/compressed' and 'predictions' directories.
    Takes into account that the on-disk folder might include a suffix like "her2_39b69"
    whereas the config protein key might just be "her2".
    """
    import yaml
    
    base = Path(base_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # All known protein IDs from the config
    valid_ids = list(config.get("proteins", {}).keys())
    
    # Initialize a dict with zero counts for each protein ID
    completed_counts = {pid: 0 for pid in valid_ids}
    
    def count_prediction_files(path: Path) -> int:
        """Returns count of .h5 or .npz files in directory"""
        h5_count = len(list(path.glob("*.h5")))
        npz_count = len(list(path.glob("*.npz")))
        # We expect pairs of .h5 and .npz files, so divide by 2
        return (h5_count + npz_count) // 2
    
    # For each directory in base_path, see if the name starts with a known protein ID
    for directory in base.iterdir():
        if not directory.is_dir():
            continue
        
        # Does this directory name match (or start with) any known protein ID?
        matched_id = None
        for pid in valid_ids:
            if directory.name.startswith(pid):
                matched_id = pid
                break
        
        if not matched_id:
            continue
            
        # Now search inside this directory for completed jobs
        for compressed_dir in directory.rglob("out/compressed"):
            if compressed_dir.is_dir():
                completed_counts[matched_id] += count_prediction_files(compressed_dir)
        
        for preds_dir in directory.rglob("predictions"):
            if preds_dir.is_dir():
                completed_counts[matched_id] += count_prediction_files(preds_dir)
    
    return completed_counts

def get_job_runtime_from_logs(job_name: str, protein_dir: Path) -> float:
    """
    Get actual runtime in minutes for a completed job by parsing its log file.
    Returns None if log file cannot be parsed or job didn't complete.
    """
    log_file = protein_dir / "logs" / f"{job_name}.err"
    if not log_file.exists():
        return None
        
    try:
        # Read first and last timestamps from log
        with open(log_file) as f:
            lines = f.readlines()
            
            # Check if job actually completed by looking for final save line
            completed = False
            for line in reversed(lines):
                if "INFO - Saving all atom data to" in line:
                    completed = True
                    break
            
            if not completed:
                return None
            
            # Find first timestamp
            for line in lines:
                if " - " in line:  # Look for timestamp separator
                    start_time = datetime.strptime(line.split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f")
                    break
            
            # Find last timestamp (search in reverse)
            for line in reversed(lines):
                if " - " in line:
                    end_time = datetime.strptime(line.split(" - ")[0], "%Y-%m-%d %H:%M:%S,%f")
                    break
                    
            runtime = (end_time - start_time).total_seconds() / 60  # Convert to minutes
            return runtime
            
    except Exception as e:
        logger.debug(f"Could not parse runtime from log {log_file}: {e}")
        return None

def get_average_runtime(protein_id: str, base_path: str) -> float:
    """Get average runtime of completed jobs for a protein"""
    protein_dir = None
    base = Path(base_path)
    
    # Find protein directory (handle cases like her2_39b69)
    for directory in base.iterdir():
        if directory.is_dir() and directory.name.startswith(protein_id):
            protein_dir = directory
            break
    
    if not protein_dir:
        return None
        
    # Get runtimes from completed job logs
    runtimes = []
    for log_file in (protein_dir / "iterative/WT/logs").glob("*.err"):
        job_name = log_file.stem
        runtime = get_job_runtime_from_logs(job_name, protein_dir / "iterative/WT")
        if runtime:
            runtimes.append(runtime)
    
    return sum(runtimes) / len(runtimes) if runtimes else None

def estimate_completion_time(completed: int, total: int, running: int, protein_id: str, base_path: str) -> str:
    """
    Estimate completion time based on average runtime times remaining jobs,
    divided by number of running jobs (available GPUs)
    """
    if running == 0 or total == completed:
        return "N/A"
    
    # Get average runtime from completed jobs
    avg_runtime = get_average_runtime(protein_id, base_path)
    if not avg_runtime:
        return "N/A"
    
    # Calculate total time needed for remaining jobs
    remaining_jobs = total - completed
    total_minutes = (remaining_jobs * avg_runtime) / running  # Divide by number of running jobs
    
    est_completion = datetime.now() + timedelta(minutes=total_minutes)
    return est_completion.strftime('%Y-%m-%d %H:%M:%S')

def create_stats_table(total_proteins, completed_jobs, running_jobs, pending_jobs, failed_jobs=0, 
                      protein_progress=None, jobs=None, args=None):
    """Create the statistics table with per-protein progress and estimates"""
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    
    # Overall progress
    stats_table.add_row("[bold]Total Jobs[/bold]", str(total_proteins))
    stats_table.add_row("[bold]Completed[/bold]", f"[green]{completed_jobs}[/green]")
    stats_table.add_row("[bold]Running[/bold]", f"[yellow]{running_jobs}[/yellow]")
    stats_table.add_row("[bold]Pending[/bold]", f"[blue]{pending_jobs}[/blue]")
    
    if failed_jobs > 0:
        stats_table.add_row("[bold]Failed[/bold]", f"[red]{failed_jobs}[/red]")
    
    # Overall completion percentage
    if total_proteins > 0:
        percentage = (completed_jobs / total_proteins) * 100
        stats_table.add_row(
            "[bold]Overall Progress[/bold]",
            f"[cyan]{percentage:.1f}%[/cyan]"
        )
    
    # Add separator
    stats_table.add_row("", "")
    stats_table.add_row("[bold]Per-Protein Progress[/bold]", "")
    
    # Add per-protein progress with estimates
    if protein_progress and jobs and args:
        for protein_id, (completed, total) in protein_progress.items():
            if total > 0:
                percentage = (completed / total) * 100
                progress_bar = "━" * int(percentage/5) + "─" * (20 - int(percentage/5))
                
                # Calculate running jobs for this protein
                protein_running = 0
                for job in jobs:
                    if (job['state'] == 'RUNNING' and 
                        (job['name'].startswith('WT_') or 
                         job['name'].startswith(f"{protein_id}_") or
                         any(mut.startswith(protein_id) for mut in job['name'].split('_')))):
                        protein_running += 1
                
                # Get average runtime and estimate only if we have completed jobs
                status_line = f"{progress_bar} [cyan]{percentage:.1f}%[/cyan] ({completed}/{total})"
                if completed > 0:  # Only show runtime stats if we have completed jobs
                    avg_runtime = get_average_runtime(protein_id, args.path)
                    if avg_runtime:  # Only show if we got valid runtime data
                        avg_time_str = f"{avg_runtime:.1f}m"
                        est_time = estimate_completion_time(
                            completed, total, protein_running,
                            protein_id, args.path
                        )
                        status_line += f"\n   Avg: {avg_time_str} | Est: {est_time}"
                
                stats_table.add_row(
                    f"[bold]{protein_id}[/bold]",
                    status_line
                )
    
    return stats_table

def create_jobs_table(jobs):
    """Create the active jobs table"""
    # Sort jobs: RUNNING first, then PENDING, then others
    def job_sort_key(job):
        state_order = {
            'RUNNING': 0,
            'PENDING': 1,
            'FAILED': 2,
            'COMPLETED': 3
        }
        return (state_order.get(job['state'], 99), job['id'])
    
    sorted_jobs = sorted(jobs, key=job_sort_key)
    
    active_jobs_table = Table(
        "Job ID",
        "Name",
        "State",
        "Runtime",
        "Reason",
        title="Active Jobs",
        expand=True,
        show_header=True,
        header_style="bold blue"
    )
    
    # Set column widths and justify
    active_jobs_table.columns[0].width = 10  # Job ID
    active_jobs_table.columns[1].width = 30  # Name
    active_jobs_table.columns[2].width = 10  # State
    active_jobs_table.columns[3].width = 10  # Runtime
    active_jobs_table.columns[4].width = 20  # Reason
    
    # Set column justify
    active_jobs_table.columns[0].justify = "right"
    active_jobs_table.columns[1].justify = "left"
    active_jobs_table.columns[2].justify = "center"
    active_jobs_table.columns[3].justify = "right"
    active_jobs_table.columns[4].justify = "left"
    
    for job in sorted_jobs:
        state_color = {
            'RUNNING': 'green',
            'PENDING': 'yellow',
            'FAILED': 'red',
            'COMPLETED': 'blue'
        }.get(job['state'], 'white')
        
        # Truncate job name if too long
        name = job['name']
        if len(name) > 27:
            name = name[:24] + "..."
        
        active_jobs_table.add_row(
            job['id'],
            name,
            f"[{state_color}]{job['state']}[/{state_color}]",
            job['runtime'],
            job['reason'][:17] + "..." if len(job['reason']) > 20 else job['reason']
        )
    return active_jobs_table

def get_slurm_jobs() -> List[Dict]:
    """
    Get SLURM job information for current user.
    
    Returns:
        List of dictionaries containing job information with keys:
        id, name, state, runtime, timelimit, reason
    """
    # Only get current jobs from squeue
    cmd = ["squeue", "--me", "--format=%i|%j|%T|%M|%l|%R"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    jobs = []
    
    # Process current jobs
    for line in result.stdout.strip().split('\n'):
        if '|' in line:
            job_id, name, state, runtime, timelimit, reason = line.split('|')
            # Skip header line
            if job_id == "JOBID":
                continue
            # Skip batch jobs
            if name.endswith('.batch'):
                continue
                
            jobs.append({
                'id': job_id,
                'name': name,
                'state': state,
                'runtime': runtime,
                'timelimit': timelimit,
                'reason': reason
            })
    
    return jobs

def get_job_counts(jobs: List[Dict]) -> Tuple[int, int, int, int]:
    """
    Get accurate counts of jobs in different states.
    
    Args:
        jobs: List of job dictionaries from get_slurm_jobs()
        
    Returns:
        Tuple of (running, pending, completed, failed) counts
    """
    running = 0
    pending = 0
    completed = 0
    failed = 0
    
    # Count jobs by state
    for job in jobs:
        state = job['state']
        if state == 'RUNNING':
            running += 1
        elif state == 'PENDING':
            pending += 1
        elif state == 'COMPLETED':
            completed += 1
        elif state == 'FAILED':
            failed += 1
    
    return running, pending, completed, failed

def parse_config_for_total_jobs(config_path: str) -> int:
    """
    Parse protein config to get total expected jobs.
    
    Args:
        config_path: Path to protein configuration YAML file
        
    Returns:
        Total number of expected jobs across all proteins
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    total_jobs = 0
    for protein_id, protein_config in config.get('proteins', {}).items():
        sequence_length = len(protein_config['sequence'])
        
        # Count iterative masking jobs
        if protein_config.get('iterative_masking', {}).get('enabled', False):
            # Each position for WT
            total_jobs += sequence_length
            
            # Each position for each mutation
            mutations = protein_config['iterative_masking'].get('mutations', [])
            total_jobs += len(mutations) * sequence_length
        
        # Count apriori masking jobs
        if protein_config.get('apriori_masking', {}).get('enabled', False):
            experiments = protein_config['apriori_masking'].get('experiments', [])
            for exp in experiments:
                conditions = exp.get('conditions', [])
                total_jobs += len(conditions)
        
        # Count frustra masking jobs
        if protein_config.get('frustra_masking', {}).get('enabled', False):
            total_jobs += 1  # One job per protein for frustra masking
    
    logger.debug(f"Total expected jobs: {total_jobs}")
    return total_jobs

def get_completed_jobs_count(base_path: str) -> int:
    """
    Get number of completed jobs by checking output directories recursively.
    
    Args:
        base_path: Base directory path to search
        
    Returns:
        Count of completed jobs based on output files
    """
    base_path = Path(base_path)
    completed = 0
    
    def count_prediction_files(path: Path) -> int:
        """Returns count of .h5 or .npz files in directory"""
        h5_count = len(list(path.glob("*.h5")))
        npz_count = len(list(path.glob("*.npz")))
        # We expect pairs of .h5 and .npz files, so divide by 2
        return (h5_count + npz_count) // 2
    
    # Check all experiment directories recursively
    for protein_dir in base_path.iterdir():
        if not protein_dir.is_dir():
            continue
        
        # Look in iterative/WT/out/compressed and other experiment directories
        for compressed_dir in protein_dir.rglob("out/compressed"):
            if compressed_dir.is_dir():
                completed += count_prediction_files(compressed_dir)
        
        # Also check apriori masking results
        for compressed_dir in protein_dir.rglob("**/apriori/**/compressed"):
            if compressed_dir.is_dir():
                completed += count_prediction_files(compressed_dir)
    
    logger.debug(f"Total completed jobs: {completed}")
    return completed

def create_status_layout():
    """Create the layout for status display"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    return layout
