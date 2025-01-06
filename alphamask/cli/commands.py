"""Command handlers for AlphaMask CLI"""

import multiprocessing
# Set multiprocessing start method to 'spawn' for JAX compatibility
multiprocessing.set_start_method('spawn', force=True)

import logging

# Configure logging first
logger = logging.getLogger("alphamask.cli.commands")



__all__ = [
    'setup_cmd',
    'submit_jobs_cmd',
    'help_cmd',
    'predict_job_cmd',
    'extract_pdbs_cmd',
    'status_cmd',
    'analyze_cmd',
    'recycle_analysis',
    'resubmit_incomplete'
]