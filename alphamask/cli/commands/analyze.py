"""Analysis command implementation."""

import logging
import yaml
from pathlib import Path
import traceback
import argparse
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict, Any

from colabdesign.af.contrib import predict

from ...analysis import RMSDAnalysis, create_analysis_config, AnalysisConfig
from .utils import show_summary, console, ensure_directory

logger = logging.getLogger(__name__)

def analyze_cmd(args):
    """Analyze protein structures and generate visualizations.
    
    Following the analysis pipeline:
    1. Load/calculate RMSD data
    2. Generate basic visualizations
    3. Generate recycle analysis if requested
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing protein structures...", total=None)
            
            # Load config
            logger.debug(f"Loading config: {args.config}")
            with open(args.config) as f:
                config = yaml.safe_load(f)
            
            # Get proteins to process
            proteins = args.proteins if args.proteins else list(config.get("proteins", {}).keys())
            if not proteins:
                raise ValueError("No proteins found in config file")
            
            success = True
            for protein_id in proteins:
                try:
                    logger.info(f"Processing protein: {protein_id}")
                    
                    # Setup protein configuration
                    protein_config = config['proteins'][protein_id]
                    protein_hash = predict.get_hash(protein_config['sequence'])[:5]
                    
                    # Create analysis configuration
                    analysis_config = _create_protein_analysis_config(protein_config)
                    
                    # Setup directories
                    protein_dir = Path(args.path) / f"{protein_id}_{protein_hash}"
                    analysis_dir = protein_dir / "analysis"
                    analysis_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each experiment type
                    for exp_type in ["iterative", "apriori"]:
                        success &= _process_experiment_type(
                            exp_type=exp_type,
                            protein_id=protein_id,
                            protein_dir=protein_dir,
                            analysis_dir=analysis_dir,
                            analysis_config=analysis_config,
                            protein_config=protein_config,
                            args=args
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to process protein {protein_id}: {str(e)}")
                    if args.debug:
                        logger.debug(traceback.format_exc())
                    success = False
                
                progress.update(task, advance=1)
            
            _show_analysis_summary(args, proteins, success)
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())
        raise


def _load_config(config_path: str) -> Dict:
    """Load configuration file."""
    logger.debug(f"Loading config: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)

def _get_proteins_to_process(config: Dict, specified_proteins: list = None) -> list:
    """Get list of proteins to process."""
    proteins = specified_proteins if specified_proteins else list(config.get("proteins", {}).keys())
    if not proteins:
        raise ValueError("No proteins found in config file")
    return proteins


def _create_protein_analysis_config(protein_config: dict) -> AnalysisConfig:
    """Create analysis configuration for a protein."""
    try:
        # Validate references
        references = protein_config['analysis']['references']
        ref1_path = Path(references['ref1']['path'])
        if not ref1_path.exists():
            raise FileNotFoundError(f"Reference 1 file not found: {ref1_path}")
        
        if 'ref2' in references:
            ref2_path = Path(references['ref2']['path'])
            if not ref2_path.exists():
                raise FileNotFoundError(f"Reference 2 file not found: {ref2_path}")
        
        return create_analysis_config({
            'references': references,
            'atom_selection': protein_config['analysis'].get('atom_selection', 'CA'),
            'regions': protein_config['analysis'].get('regions', []),
            'use_region': bool(protein_config['analysis'].get('regions', []))
        })
        
    except Exception as e:
        logger.error(f"Failed to create analysis config: {str(e)}")
        raise

def _process_experiment_type(
    exp_type: str,
    protein_id: str,
    protein_dir: Path,
    analysis_dir: Path,
    analysis_config: AnalysisConfig,
    protein_config: dict,
    args: argparse.Namespace
) -> bool:
    """Process a specific experiment type (iterative or apriori)."""
    exp_dir = protein_dir / exp_type
    if not exp_dir.exists():
        return True
        
    success = True
    logger.debug(f"Processing experiment type: {exp_type}")
    
    if exp_type == "iterative":
        success = _process_wt_analysis(
            protein_id=protein_id,
            wt_dir=exp_dir / "WT",
            analysis_dir=analysis_dir / "iterative" / "WT",
            analysis_config=analysis_config,
            args=args
        )
    elif exp_type == "apriori":
        success = _process_apriori_experiments(
            protein_id=protein_id,
            exp_dir=exp_dir,
            analysis_dir=analysis_dir / "apriori",
            analysis_config=analysis_config,
            experiments=protein_config.get('apriori_masking', {}).get('experiments', []),
            args=args
        )
    
    return success

def _process_wt_analysis(
    protein_id: str,
    wt_dir: Path,
    analysis_dir: Path,
    analysis_config: AnalysisConfig,
    args: argparse.Namespace
) -> bool:
    """Process wild-type analysis."""
    if not wt_dir.exists():
        return True
        
    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis pipeline
        pipeline = RMSDAnalysis(
            config=analysis_config,
            output_dir=analysis_dir,
            save_plots=not args.no_plots,
            plot_format=args.format,
            force=args.force,
            parallel=args.parallel,
            overwrite=args.overwrite
        )
        
        # Run RMSD analysis
        results = pipeline.run_analysis(
            exp_dir=wt_dir / "out" / "compressed",
            incremental=args.incremental,
            overwrite=args.overwrite
        )
        
        if results and not args.no_plots:
            _generate_visualizations(
                pipeline=pipeline,
                protein_id=protein_id,
                output_dir=analysis_dir,
                args=args
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed for {protein_id} WT: {str(e)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())
        return False

def _generate_visualizations(
    pipeline: RMSDAnalysis,
    protein_id: str,
    output_dir: Path,
    args: argparse.Namespace
) -> None:
    """Generate visualization plots."""
    try:
        positions = pipeline.storage.get_positions()
        
        # Generate basic plots for each position
        for pos in sorted(positions):
            pos_dir = output_dir / f"pos_{pos}"
            pos_dir.mkdir(parents=True, exist_ok=True)
            
            # Basic plots (always generated)
            pipeline.visualizer.create_plots(
                storage=pipeline.storage,
                position=pos,
                output_dir=pos_dir,
                format=args.format,
                show=False,
                interactive=True
            )
            
            # # Recycle analysis plots (if requested)
            # if args.recycle:
            #     recycle_dir = pos_dir / "recycle_analysis"
            #     recycle_dir.mkdir(parents=True, exist_ok=True)
            #     
            #     # Individual recycle landscapes
            #     pipeline.visualizer.create_recycle_landscapes(
            #         storage=pipeline.storage,
            #         position=pos,
            #         output_dir=recycle_dir,
            #         format=args.format,
            #         show=False
            #     )
            #     
            #     # Recycle progression plots
            #     pipeline.visualizer.create_recycle_progression(
            #         storage=pipeline.storage,
            #         position=pos,
            #         output_dir=recycle_dir,
            #         format=args.format,
            #         show=False
            #     )
            #     
            #     # Recycle summary plots
            #     pipeline.visualizer.create_recycle_summary(
            #         storage=pipeline.storage,
            #         position=pos,
            #         output_dir=recycle_dir,
            #         format=args.format,
            #         show=False
            #     )
            #     
            #     # Model comparison plots
            #     pipeline.visualizer.create_model_comparison(
            #         storage=pipeline.storage,
            #         position=pos,
            #         output_dir=recycle_dir,
            #         format=args.format,
            #         show=False
            #     )
            #     
            #     # Cumulative landscape plots
            #     pipeline.visualizer.create_cumulative_landscapes(
            #         storage=pipeline.storage,
            #         position=pos,
            #         output_dir=recycle_dir,
            #         format=args.format,
            #         show=False
            #     )
            #     
            #     # Summary collages
            #     pipeline.visualizer.create_summary_collages(
            #         storage=pipeline.storage,
            #         position=pos,
            #         output_dir=recycle_dir,
            #         format=args.format,
            #         show=False
            #     )
            #     
            #     # Model comparisons
            #     pipeline.visualizer.create_model_comparisons(
            #         storage=pipeline.storage,
            #         position=pos,
            #         output_dir=recycle_dir,
            #         format=args.format,
            #         show=False
            #     )
                
    except Exception as e:
        logger.error(f"Failed to generate plots for {protein_id}: {str(e)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(traceback.format_exc())

def _show_analysis_summary(args: argparse.Namespace, proteins: list, success: bool) -> None:
    """Show analysis summary."""
    show_summary(
        success=success,
        title="Analysis Complete",
        details={
            "Config File": args.config,
            "Base Path": args.path,
            "Proteins": ", ".join(proteins),
            "Plot Format": args.format if not args.no_plots else "disabled",
            "Parallel Jobs": args.parallel,
            "Mode": "Incremental" if args.incremental else "Full",
            "Recycle Analysis": "Enabled" if args.recycle else "Disabled",
            "Status": "Success" if success else "Some analyses failed"
        }
    )

def _process_apriori_experiments(
    protein_id: str,
    exp_dir: Path,
    analysis_dir: Path,
    analysis_config: AnalysisConfig,
    experiments: list,
    args: argparse.Namespace
) -> bool:
    """Process apriori masking experiments."""
    success = True
    
    for experiment in experiments:
        exp_name = experiment['name']
        exp_subdir = exp_dir / exp_name
        if exp_subdir.exists():
            try:
                exp_analysis_dir = analysis_dir / exp_name
                exp_analysis_dir.mkdir(parents=True, exist_ok=True)
                
                pipeline = RMSDAnalysis(
                    config=analysis_config,
                    output_dir=exp_analysis_dir,
                    save_plots=not args.no_plots,
                    plot_format=args.format,
                    force=args.force,
                    parallel=args.parallel,
                    overwrite=args.overwrite
                )
                
                results = pipeline.run_analysis(
                    exp_dir=exp_subdir / "out" / "compressed",
                    incremental=args.incremental,
                    overwrite=args.overwrite
                )
                
                if results and not args.no_plots:
                    _generate_visualizations(
                        pipeline=pipeline,
                        protein_id=protein_id,
                        output_dir=exp_analysis_dir,
                        args=args
                    )
                    
            except Exception as e:
                logger.error(f"Analysis failed for {protein_id} apriori {exp_name}: {str(e)}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())
                success = False
                
    return success
