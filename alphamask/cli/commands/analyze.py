"""Analysis command implementation."""

import logging
import yaml
from pathlib import Path
import traceback
import argparse
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Dict, Any
import numpy as np
from typing import Optional

from colabdesign.af.contrib import predict

from ...analysis import RMSDAnalysis, create_analysis_config, AnalysisConfig
from .utils import show_summary, console, ensure_directory
# Import create_com
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
                    
                    # Process each experiment type based on user selection
                    for exp_type in args.experiment_types:
                        logger.info(f"Processing {exp_type} experiments...")
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
    
    
    if exp_type == "apriori":
        success = _process_apriori_experiments(
            protein_id=protein_id,
            exp_dir=exp_dir,
            analysis_dir=analysis_dir / "apriori",
            analysis_config=analysis_config,
            experiments=protein_config.get('apriori_masking', {}).get('experiments', []),
            args=args
        )
    
    elif exp_type == "iterative":
        # Process WT first
        wt_success = _process_iterative_analysis(
            protein_id=protein_id,
            wt_dir=exp_dir / "WT",
            analysis_dir=analysis_dir / "iterative" / "WT",
            analysis_config=analysis_config,
            args=args
        )
        success &= wt_success
        
        # Process iterative mutations if enabled
        if protein_config.get('iterative_masking', {}).get('enabled', False):
            mutations = protein_config['iterative_masking'].get('mutations', [])
            for mutation_set in mutations:
                # Create mutation name (e.g., "T150A_L157R")
                mutation_name = "_".join(mutation_set)
                mutation_dir = exp_dir / mutation_name
                
                if mutation_dir.exists():
                    logger.info(f"Processing iterative mutation: {mutation_name}")
                    mut_success = _process_iterative_analysis(  # Reuse the same function for mutants
                        protein_id=protein_id,
                        wt_dir=mutation_dir,  # Use mutation directory instead of WT
                        analysis_dir=analysis_dir / "iterative" / mutation_name,
                        analysis_config=analysis_config,
                        args=args
                    )
                    success &= mut_success
                else:
                    logger.warning(f"Mutation directory not found: {mutation_dir}")
                    
 
    
    return success

def _process_iterative_analysis(
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
    args: argparse.Namespace,
    max_rmsd: Optional[float] = None
) -> None:
    """Generate visualization plots."""
    try:
        positions = pipeline.storage.get_positions()
        
        # Round to the nearest biggest integer
        max_rmsd = int(max_rmsd) + (1 if max_rmsd % 1 > 0 else 0)
        logger.info(f"Global max RMSD: {max_rmsd:.2f} Å")
        
        # Create directory for combined analyses
        combined_dir = output_dir / "combined_analysis"
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate combined landscape plot
        pipeline.visualizer.create_combined_landscape(
            storage=pipeline.storage,
            output_dir=combined_dir,
            format=args.format,
            show=False,
            max_rmsd=max_rmsd
        )
        logger.info("Generated combined landscape plot")
        
        # Generate breakdown plots for all positions
        pipeline.visualizer.create_combined_landscape_breakdown(
            storage=pipeline.storage,
            output_dir=combined_dir,
            format=args.format,
            show=False,
            max_rmsd=max_rmsd
        )
        logger.info("Generated combined landscape breakdown plots")
        
        # Only generate per-position plots if explicitly requested
        if args.per_position_plots:
            logger.info("Generating per-position plots")
            for pos in sorted(positions):
                pos_dir = output_dir / f"pos_{pos}"
                pos_dir.mkdir(parents=True, exist_ok=True)
                
                # Detailed plots
                pipeline.visualizer.create_plots(
                    storage=pipeline.storage,
                    position=pos,
                    output_dir=pos_dir,
                    format=args.format,
                    show=False,
                    interactive=True,
                    recycle=args.recycle,
                    max_rmsd=max_rmsd
                )
                logger.info(f"Generated plots for position {pos}")
                
                if args.recycle:
                    recycle_dir = pos_dir / "recycle_analysis"
                    recycle_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create recycle analysis plots
                    pipeline.visualizer.create_recycle_summary(
                        storage=pipeline.storage,
                        position=pos,
                        output_dir=recycle_dir,
                        format=args.format,
                        show=False,
                        max_rmsd=max_rmsd
                    )
                    
                    pipeline.visualizer.create_cumulative_landscapes(
                        storage=pipeline.storage,
                        position=pos,
                        output_dir=recycle_dir,
                        format=args.format,
                        show=False,
                        max_rmsd=max_rmsd
                    )
                    logger.info(f"Generated recycle analysis plots for position {pos}")
        else:
            logger.info("Skipping per-position plots (use --per-position-plots to generate them)")
                
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
    
    # Filter regions based on command line argument
    regions_to_process = analysis_config.regions
    if args.regions:
        regions_to_process = [r for r in analysis_config.regions if r.name in args.regions]
        if not regions_to_process:
            logger.warning(f"No matching regions found for {args.regions}. Available regions: {[r.name for r in analysis_config.regions]}")
            return False
    
    # Process each region separately
    for region in regions_to_process:
        logger.info(f"Processing region: {region.name} (residues {region.start}-{region.end})")
        
        # Create region-specific config
        region_config = AnalysisConfig(
            references=analysis_config.references,
            regions=[region],  # Only use this specific region
            atom_selection=analysis_config.atom_selection,
            statistical_tests=analysis_config.statistical_tests,
            plots=analysis_config.plots,
            export=analysis_config.export,
            debug=analysis_config.debug
        )
        
        # Create region-specific output directory
        region_analysis_dir = analysis_dir / region.name
        
        for experiment in experiments:
            exp_name = experiment['name']
            exp_subdir = exp_dir / exp_name
            if exp_subdir.exists():
                try:
                    exp_analysis_dir = region_analysis_dir / exp_name
                    exp_analysis_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each condition
                    conditions = [
                        ("unmasked_unmutated", "Unmasked, Unmutated"),
                        ("unmasked_mutated", "Unmasked, Mutated"),
                        ("masked_unmutated", "Masked, Unmutated"),
                        ("masked_mutated", "Masked, Mutated")
                    ]
                    
                    # Calculate global max RMSD across all conditions first
                    max_rmsd = 0
                    storages = {}
                    
                    # First pass: analyze and collect max RMSD
                    for condition_dir, condition_desc in conditions:
                        condition_path = exp_subdir / condition_dir
                        if condition_path.exists():
                            logger.info(f"Processing condition: {condition_desc}")
                            condition_analysis_dir = exp_analysis_dir / condition_dir
                            condition_analysis_dir.mkdir(parents=True, exist_ok=True)
                            
                            pipeline = RMSDAnalysis(
                                config=region_config,  # Use region-specific config
                                output_dir=condition_analysis_dir,
                                save_plots=not args.no_plots,
                                plot_format=args.format,
                                force=args.force,
                                parallel=args.parallel,
                                overwrite=args.overwrite
                            )
                            
                            # Run RMSD analysis
                            results = pipeline.run_analysis(
                                exp_dir=condition_path / "out" / "compressed",
                                incremental=args.incremental,
                                overwrite=args.overwrite
                            )
                            
                            if results:
                                storages[condition_desc] = pipeline.storage
                                # Update max_rmsd
                                positions = pipeline.storage.get_positions()
                                for pos in positions:
                                    recycle_data = pipeline.storage.get_all_recycle_data(pos)
                                    for model_data in recycle_data.values():
                                        for data in model_data.values():
                                            if data.rmsd_ref1 is not None:
                                                max_rmsd = max(max_rmsd, np.max(data.rmsd_ref1))
                                            if data.rmsd_ref2 is not None:
                                                max_rmsd = max(max_rmsd, np.max(data.rmsd_ref2))
                        else:
                            logger.debug(f"Condition directory not found: {condition_path}")
                    
                    # Round to the nearest biggest integer
                    max_rmsd = int(max_rmsd) + (1 if max_rmsd % 1 > 0 else 0)
                    logger.info(f"Global max RMSD across all conditions: {max_rmsd:.2f} Å")
                    
                    # Second pass: generate visualizations with consistent max_rmsd
                    for condition_dir, condition_desc in conditions:
                        condition_path = exp_subdir / condition_dir
                        if condition_path.exists() and condition_desc in storages:
                            condition_analysis_dir = exp_analysis_dir / condition_dir
                            if not args.no_plots:
                                _generate_visualizations(
                                    pipeline=pipeline,
                                    protein_id=protein_id,
                                    output_dir=condition_analysis_dir,
                                    args=args,
                                    max_rmsd=max_rmsd  # Pass consistent max_rmsd
                                )
                    
                    # Create summary landscape plot
                    if len(storages) > 1:
                        pipeline.visualizer.apriori_create_summary_landscape(
                            storages=storages,  # Pass all storages
                            output_dir=exp_analysis_dir,
                            format=args.format,
                            show=False,
                            max_rmsd=max_rmsd
                        )
                except Exception as e:
                    logger.error(f"Analysis failed for {protein_id} apriori {exp_name}: {str(e)}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(traceback.format_exc())
                    success = False
                    
    return success

def _process_iterative_experiments(
    protein_id: str,
    exp_dir: Path,
    analysis_dir: Path,
    analysis_config: AnalysisConfig,
    experiments: list,
    args: argparse.Namespace
) -> bool:
    """Process iterative masking experiments."""
    success = True
    
    # Calculate global max RMSD across all experiments first
    max_rmsd = 0
    storages = {}
    
    # First pass: analyze and collect max RMSD
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
                
                # Run RMSD analysis
                results = pipeline.run_analysis(
                    exp_dir=exp_subdir / "out" / "compressed",
                    incremental=args.incremental,
                    overwrite=args.overwrite
                )
                
                if results:
                    storages[exp_name] = pipeline.storage
                    # Update max_rmsd
                    positions = pipeline.storage.get_positions()
                    for pos in positions:
                        recycle_data = pipeline.storage.get_all_recycle_data(pos)
                        for model_data in recycle_data.values():
                            for data in model_data.values():
                                if data.rmsd_ref1 is not None:
                                    max_rmsd = max(max_rmsd, np.max(data.rmsd_ref1))
                                if data.rmsd_ref2 is not None:
                                    max_rmsd = max(max_rmsd, np.max(data.rmsd_ref2))
            except Exception as e:
                logger.error(f"Analysis failed for {protein_id} iterative {exp_name}: {str(e)}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())
                success = False
    
    # Round to the nearest biggest integer
    max_rmsd = int(max_rmsd) + (1 if max_rmsd % 1 > 0 else 0)
    logger.info(f"Global max RMSD across all experiments: {max_rmsd:.2f} Å")
    
    # Second pass: generate visualizations with consistent max_rmsd
    for experiment in experiments:
        exp_name = experiment['name']
        exp_subdir = exp_dir / exp_name
        if exp_subdir.exists() and exp_name in storages:
            try:
                exp_analysis_dir = analysis_dir / exp_name
                if not args.no_plots:
                    _generate_visualizations(
                        pipeline=pipeline,
                        protein_id=protein_id,
                        output_dir=exp_analysis_dir,
                        args=args,
                        max_rmsd=max_rmsd  # Pass consistent max_rmsd
                    )
            except Exception as e:
                logger.error(f"Visualization failed for {protein_id} iterative {exp_name}: {str(e)}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())
                success = False
    
    # Create summary landscape plot if we have multiple experiments
    if len(storages) > 1 and not args.no_plots:
        try:
            # Format experiment names for better display
            formatted_storages = {}
            for exp_name, storage in storages.items():
                # Extract mutation from experiment name (e.g., "I89S" from "kortemme_et_al_I89S")
                mutation = exp_name.split('_')[-1]
                formatted_storages[mutation] = storage
            
            pipeline.visualizer.iterative_create_summary_landscape(
                storages=formatted_storages,
                output_dir=analysis_dir,
                format=args.format,
                show=False,
                max_rmsd=max_rmsd
            )
            logger.info("Generated iterative summary landscape plot")
        except Exception as e:
            logger.error(f"Failed to generate iterative summary plot: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            success = False
    
    return success
