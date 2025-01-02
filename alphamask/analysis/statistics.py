"""
Statistical analysis functionality for RMSD results.
"""
from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import logging
import json
from scipy import stats
from .rmsd import RMSDResult
import traceback

logger = logging.getLogger(__name__)

@dataclass
class RMSDStatistics:
    """Container for RMSD statistical analysis results."""
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    iqr: float
    structure_name: str
    reference: str  # 'ref1' or 'ref2'
    plddt_correlation: Optional[float] = None
    significance_tests: Optional[Dict[str, Any]] = None

class RMSDAnalyzer:
    """
    Analyzes RMSD calculation results and generates statistics.
    """
    
    def calculate_statistics(
        self,
        results: List[RMSDResult],
        structure_name: str,
        reference: str = "ref1",
        include_significance: bool = True
    ) -> RMSDStatistics:
        """
        Calculate comprehensive statistics for a set of RMSD results.
        
        Args:
            results: List of RMSD results
            structure_name: Name of the structure being analyzed
            reference: Which reference structure to analyze ('ref1' or 'ref2')
            include_significance: Whether to include significance tests
            
        Returns:
            Statistical analysis results
            
        Raises:
            ValueError: If no valid RMSD values found
        """
        if not results:
            raise ValueError("No results provided for analysis")
            
        # Extract RMSD values
        if reference == "ref1":
            rmsds = [result.rmsd_ref1 for result in results]
        else:
            rmsds = [result.rmsd_ref2 for result in results if result.rmsd_ref2 is not None]
            
        if not rmsds:
            raise ValueError(f"No valid RMSD values found for reference {reference}")
            
        rmsds = np.array(rmsds)
        
        # Calculate pLDDT correlation if available
        plddt_values = [r.plddt for r in results if r.plddt is not None]
        plddt_correlation = None
        if plddt_values:
            if reference == "ref1":
                rmsd_for_corr = [r.rmsd_ref1 for r in results if r.plddt is not None]
            else:
                rmsd_for_corr = [r.rmsd_ref2 for r in results if r.plddt is not None and r.rmsd_ref2 is not None]
            
            if rmsd_for_corr:
                plddt_correlation = float(np.corrcoef(plddt_values, rmsd_for_corr)[0,1])
        
        # Calculate significance tests if requested
        significance_tests = None
        if include_significance:
            significance_tests = self._calculate_significance_tests(rmsds)
        
        return RMSDStatistics(
            count=len(rmsds),
            min=float(np.min(rmsds)),
            max=float(np.max(rmsds)),
            mean=float(np.mean(rmsds)),
            median=float(np.median(rmsds)),
            std=float(np.std(rmsds)),
            q1=float(np.percentile(rmsds, 25)),
            q3=float(np.percentile(rmsds, 75)),
            iqr=float(np.percentile(rmsds, 75) - np.percentile(rmsds, 25)),
            structure_name=structure_name,
            reference=reference,
            plddt_correlation=plddt_correlation,
            significance_tests=significance_tests
        )

    def _calculate_significance_tests(self, rmsds: np.ndarray) -> Dict[str, Any]:
        """Calculate statistical significance tests."""
        try:
            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(rmsds)
            
            # One-sample t-test against mean
            ttest_stat, ttest_p = stats.ttest_1samp(rmsds, np.mean(rmsds))
            
            return {
                "shapiro": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": float(shapiro_p) > 0.05
                },
                "ttest": {
                    "statistic": float(ttest_stat),
                    "p_value": float(ttest_p)
                }
            }
        except Exception as e:
            logger.warning(f"Failed to calculate significance tests: {str(e)}")
            return {}

    def create_summary_dataframe(
        self,
        stats_list: List[RMSDStatistics]
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame from multiple statistical analyses.
        
        Args:
            stats_list: List of statistical analysis results
            
        Returns:
            DataFrame containing summary statistics
        """
        data = []
        for stats in stats_list:
            row = {
                'Structure': stats.structure_name,
                'Reference': stats.reference,
                'Count': stats.count,
                'Min': stats.min,
                'Max': stats.max,
                'Mean': stats.mean,
                'Median': stats.median,
                'Std': stats.std,
                'Q1': stats.q1,
                'Q3': stats.q3,
                'IQR': stats.iqr,
                'pLDDT_Correlation': stats.plddt_correlation
            }
            
            # Add significance test results if available
            if stats.significance_tests:
                if 'shapiro' in stats.significance_tests:
                    row.update({
                        'Is_Normal': stats.significance_tests['shapiro']['is_normal'],
                        'Shapiro_p': stats.significance_tests['shapiro']['p_value']
                    })
                if 'ttest' in stats.significance_tests:
                    row.update({
                        'T_test_p': stats.significance_tests['ttest']['p_value']
                    })
            
            data.append(row)
            
        return pd.DataFrame(data)

    def create_detailed_dataframe(
        self,
        results_dict: Dict[str, List[RMSDResult]],
        control_results: Optional[List[RMSDResult]] = None
    ) -> pd.DataFrame:
        """
        Create a detailed DataFrame containing all RMSD results.
        
        Args:
            results_dict: Dictionary mapping structure names to RMSD results
            control_results: Optional control structure results
            
        Returns:
            DataFrame containing detailed results
        """
        data = []
        
        # Add control data if available
        if control_results is not None:
            for result in control_results:
                row = {
                    'Structure': 'Control',
                    'Model': result.model_name,
                    'RMSD_Ref1': result.rmsd_ref1,
                    'RMSD_Ref2': result.rmsd_ref2,
                    'pLDDT': result.plddt
                }
                data.append(row)
                
        # Add mutation data
        for structure_name, results in results_dict.items():
            for result in results:
                row = {
                    'Structure': structure_name,
                    'Model': result.model_name,
                    'RMSD_Ref1': result.rmsd_ref1,
                    'RMSD_Ref2': result.rmsd_ref2,
                    'pLDDT': result.plddt
                }
                data.append(row)
                
        return pd.DataFrame(data)

    def save_results(
        self,
        detailed_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        output_dir: Union[str, Path],
        formats: List[str] = ["csv", "json"]
    ) -> None:
        """
        Save analysis results to files.
        
        Args:
            detailed_df: Detailed results DataFrame
            summary_df: Summary statistics DataFrame
            output_dir: Directory to save results
            formats: List of formats to save in
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            if fmt == "csv":
                detailed_df.to_csv(output_dir / 'detailed_rmsd_data.csv', index=False)
                summary_df.to_csv(output_dir / 'rmsd_summary_statistics.csv', index=False)
            elif fmt == "json":
                with open(output_dir / 'detailed_rmsd_data.json', 'w') as f:
                    json.dump(detailed_df.to_dict('records'), f, indent=2)
                with open(output_dir / 'rmsd_summary_statistics.json', 'w') as f:
                    json.dump(summary_df.to_dict('records'), f, indent=2)

    def compare_distributions(
        self,
        results1: List[RMSDResult],
        results2: List[RMSDResult],
        reference: str = "ref1"
    ) -> Dict[str, Any]:
        """
        Compare two RMSD distributions using statistical tests.
        
        Args:
            results1: First set of RMSD results
            results2: Second set of RMSD results
            reference: Which reference to use for comparison
            
        Returns:
            Dictionary containing test results
        """
        try:
            # Extract RMSD values
            if reference == "ref1":
                rmsds1 = [r.rmsd_ref1 for r in results1]
                rmsds2 = [r.rmsd_ref1 for r in results2]
            else:
                rmsds1 = [r.rmsd_ref2 for r in results1 if r.rmsd_ref2 is not None]
                rmsds2 = [r.rmsd_ref2 for r in results2 if r.rmsd_ref2 is not None]
            
            if not rmsds1 or not rmsds2:
                raise ValueError("Empty RMSD lists")
            
            # Perform statistical tests
            ks_stat, ks_p = stats.ks_2samp(rmsds1, rmsds2)  # Kolmogorov-Smirnov test
            mw_stat, mw_p = stats.mannwhitneyu(rmsds1, rmsds2)  # Mann-Whitney U test
            
            return {
                "ks_test": {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_p),
                    "significant": float(ks_p) < 0.05
                },
                "mann_whitney": {
                    "statistic": float(mw_stat),
                    "p_value": float(mw_p),
                    "significant": float(mw_p) < 0.05
                },
                "effect_size": {
                    "cohens_d": float(self._cohens_d(rmsds1, rmsds2))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to compare distributions: {str(e)}")
            return {}

    def _cohens_d(self, x: List[float], y: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std 

    def aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple files."""
        try:
            aggregated = {}
            for structure_name, stats in results.items():
                if not isinstance(stats, dict):
                    logger.warning(f"Invalid stats format for {structure_name}")
                    continue
                    
                # Copy stats to aggregated results
                aggregated[structure_name] = {
                    'mean': stats.get('mean', 0.0),
                    'std': stats.get('std', 0.0),
                    'min': stats.get('min', 0.0),
                    'max': stats.get('max', 0.0),
                    'n_samples': stats.get('n_samples', 0)
                }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate results: {str(e)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            return {} 