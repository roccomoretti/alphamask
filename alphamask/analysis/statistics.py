"""
Statistical analysis functionality for RMSD results.
"""
from typing import List, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import logging
from .rmsd import RMSDResult

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

class RMSDAnalyzer:
    """
    Analyzes RMSD calculation results and generates statistics.
    """
    
    @staticmethod
    def calculate_statistics(
        results: List[RMSDResult],
        structure_name: str,
        reference: str = "ref1"
    ) -> RMSDStatistics:
        """
        Calculate comprehensive statistics for a set of RMSD results.
        
        Args:
            results: List of RMSD results
            structure_name: Name of the structure being analyzed
            reference: Which reference structure to analyze ('ref1' or 'ref2')
            
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
            reference=reference
        )

    @staticmethod
    def create_summary_dataframe(
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
                'IQR': stats.iqr
            }
            data.append(row)
            
        return pd.DataFrame(data)

    @staticmethod
    def create_detailed_dataframe(
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

    @staticmethod
    def save_results(
        detailed_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        output_dir: Union[str, Path],
        save_numpy: bool = True
    ) -> None:
        """
        Save analysis results to files.
        
        Args:
            detailed_df: Detailed results DataFrame
            summary_df: Summary statistics DataFrame
            output_dir: Directory to save results
            save_numpy: Whether to also save results as numpy arrays
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSVs
        detailed_df.to_csv(output_dir / 'detailed_rmsd_data.csv', index=False)
        summary_df.to_csv(output_dir / 'rmsd_summary_statistics.csv', index=False)
        
        # Save numpy arrays if requested
        if save_numpy:
            np.save(
                output_dir / 'detailed_rmsd_data.npy',
                {
                    'detailed_data': detailed_df.to_dict('records'),
                    'summary_data': summary_df.to_dict('records')
                }
            )

    @staticmethod
    def filter_outliers(
        results: List[RMSDResult],
        zscore_threshold: float = 3.0
    ) -> List[RMSDResult]:
        """
        Filter out RMSD outliers based on Z-score.
        
        Args:
            results: List of RMSD results
            zscore_threshold: Z-score threshold for outlier detection
            
        Returns:
            Filtered list of results
        """
        rmsds = np.array([result.rmsd_ref1 for result in results])
        zscores = np.abs((rmsds - np.mean(rmsds)) / np.std(rmsds))
        
        return [result for result, zscore in zip(results, zscores) 
                if zscore <= zscore_threshold]

    @staticmethod
    def analyze_plddt_rmsd_correlation(
        results: List[RMSDResult]
    ) -> Dict[str, float]:
        """
        Analyze correlation between pLDDT scores and RMSD values.
        
        Args:
            results: List of RMSD results
            
        Returns:
            Dictionary containing correlation coefficients
        """
        plddt_values = np.array([result.plddt for result in results if result.plddt is not None])
        rmsd1_values = np.array([result.rmsd_ref1 for result in results if result.plddt is not None])
        
        correlations = {
            'pearson_ref1': float(np.corrcoef(plddt_values, rmsd1_values)[0,1])
        }
        
        # Add correlation for second reference if available
        rmsd2_values = [result.rmsd_ref2 for result in results 
                       if result.plddt is not None and result.rmsd_ref2 is not None]
        if rmsd2_values:
            correlations['pearson_ref2'] = float(
                np.corrcoef(plddt_values, np.array(rmsd2_values))[0,1]
            )
            
        return correlations 