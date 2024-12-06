from typing import Optional, Dict, Any, List, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pandas as pd
from ..core.msa import MSAUtils
import numpy as np
class MSAVisualizer:
    """
    Visualizer for MSA and coevolution analysis.
    """
    
    def __init__(
        self,
        show_msa_plots: bool = True,
        show_coevolution: bool = True,
        plot_window: int = 10
    ):
        self.show_msa_plots = show_msa_plots
        self.show_coevolution = show_coevolution
        self.plot_window = plot_window
        self.msa_utils = MSAUtils()

    def visualize_pipeline_results(
        self,
        pipeline: Any,
        parent_path: str,
        mutation_str: Optional[str] = None
    ) -> None:
        """
        Visualize MSA and coevolution results from a pipeline.
        
        Args:
            pipeline: Pipeline instance containing stored MSAs
            parent_path: Path to save visualization files
            mutation_str: Optional mutation string for labeling
        """
        msas = pipeline.get_stored_msas()
        
        if msas['original'] is not None:
            if self.show_msa_plots:
                self._visualize_msas(msas, pipeline.jobname, parent_path)

            if self.show_coevolution:
                self._visualize_coevolution(msas, pipeline.jobname, parent_path, mutation_str)

            # Print statistics
            print("\n📈 MSA Statistics:")
            print(f"Total sequences: {msas['original'].shape[0]}")
            print(f"Sequence length: {msas['original'].shape[1]}")
            if mutation_str:
                print(f"Mutation: {mutation_str}")

    def _visualize_msas(
        self,
        msas: Dict,
        jobname: str,
        parent_path: str
    ) -> None:
        """Visualize MSA arrays using subplots."""
        print("\n📊 MSA Visualizations")
        print("-" * 50)
        
        # Create output directory if it doesn't exist
        output_dir = Path(parent_path) / jobname
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3 if msas['masked'] is not None else 1,
            subplot_titles=("Original MSA", "Mutated MSA", "Masked MSA")
            if msas['masked'] is not None else ("Original MSA",)
        )
        
        # Add original MSA
        fig.add_trace(
            go.Heatmap(
                z=msas['original'],
                colorscale=self._get_msa_colorscale(),
                showscale=True,
                name="Original"
            ),
            row=1, col=1
        )
        
        if msas['mutated'] is not None:
            fig.add_trace(
                go.Heatmap(
                    z=msas['mutated'],
                    colorscale=self._get_msa_colorscale(),
                    showscale=True,
                    name="Mutated"
                ),
                row=1, col=2
            )
        
        if msas['masked'] is not None:
            fig.add_trace(
                go.Heatmap(
                    z=msas['masked'],
                    colorscale=self._get_msa_colorscale(),
                    showscale=True,
                    name="Masked"
                ),
                row=1, col=3
            )
        
        # Update layout
        fig.update_layout(
            title="Multiple Sequence Alignment Analysis",
            height=500,
            width=1500 if msas['masked'] is not None else 500,
            showlegend=True
        )
        
        # Update axes labels
        for i in range(1, 4):
            fig.update_xaxes(title_text="Position", row=1, col=i)
            fig.update_yaxes(title_text="Sequence" if i == 1 else "", row=1, col=i)
        
        # Save and display
        fig.write_html(str(output_dir / "msa_visualization.html"))
        fig.write_image(str(output_dir / "msa_visualization.pdf"))
        fig.show()

    def _get_msa_colorscale(self) -> List[List[Union[float, str]]]:
        """Get colorscale for MSA visualization."""
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#1a55FF', '#55a2FF', '#55FFB1', '#a2FF55', '#FFEA1a',
            '#FF551a', '#FF1a55', '#FF1aa3', '#B51aFF', '#1a8CFF',
            '#1aFF55', '#7F1aFF'
        ]
        
        n_colors = len(colors)
        return [
            [i/(n_colors-1), color] for i, color in enumerate(colors)
        ]

    def _get_top_coevolving_pairs(
        self,
        coev_matrix: np.ndarray,
        n_pairs: int = 10
    ) -> pd.DataFrame:
        """
        Get the top coevolving residue pairs from a coevolution matrix.
        
        Args:
            coev_matrix: Coevolution matrix
            n_pairs: Number of top pairs to return
            
        Returns:
            DataFrame with top coevolving pairs
        """
        pairs = []
        for i in range(len(coev_matrix)):
            for j in range(i+1, len(coev_matrix)):
                pairs.append({
                    'Position 1': i+1,  # 1-indexed
                    'Position 2': j+1,  # 1-indexed
                    'Coevolution Score': coev_matrix[i,j]
                })
        
        # Create DataFrame and sort by coevolution score
        df = pd.DataFrame(pairs)
        df = df.sort_values('Coevolution Score', ascending=False)
        
        return df.head(n_pairs)

    def _visualize_coevolution(
        self,
        msas: Dict,
        jobname: str,
        parent_path: str,
        mutation_str: Optional[str] = None,
        n_top_pairs: int = 10
    ) -> None:
        """Visualize coevolution matrices and top pairs."""
        print("\n🔄 Coevolution Analysis")
        print("-" * 50)
        
        # Create output directory if it doesn't exist
        output_dir = Path(parent_path) / jobname
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate coevolution matrices
        coev_original = self.msa_utils.get_coevolution(msas['original'])
        
        # Get and display top coevolving pairs
        print("\n📊 Top Coevolving Pairs:")
        print("-" * 50)
        
        # Original MSA
        print("\nOriginal MSA:")
        df_original = self._get_top_coevolving_pairs(coev_original, n_top_pairs)
        print(df_original.to_string(index=False))
        df_original.to_csv(output_dir / "top_coevolving_pairs_original.csv", index=False)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3 if msas['masked'] is not None else 1,
            subplot_titles=("Original Coevolution", "Mutated Coevolution", "Masked Coevolution")
            if msas['masked'] is not None else ("Original Coevolution",)
        )
        
        fig.add_trace(
            go.Heatmap(z=coev_original, colorscale='Viridis', name="Original"),
            row=1, col=1
        )
        
        if msas['mutated'] is not None:
            coev_mutated = self.msa_utils.get_coevolution(msas['mutated'])
            fig.add_trace(
                go.Heatmap(z=coev_mutated, colorscale='Viridis', name="Mutated"),
                row=1, col=2
            )
            
            # Get and display top pairs for mutated MSA
            print("\nMutated MSA:")
            df_mutated = self._get_top_coevolving_pairs(coev_mutated, n_top_pairs)
            print(df_mutated.to_string(index=False))
            df_mutated.to_csv(output_dir / "top_coevolving_pairs_mutated.csv", index=False)
        
        if msas['masked'] is not None:
            coev_masked = self.msa_utils.get_coevolution(msas['masked'])
            fig.add_trace(
                go.Heatmap(z=coev_masked, colorscale='Viridis', name="Masked"),
                row=1, col=3
            )
            
            # Get and display top pairs for masked MSA
            print("\nMasked MSA:")
            df_masked = self._get_top_coevolving_pairs(coev_masked, n_top_pairs)
            print(df_masked.to_string(index=False))
            df_masked.to_csv(output_dir / "top_coevolving_pairs_masked.csv", index=False)
        
        fig.update_layout(
            title="Coevolution Analysis" + (f" - {mutation_str}" if mutation_str else ""),
            height=500,
            width=1500 if msas['masked'] is not None else 500,
            showlegend=True
        )
        
        # Save and display
        fig.write_html(str(output_dir / "coevolution.html"))
        fig.show()
        
        # Create combined comparison table if multiple analyses exist
        dfs = {'Original': df_original}
        if msas['mutated'] is not None:
            dfs['Mutated'] = df_mutated
        if msas['masked'] is not None:
            dfs['Masked'] = df_masked
            
        if len(dfs) > 1:
            print("\n📈 Comparison of Top Coevolving Pairs:")
            print("-" * 50)
            comparison_df = pd.concat(
                [df.assign(Analysis=name) for name, df in dfs.items()],
                ignore_index=True
            )
            print(comparison_df.to_string(index=False))
            comparison_df.to_csv(output_dir / "coevolving_pairs_comparison.csv", index=False)