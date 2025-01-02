"""Configuration handling for analysis module."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReferenceConfig:
    """Configuration for a reference structure."""
    path: Path
    description: Optional[str] = None

@dataclass
class RegionConfig:
    """Configuration for a protein region."""
    name: str
    start: int
    end: int
    description: Optional[str] = None

@dataclass
class StatTestConfig:
    """Configuration for statistical tests."""
    type: str
    alpha: float = 0.05

@dataclass
class PlotConfig:
    """Configuration for plot types."""
    enabled: bool = True
    style: str = "default"

@dataclass
class ExportConfig:
    """Configuration for export options."""
    formats: List[str] = None
    plots: List[str] = None

    def __post_init__(self):
        if self.formats is None:
            self.formats = ["csv", "json"]
        if self.plots is None:
            self.plots = ["pdf", "png"]

@dataclass
class AnalysisConfig:
    """Main configuration for analysis."""
    references: Dict[str, ReferenceConfig]
    regions: List[RegionConfig]
    atom_selection: str = "CA"
    statistical_tests: List[StatTestConfig] = None
    plots: Dict[str, PlotConfig] = None
    export: ExportConfig = None
    debug: bool = False

    def __post_init__(self):
        # Convert string paths to Path objects
        for ref_key, ref_config in self.references.items():
            if isinstance(ref_config.path, str):
                self.references[ref_key].path = Path(ref_config.path)

        # Initialize defaults
        if self.statistical_tests is None:
            self.statistical_tests = []
        
        if self.plots is None:
            self.plots = {}
            
        if self.export is None:
            self.export = ExportConfig()

    def validate(self) -> bool:
        """Validate the configuration."""
        try:
            # Check reference paths exist
            for ref_key, ref_config in self.references.items():
                if not ref_config.path.exists():
                    raise ValueError(f"Reference path does not exist: {ref_config.path}")

            # Validate region indices
            for region in self.regions:
                if region.start < 1:
                    raise ValueError(f"Region {region.name} has invalid start index: {region.start}")
                if region.end < region.start:
                    raise ValueError(f"Region {region.name} has invalid end index: {region.end}")

            # Validate atom selection
            if self.atom_selection not in ["CA", "backbone", "all"]:
                raise ValueError(f"Invalid atom selection: {self.atom_selection}")

            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False

def create_analysis_config(config_dict: Dict) -> AnalysisConfig:
    """Create analysis configuration from dictionary."""
    try:
        # Process references
        references = {}
        for ref_key, ref_data in config_dict.get("references", {}).items():
            # Convert path to absolute path
            ref_path = Path(ref_data["path"]).resolve()
            if not ref_path.exists():
                raise ValueError(f"Reference file not found: {ref_path}")
                
            references[ref_key] = ReferenceConfig(
                path=ref_path,
                description=ref_data.get("description")
            )

        # Process regions
        regions = []
        for region_data in config_dict.get("regions", []):
            regions.append(RegionConfig(
                name=region_data["name"],
                start=region_data["start"],
                end=region_data["end"],
                description=region_data.get("description")
            ))

        # Process statistical tests
        tests = []
        for test_data in config_dict.get("statistical_tests", []):
            tests.append(StatTestConfig(
                type=test_data["type"],
                alpha=test_data.get("alpha", 0.05)
            ))

        # Process plots
        plots = {}
        for plot_key, plot_data in config_dict.get("plots", {}).items():
            plots[plot_key] = PlotConfig(
                enabled=plot_data.get("enabled", True),
                style=plot_data.get("style", "default")
            )

        # Create export config
        export_data = config_dict.get("export", {})
        export = ExportConfig(
            formats=export_data.get("formats"),
            plots=export_data.get("plots")
        )

        return AnalysisConfig(
            references=references,
            regions=regions,
            atom_selection=config_dict.get("atom_selection", "CA"),
            statistical_tests=tests,
            plots=plots,
            export=export,
            debug=config_dict.get("debug", False)
        )

    except Exception as e:
        logger.error(f"Failed to create analysis config: {str(e)}")
        raise 