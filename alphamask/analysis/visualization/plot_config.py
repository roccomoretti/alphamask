from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class PlotConfig:
    """
    Configuration for plot appearance.
    """
    plot_width: int = 15
    plot_height: int = 10
    colorscale: List[Tuple[float, Tuple[float, float, float]]] = None
    font_size: int = 12
    dpi: int = 300
    style: str = "default"
    interactive: bool = True
    show_statistics: bool = True
    show_significance: bool = True
    
    def __post_init__(self):
        if self.colorscale is None:
            # Default colorscale (copied from RMSDVisualizer)
            self.colorscale = [
                (0.0, (1.0, 1.0, 1.0)),
                (1/12, (0.416, 0.133, 0.996)),
                (3/12, (0.059, 0.651, 0.937)),
                (5/12, (0.314, 0.957, 0.800)),
                (7/12, (0.686, 0.957, 0.592)),
                (9/12, (1.0, 0.655, 0.349)),
                (11/12, (1.0, 0.133, 0.067)),
                (1.0, (1.0, 0.0, 0.0))
            ] 