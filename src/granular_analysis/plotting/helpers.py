from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from granular_analysis.config import RESULTS_DIR
from granular_analysis.plotting.plotstyle import PLOT_STYLE, export_figure, german_decimal


def viridis_palette(n_colors: int = 6):
    """Return colours from the lower 80% of Viridis."""
    if n_colors < 1:
        raise ValueError('n_colors must be at least 1')
    return plt.get_cmap('viridis')(np.linspace(0.0, 0.8, n_colors))


def save_figure(fig: plt.Figure, out_path: str | Path | None = None, *,
                default_name: str = 'plot.png', dpi: int = 600) -> str:
    """Export matching TeX SVG/PDF plus the requested format, then close."""
    target = Path(out_path) if out_path is not None else Path(RESULTS_DIR) / default_name
    if not target.suffix:
        target = target.with_suffix('.pdf')
    export_figure(fig, target, dpi=dpi)
    plt.close(fig)
    return str(target)
