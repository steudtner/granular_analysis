"""Run the 2D upper-edge and slope-model pipeline over an alpha sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.io.loader import load_last_frame
from granular_analysis.paths import CasePaths
from granular_analysis.scripts.plot_slope_model_selection import plot_model_selection
from granular_analysis.scripts.run_slope_analysis import main as run_slope


def run_slope_alpha_sensitivity(
    case: str | Path,
    output_dir: str | Path,
    *,
    z_value: float,
    lz: float,
    alpha_start: float,
    alpha_stop: float,
    alpha_step: float,
    skip_first: int,
    upper_edge_bins: int = 50,
) -> None:
    """Run independent upper-edge and slope fits for every alpha value."""
    if alpha_step <= 0:
        raise ValueError("alpha_step must be positive")
    if alpha_stop < alpha_start:
        raise ValueError("alpha_stop must be greater than or equal to alpha_start")

    data_dir = CasePaths.from_input(case).require_trn()
    particles = load_last_frame(data_dir)
    output_dir = Path(output_dir)
    alphas = np.round(
        np.arange(alpha_start, alpha_stop + alpha_step / 2, alpha_step), 10
    )

    for alpha in alphas:
        alpha_dir = output_dir / f"alpha_{alpha:.1f}"
        upper_edge_dir = alpha_dir / "2d"
        result = compute_slice_porosity(
            particles,
            z0=z_value,
            alpha=float(alpha),
            lz=lz,
            n_bins=upper_edge_bins,
            skip_first=skip_first,
            output_dir=str(upper_edge_dir),
            write_upper_edge=True,
        )
        if result is None:
            raise RuntimeError(f"No valid slice computed at z={z_value}")
        run_slope(force=True, out=str(alpha_dir), alpha=float(alpha))

    plot_model_selection(output_dir, output_dir / "model_selection")
