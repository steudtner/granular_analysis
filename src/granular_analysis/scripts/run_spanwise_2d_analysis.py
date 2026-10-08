"""Create spanwise 2D area and porosity profiles for one case."""

from __future__ import annotations

from granular_analysis.plotting import plotstyle

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.config import SKIP_FIRST_PARTICLES
from granular_analysis.io.loader import load_last_frame
from granular_analysis.paths import CasePaths
from granular_analysis.plotting.helpers import viridis_palette


def _mean_std_band(axis, value: float, std: float, colour) -> None:
    axis.axhline(value, color=colour, linestyle="--", linewidth=1.2,
                label=rf"Mittelwert: {value:.3f}")
    axis.axhspan(value - std, value + std, color=colour, alpha=0.12,
                label=rf"$\pm 1\sigma$: {std:.3f}")


def run_spanwise_2d_analysis(
    case: str | Path,
    output_dir: str | Path,
    *,
    alpha: float = 0.5,
    lz: float = 2.5,
    z_step: float = 0.05,
    skip_first: int = SKIP_FIRST_PARTICLES,
) -> pd.DataFrame:
    """Evaluate alpha-shape geometry over equally spaced spanwise planes."""
    if lz <= 0 or z_step <= 0:
        raise ValueError("lz and z_step must be positive")

    data_dir = CasePaths.from_input(case).require_trn()
    particles = load_last_frame(data_dir)
    frame = max(data_dir.glob("ell_trn_*.bin"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    z_values = np.round(np.arange(0.0, lz, z_step), 10)

    rows = []
    for z_value in z_values:
        result = compute_slice_porosity(
            particles,
            z0=float(z_value),
            alpha=alpha,
            lz=lz,
            skip_first=skip_first,
        )
        if result is None:
            raise RuntimeError(f"No valid slice computed at z={z_value}")
        rows.append({
            "z": float(z_value),
            "ellipse_union_area": float(result["A_ell"]),
            "alphashape_area": float(result["A_alpha"]),
            "porosity": float(result["porosity"]),
            "alpha": alpha,
            "lz": lz,
            "frame": frame.name,
            "skip_first": skip_first,
        })

    data = pd.DataFrame(rows)
    data.to_csv(output_dir / "spanwise_2d_summary.csv", index=False)
    palette = viridis_palette()

    fig, axis = plt.subplots(figsize=(plotstyle.figwidth_full, 4.6), constrained_layout=True)
    colour = palette[2]
    axis.plot(data["z"], data["porosity"], "o-", color=colour,
              linewidth=1.2, markersize=4, label="Porosität")
    _mean_std_band(axis, data["porosity"].mean(), data["porosity"].std(ddof=1), colour)
    axis.set(
        xlabel=r"Spannrichtung, $z\,D_{\mathrm{Eq}}^{-1}$",
        ylabel="Porosität, $\\phi$",
        title=rf"primeJainSU: Porosität über die Spannrichtung ($\alpha={alpha:g}$)",
        xlim=(0, lz),
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    plotstyle.export_figure(fig, output_dir / "porosity_over_z.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(plotstyle.figwidth_full, 6.4), sharex=True,
                             constrained_layout=True)
    for axis, (column, label, colour) in zip(axes, (
        ("ellipse_union_area", r"Ellipsenfläche, $A_{\mathrm{ell}}$", palette[1]),
        ("alphashape_area", r"Alpha-Shape-Fläche, $A_{\alpha}$", palette[3]),
    )):
        axis.plot(data["z"], data[column], "o-", color=colour,
                  linewidth=1.2, markersize=4, label=label)
        _mean_std_band(axis, data[column].mean(), data[column].std(ddof=1), colour)
        axis.set_ylabel(r"Fläche [$D_{\mathrm{Eq}}^2$]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[1].set(xlabel=r"Spannrichtung, $z\,D_{\mathrm{Eq}}^{-1}$", xlim=(0, lz))
    fig.suptitle(rf"primeJainSU: Schnittflächen über die Spannrichtung ($\alpha={alpha:g}$)")
    plotstyle.export_figure(fig, output_dir / "areas_over_z.png", dpi=200)
    plt.close(fig)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="Simulation case or transient-data path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lz", type=float, default=2.5)
    parser.add_argument("--z-step", type=float, default=0.05)
    parser.add_argument("--skip-first", type=int, default=SKIP_FIRST_PARTICLES)
    args = parser.parse_args()
    data = run_spanwise_2d_analysis(
        args.case, args.out, alpha=args.alpha, lz=args.lz,
        z_step=args.z_step, skip_first=args.skip_first,
    )
    print(data.describe().loc[["mean", "std"]].to_string())


if __name__ == "__main__":
    main()
