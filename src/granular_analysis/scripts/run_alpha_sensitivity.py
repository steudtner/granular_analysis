"""Evaluate 2D alpha-shape area and porosity over an alpha sweep."""

from __future__ import annotations

from granular_analysis.plotting import plotstyle

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.config import LZ, SKIP_FIRST_PARTICLES
from granular_analysis.io.loader import load_last_frame
from granular_analysis.paths import CasePaths


def run_sensitivity(
    case: str | Path,
    output_dir: str | Path,
    *,
    z_value: float = LZ / 2,
    lz: float = LZ,
    alpha_start: float = 0.1,
    alpha_stop: float = 1.0,
    alpha_step: float = 0.1,
    skip_first: int = SKIP_FIRST_PARTICLES,
) -> pd.DataFrame:
    """Compute and plot alpha-shape sensitivity for the latest case frame."""
    data_dir = CasePaths.from_input(case).require_trn()
    frame = max(data_dir.glob("ell_trn_*.bin"))
    particles = load_last_frame(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = np.round(
        np.arange(alpha_start, alpha_stop + alpha_step / 2, alpha_step), 10
    )
    rows = []
    shapes = []
    for alpha in alphas:
        result = compute_slice_porosity(
            particles,
            z0=z_value,
            alpha=float(alpha),
            lz=lz,
            skip_first=skip_first,
        )
        if result is None:
            raise RuntimeError(f"No particles intersect the slice at z={z_value}")
        shapes.append(result["shape"])
        rows.append(
            {
                "alpha": float(alpha),
                "alphashape_area": float(result["A_alpha"]),
                "ellipse_union_area": float(result["A_ell"]),
                "porosity": float(result["porosity"]),
                "frame": frame.name,
                "z": z_value,
                "lz": lz,
                "skip_first": skip_first,
            }
        )

    data = pd.DataFrame(rows)
    data.to_csv(output_dir / "alpha_sensitivity.csv", index=False)

    fig, area_axis = plt.subplots(figsize=(plotstyle.figwidth_full, 4.6), constrained_layout=True)
    area_axis.plot(
        data["alpha"], data["alphashape_area"], "o-", color="tab:blue",
        linewidth=1.2, label="Alpha-Shape-Fläche",
    )
    area_axis.set(
        xlabel=r"Alpha-Parameter, $\alpha$",
        ylabel=r"Alpha-Shape-Fläche, $A_\alpha$ [$D_{\mathrm{Eq}}^2$]",
        xticks=alphas,
        ylim=(400, 500),
        title=f"primeJainSU: Sensitivität gegenüber dem Alpha-Parameter (z = {z_value:g})",
    )
    area_axis.grid(True, alpha=0.3)
    area_axis.legend(loc="best")
    plotstyle.export_figure(fig, output_dir / "alpha_sensitivity.png", dpi=200)
    plt.close(fig)

    valid_shapes = [np.asarray(shape) for shape in shapes if shape is not None]
    if valid_shapes:
        all_coordinates = np.vstack(valid_shapes)
        x_min, y_min = all_coordinates.min(axis=0)
        x_max, y_max = all_coordinates.max(axis=0)
        x_padding = 0.02 * (x_max - x_min)
        y_padding = 0.02 * (y_max - y_min)

        figure, axes = plt.subplots(
            4, 5, figsize=(plotstyle.figwidth_full, 10), sharex=True, sharey=True,
            constrained_layout=True,
        )
        for axis, alpha, shape in zip(axes.flat, alphas, shapes):
            if shape is not None:
                coordinates = np.asarray(shape)
                axis.fill(
                    coordinates[:, 0], coordinates[:, 1],
                    color="tab:blue", alpha=0.25,
                )
                axis.plot(
                    coordinates[:, 0], coordinates[:, 1],
                    color="tab:blue", linewidth=1.2,
                )
            axis.set_title(rf"$\alpha = {alpha:.1f}$")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=0.2)
            axis.set_xlim(x_min - x_padding, x_max + x_padding)
            axis.set_ylim(y_min - y_padding, y_max + y_padding)
        figure.supxlabel(r"$x\,D_{\mathrm{Eq}}^{-1}$")
        figure.supylabel(r"$y\,D_{\mathrm{Eq}}^{-1}$")
        figure.suptitle(
            f"Alpha-Shapes für primeJainSU bei z = {z_value:g}", fontsize=10
        )
        plotstyle.export_figure(figure, output_dir / "alpha_shapes.png", dpi=200)
        plt.close(figure)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="Simulation case or transient-data path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--z", type=float, default=LZ / 2)
    parser.add_argument("--lz", type=float, default=LZ)
    parser.add_argument("--alpha-start", type=float, default=0.1)
    parser.add_argument("--alpha-stop", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--skip-first", type=int, default=SKIP_FIRST_PARTICLES)
    args = parser.parse_args()
    data = run_sensitivity(
        args.case,
        args.out,
        z_value=args.z,
        lz=args.lz,
        alpha_start=args.alpha_start,
        alpha_stop=args.alpha_stop,
        alpha_step=args.alpha_step,
        skip_first=args.skip_first,
    )
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
