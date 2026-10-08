"""Utilities to compare metrics across multiple simulation cases.

Provides functions to compute and save comparisons for porosity over z and
kinetic energy time series. Cases are provided explicitly as a mapping
from case name to data folder.
"""
from __future__ import annotations

from granular_analysis.plotting import plotstyle

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _iter_with_progress(items, desc: str, show: bool):
    """Return an iterator over items with optional progress reporting.

    If `tqdm` is available and `show` is True, returns a tqdm iterator.
    Otherwise, when `show` is True, prints a simple `i/N` status on each
    iteration. If `show` is False, returns the plain items iterable.
    """
    if not show:
        return items
    if tqdm is not None:
        return tqdm(items, desc=desc)

    # Simple textual fallback progress indicator
    total = len(items)

    def gen():
        for i, it in enumerate(items, start=1):
            print(f"{desc}: {i}/{total}", end="\r", flush=True)
            yield it
        print("" )

    return gen()

from granular_analysis.config import RESULTS_DIR, Z_LEVELS, LZ, ALPHA_2D, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES
from granular_analysis.paths import CasePaths
from granular_analysis.io.loader import load_frame, load_frame_by_time, filter_particles

# Reuse existing analysers where convenient
from granular_analysis.scripts.run_slice_comparison import compute_2d_slice_summary
from granular_analysis.scripts.run_energy_analysis import compute_energy_dataframe
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.geometry.alpha_utils import densify_polygon
from granular_analysis.plotting.plots import plot_profile, plot_energy_time_series


def _ensure_outdir(path: str | None, sub: str) -> str:
    if path is None:
        outdir = os.path.join(RESULTS_DIR, "comparison", sub)
    else:
        outdir = os.path.join(path, sub)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def compare_porosity_z(cases: Dict[str, str], out_dir: str | None = None, *, slice_z: Optional[float] = None, include_alpha: bool = False, densify_max_segment_length: Optional[float] = None, show_progress: bool = False, frame: Optional[int | str] = None, time: Optional[float] = None, time_mode: str = 'nearest') -> Dict[str, pd.DataFrame]:
    """Compute porosity-vs-z for multiple cases and save combined CSV + plot.

    cases: mapping case_name -> folder_path
    Returns dict of case_name -> DataFrame
    """
    outdir = _ensure_outdir(out_dir, "porosity_z")
    dfs = {}
    combined = []
    case_items = list(cases.items())
    iter_cases = _iter_with_progress(case_items, desc="Cases", show=show_progress)
    for name, folder in iter_cases:
        try:
            # allow explicit frame selection for reproducibility
            if time is not None:
                ell = load_frame_by_time(folder, time, mode=time_mode)
            else:
                ell = load_frame(folder, frame=frame)
        except Exception as exc:
            print(f"ERROR loading case '{name}' from {folder}: {exc}")
            continue

        # apply default filtering (use config flags to skip initial particles)
        pts = filter_particles(ell, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES)
        df = compute_2d_slice_summary(
            pts,
            Z_LEVELS,
            alpha=ALPHA_2D,
            skip_first=0,
        )
        dfs[name] = df

        # save per-case CSV
        csv_path = os.path.join(outdir, f"{name}_slice_summary.csv")
        df.to_csv(csv_path, index=False)
        combined.append(df.assign(case=name))

    # optional: compute and plot alpha-shapes at a single slice_z for visual comparison
    if include_alpha:
        # determine default slice_z (use provided or config LZ/2)
        z0 = slice_z if slice_z is not None else float(LZ) / 2.0
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 4))
        case_items = list(cases.items())
        iter_cases = _iter_with_progress(case_items, desc=f"Alpha shapes @ z={z0:.2f}", show=show_progress)
        for name, folder in iter_cases:
            try:
                if time is not None:
                    ell = load_frame_by_time(folder, time, mode=time_mode)
                else:
                    ell = load_frame(folder, frame=frame)
            except Exception:
                continue
            pts = filter_particles(ell, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES)
            res = compute_slice_porosity(pts, z0, ALPHA_2D, LZ)
            shape = None if res is None else res.get("shape")
            if shape is None:
                continue
            coords = shape
            if densify_max_segment_length is not None and densify_max_segment_length > 0:
                coords = densify_polygon(coords, densify_max_segment_length)
            ax.plot(coords[:, 0], coords[:, 1], label=name)
            ax.fill(coords[:, 0], coords[:, 1], alpha=0.15)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Alpha shape comparison at z={z0:.3f}")
        ax.grid(True)
        ax.legend()
        plotstyle.prepare_figure(fig)
        fig.tight_layout()
        png = os.path.join(outdir, "porosity_z_alphashape_comparison.png")
        plotstyle.export_figure(fig, png, dpi=200)
        plt.close(fig)
        print(f"Saved alpha-shape comparison to {png}")

    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        combined_csv = os.path.join(outdir, "combined_slice_summary.csv")
        all_df.to_csv(combined_csv, index=False)

        # per-case plots using existing template
        for name, df in dfs.items():
            try:
                per_png = os.path.join(outdir, f"{name}_porosity_z.png")
                plot_profile(df, per_png)
            except Exception:
                # ignore per-case plotting failures
                pass

        # combined overlay plot
        png = os.path.join(outdir, "porosity_z_comparison.png")
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 5))
        for name, df in dfs.items():
            ax.plot(df["z"], df["porosity"], marker="o", label=name)
        ax.set_xlabel("z")
        ax.set_ylabel("porosity")
        ax.grid(True)
        ax.legend()
        plotstyle.prepare_figure(fig)
        fig.tight_layout()
        plotstyle.export_figure(fig, png, dpi=200)
        plt.close(fig)

        print(f"Saved comparison CSV to {combined_csv} and plot to {png}")

    return dfs


def _interpolate_series(df: pd.DataFrame, x_col: str, y_col: str, x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate y onto x_grid. Assumes x is sorted increasing."""
    xs = np.asarray(df[x_col])
    ys = np.asarray(df[y_col])
    # handle duplicates by taking mean
    ux, idx = np.unique(xs, return_index=True)
    if len(ux) < len(xs):
        # group by unique x
        ser = pd.Series(ys, index=xs)
        ser = ser.groupby(level=0).mean()
        ux = ser.index.values
        uy = ser.values
    else:
        uy = ys

    # allow extrapolation via nearest-fill for values outside range
    y_interp = np.interp(x_grid, ux, uy, left=uy[0], right=uy[-1])
    return x_grid, y_interp


def compare_kinetic_energy(cases: Dict[str, str], out_dir: str | None = None, n_time_points: int = 200, show_progress: bool = False, frame: Optional[int | str] = None) -> Dict[str, pd.DataFrame]:
    """Compute kinetic energy time series for multiple cases, resample to a common normalized time grid, and save results.

    Uses `time_norm` and `kinetic_energy_norm` columns produced by `compute_energy_dataframe`.
    Returns dict of case_name -> original DataFrame.
    """
    outdir = _ensure_outdir(out_dir, "kinetic_energy")
    dfs = {}
    t_min, t_max = None, None

    # Determine whether physical normalization is available for ALL cases
    use_physical_all = True
    case_input_paths = {}
    for name, folder in cases.items():
        try:
            case_root = CasePaths.from_input(folder).case_root
            input_paths = {
                "global": case_root / "input.dat",
                "ellipsoid": case_root / "input_ellipsoid_data.dat",
            }
            if not input_paths["global"].exists() or not input_paths["ellipsoid"].exists():
                use_physical_all = False
        except Exception:
            use_physical_all = False
            input_paths = None
        case_input_paths[name] = input_paths

    # First pass: compute data frames and collect overlapping time range
    case_items = list(cases.items())
    iter_cases = _iter_with_progress(case_items, desc="Energy: compute", show=show_progress)
    for name, folder in iter_cases:
        input_paths = case_input_paths.get(name)
        try:
            df = compute_energy_dataframe(folder, use_physical_mass=use_physical_all, skip_first=0, input_paths=(input_paths if use_physical_all else None))
        except Exception as exc:
            print(f"ERROR computing energy for '{name}': {exc}")
            continue

        dfs[name] = df
        if not df.empty:
            lo, hi = float(df["time_norm"].min()), float(df["time_norm"].max())
            # compute intersection of time ranges across cases
            t_min = lo if t_min is None else max(t_min, lo)
            t_max = hi if t_max is None else min(t_max, hi)

    if not dfs:
        return {}

    # Build common grid over the intersection (overlap) of all time ranges
    if t_min is None or t_max is None or t_min >= t_max:
        # fallback to union behavior if no overlap
        all_lo = None
        all_hi = None
        for df in dfs.values():
            if df.empty:
                continue
            lo = float(df["time_norm"].min())
            hi = float(df["time_norm"].max())
            all_lo = lo if all_lo is None else min(all_lo, lo)
            all_hi = hi if all_hi is None else max(all_hi, hi)
        t_min = all_lo if t_min is None else t_min
        t_max = all_hi if t_max is None else t_max

    x_grid = np.linspace(t_min, t_max, n_time_points)

    combined_rows = []
    combined_rows = []
    # Save and plot per-case using existing plotting template (consistent normalization)
    for name, df in dfs.items():
        if df.empty:
            continue
        per_png = os.path.join(outdir, f"{name}_kinetic_energy_norm.png")
        try:
            plot_energy_time_series(df, per_png, use_normalized=use_physical_all)
        except Exception:
            pass

    # Combined plot: use same styling as plot_energy_time_series for visual parity
    fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 4))
    iter_plot = _iter_with_progress(list(dfs.items()), desc="Energy: resample/plot", show=show_progress)

    all_x_min, all_x_max = None, None
    all_y_min, all_y_max = None, None

    for name, df in iter_plot:
        if df.empty:
            continue
        xg, yg = _interpolate_series(df, "time_norm", "kinetic_energy_norm", x_grid)
        ax.plot(xg, yg, linewidth=1.2, label=name)
        out_csv = os.path.join(outdir, f"{name}_kinetic_energy_norm.csv")
        pd.DataFrame({"time_norm": xg, "kinetic_energy_norm": yg}).to_csv(out_csv, index=False)
        combined_rows.append(pd.DataFrame({"time_norm": xg, "kinetic_energy_norm": yg, "case": name}))

        # update axis bounds
        xmin, xmax = float(xg.min()), float(xg.max())
        ymin, ymax = float(yg.min()), float(yg.max())
        all_x_min = xmin if all_x_min is None else min(all_x_min, xmin)
        all_x_max = xmax if all_x_max is None else max(all_x_max, xmax)
        all_y_min = ymin if all_y_min is None else min(all_y_min, ymin)
        all_y_max = ymax if all_y_max is None else max(all_y_max, ymax)

    # set labels consistent with per-case plots
    if use_physical_all:
        ax.set_xlabel(r"$t g^{1/2} D_{\mathrm{Eq}}^{-1/2}$", fontsize=10)
        ax.set_ylabel(r"$\sum E_{kin} \,\rho^{-1} g^{-1} D_{\mathrm{Eq}}^{-4}$", fontsize=10)
        ax.set_title("Normalized kinetic energy over time", fontsize=10)
    else:
        ax.set_xlabel("time", fontsize=10)
        ax.set_ylabel("kinetic energy", fontsize=10)
        ax.set_title("Total kinetic energy over time", fontsize=10)

    # apply axis limits based on union of per-case ranges for consistent view
    if all_x_min is not None and all_x_max is not None:
        ax.set_xlim(all_x_min, all_x_max)
    if all_y_min is not None and all_y_max is not None:
        pad = 0.05 * (all_y_max - all_y_min) if all_y_max > all_y_min else 0.1
        ax.set_ylim(max(0.0, all_y_min - pad), all_y_max + pad)

    ax.grid(True)
    ax.legend(fontsize=10)
    plotstyle.prepare_figure(fig)
    fig.tight_layout()
    png = os.path.join(outdir, "kinetic_energy_comparison.png")
    plotstyle.export_figure(fig, png, dpi=300)
    plt.close(fig)

    if combined_rows:
        all_df = pd.concat(combined_rows, ignore_index=True)
        all_csv = os.path.join(outdir, "combined_kinetic_energy_norm.csv")
        all_df.to_csv(all_csv, index=False)
        print(f"Saved combined energy CSV to {all_csv} and plot to {png}")

    return dfs
