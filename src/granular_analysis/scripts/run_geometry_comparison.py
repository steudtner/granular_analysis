"""Compare alpha contours, porosity and segmented surface fits across cases."""

from granular_analysis.plotting import plotstyle

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from granular_analysis.analysis.porosity2d import compute_slice_porosity, upper_edge_points
from granular_analysis.config import ALPHA_2D, LZ, Z_LEVELS, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES
from granular_analysis.io.fluid_slice import nearest_frame
from granular_analysis.io.loader import load_frame, _read_bin_time
from granular_analysis.paths import CasePaths, latest_trn_counter
from granular_analysis.plotting.helpers import PLOT_STYLE, save_figure
from granular_analysis.scripts.run_slice_comparison import compute_2d_slice_summary
from granular_analysis.scripts.run_slope_analysis import main as run_slope


def select_comparison_frames(paths, labels=None, counter=None, time=None):
    cases = [CasePaths.from_input(path) for path in paths]
    roots = [case.case_root.resolve() for case in cases]
    labels = labels or [root.name for root in roots]
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("Comparison needs at least two distinct cases")
    if len(labels) != len(cases) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique --compare-labels entry per case")
    selected = []
    for label, case in zip(labels, cases):
        chosen = counter if counter is not None else latest_trn_counter(case)
        if time is not None:
            _, chosen = nearest_frame(case, time)
        if chosen is None:
            raise FileNotFoundError(f"No particle frames in {case.trn}")
        file = case.trn / f"ell_trn_{chosen:06d}.bin"
        actual_time = _read_bin_time(file)
        selected.append((label, case, chosen, actual_time))
    return selected


def _save_overlay(series, prefix, xlabel, ylabel, *, equal=False, title=None):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 5))
        try:
            for label, x, y in series:
                ax.plot(x, y, label=label, linewidth=1.2,
                        markevery=max(1, len(x)//12), **plotstyle.case_style(label))
            ax.set(xlabel=xlabel, ylabel=ylabel)
            if equal:
                ax.set_aspect("equal", adjustable="datalim")
            if title:
                ax.set_title(title)
            ax.grid(alpha=.2)
            plotstyle.legend_above(ax)
            plotstyle.prepare_figure(fig)
            fig.tight_layout()
            plotstyle.export_figure(fig, str(prefix) + ".png", dpi=200)
            plt.close(fig)
        finally:
            plt.close(fig)


def slope_predictions(row, x):
    """Reconstruct the same continuous segmented fit as slope_plots."""
    slopes = np.array([float(value) for value in str(row["slopes"]).split(";")])
    raw = row["breakpoints"]
    points = [] if pd.isna(raw) or str(raw).strip() == "" else [float(v) for v in str(raw).split(";")]
    if len(slopes) != len(points) + 1:
        raise ValueError("Inconsistent slope model coefficients")
    values = float(row["intercept"]) + slopes[0]*x
    for i, point in enumerate(points):
        values += (slopes[i+1]-slopes[i])*np.maximum(0, x-point)
    return values


def run_geometry_comparison(paths, output, selected, *, labels=None, counter=None,
                            time=None, z=None, alpha=ALPHA_2D, lz=LZ,
                            skip_first=None, densify=None, upper_edge_bins=50, force=False):
    frames = select_comparison_frames(paths, labels, counter, time)
    z = lz/2 if z is None else z
    skip_first = (SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0) if skip_first is None else skip_first
    if not np.isfinite(z) or not np.isfinite(lz) or lz <= 0 or upper_edge_bins < 1 or skip_first < 0:
        raise ValueError("Invalid slice settings")
    tag = f"time_{time:g}" if time is not None else f"counter_{counter:06d}" if counter is not None else "latest"
    target = Path(output) / f"{tag}_z{z:g}"
    target.mkdir(parents=True, exist_ok=True)
    manifest = dict(requested_time=time, requested_counter=counter, z=z, alpha=alpha,
                    lz=lz, skip_first=skip_first, densify=densify, upper_edge_bins=upper_edge_bins,
                    pipelines=list(selected), cases=[])
    contours, edges, profiles, metrics, models = [], [], [], [], []
    contour_rows, edge_rows, profile_rows = [], [], []
    for index, (label, case, chosen, actual_time) in enumerate(frames):
        display = f"{label} (t={actual_time:.8g}, #{chosen})"
        print(f"Comparison: {display}")
        manifest["cases"].append(dict(label=label, path=str(case.case_root), counter=chosen, time=actual_time))
        particles = load_frame(case.trn, f"ell_trn_{chosen:06d}.bin")
        metadata = dict(case=label, counter=chosen, time=actual_time)
        if "2d" in selected or "slope" in selected:
            result = compute_slice_porosity(particles, z, alpha, lz,
                densify_max_segment_length=densify, n_bins=upper_edge_bins, skip_first=skip_first)
            if result is None or result["shape"] is None:
                raise ValueError(f"No alpha shape at z={z} for {label}")
            shape = np.asarray(result["shape"])
            contours.append((display, shape[:, 0], shape[:, 1]))
            contour_rows.append(pd.DataFrame(shape, columns=["x", "y"]).assign(**metadata))
            x, height = upper_edge_points(shape, n_bins=upper_edge_bins)
            if x is None:
                raise ValueError(f"No upper edge for {label}")
            edges.append((display, x, height))
            edge_rows.append(pd.DataFrame(dict(x=x, height=height)).assign(**metadata))
            metrics.append(dict(**metadata, z=z, **{key: result[key] for key in ("porosity", "A_ell", "A_alpha", "n")}))
            if "slope" in selected:
                # Isolate current-frame profiles from historical per-case analysis files.
                work = target / "cases" / f"case_{index:02d}"
                (work / "2d").mkdir(parents=True, exist_ok=True)
                pd.DataFrame(dict(x=x, z_max=height)).to_csv(work / "2d/alphashape_last_timestep.csv", index=False)
                run_slope(out=str(work), force=force, alpha=alpha)
                model_table = pd.read_csv(work / "slope_analysis/slope_results_R.csv")
                models.append((display, x, model_table.assign(**metadata)))
        if "slices" in selected:
            table = compute_2d_slice_summary(particles, Z_LEVELS, alpha=alpha, lz=lz,
                        skip_first=skip_first, densify=densify, upper_edge_bins=upper_edge_bins)
            profiles.append((display, table))
            profile_rows.append(table.assign(**metadata))

    title = f"z={z:g}, alpha={alpha:g}"
    if contours:
        _save_overlay(contours, target / "alphashapes_comparison", "x", "y", equal=True, title=title)
        _save_overlay(edges, target / "upper_edges_comparison", "x", "Surface height (y)", title=title)
        pd.concat(contour_rows, ignore_index=True).to_csv(target / "alphashapes.csv", index=False)
        pd.concat(edge_rows, ignore_index=True).to_csv(target / "upper_edges.csv", index=False)
        pd.DataFrame(metrics).to_csv(target / "porosity_2d.csv", index=False)
        with plt.rc_context(PLOT_STYLE):
            fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 5))
            ax.bar(np.arange(len(metrics)), [row["porosity"] for row in metrics])
            ax.set_xticks(np.arange(len(metrics)), [row[0] for row in contours], rotation=15, ha="right", fontsize=10)
            ax.set(ylabel="Porosity", title=title)
            plotstyle.prepare_figure(fig)
            fig.tight_layout()
            plotstyle.export_figure(fig, target / "porosity_2d_comparison.png", dpi=200)
            plt.close(fig)
    if profiles:
        pd.concat(profile_rows, ignore_index=True).to_csv(target / "slice_profiles.csv", index=False)
        for metric in ("porosity", "A_ell", "A_alpha"):
            _save_overlay([(label, table.z, table[metric]) for label, table in profiles],
                          target / f"{metric}_profile_comparison", "z", metric)
    if models:
        pd.concat([table for _, _, table in models], ignore_index=True).to_csv(target / "slope_models.csv", index=False)
        common = set.intersection(*(set(table.n_breakpoints) for _, _, table in models))
        if not common:
            raise ValueError("No common successful slope model across cases")
        for count in sorted(common):
            curves, predictions, angles = [], [], []
            for display, x, table in models:
                row = table.loc[table.n_breakpoints == count].iloc[0]
                grid = np.linspace(np.min(x), np.max(x), 300)
                values = slope_predictions(row, grid)
                curves.append((display, grid, values))
                predictions.append(pd.DataFrame(dict(x=grid, height=values)).assign(case=row["case"]))
                slopes = np.array([float(v) for v in str(row.slopes).split(";")])
                angles.append((display, np.arange(1, len(slopes)+1), np.degrees(np.arctan(slopes))))
            prefix = target / f"slope_{int(count)}bp_comparison"
            _save_overlay(curves, prefix, "x", "Fitted surface height (y)", title=title)
            pd.concat(predictions, ignore_index=True).to_csv(str(prefix) + ".csv", index=False)
            _save_overlay(angles, target / f"slope_{int(count)}bp_angles", "Segment (left to right)", "Slope angle [deg]")
    (target / "comparison.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if np.ptp([frame[3] for frame in frames]) > 1e-10:
        print("Note: actual case times differ; see legends and comparison.json. No states were merged.")
    print(f"Saved geometry comparisons: {target}")
    return target
