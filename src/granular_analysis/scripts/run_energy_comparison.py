"""Overlay energy curves without interpolating or extrapolating case data."""

from granular_analysis.plotting import plotstyle

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from granular_analysis.paths import CasePaths
from granular_analysis.plotting.helpers import PLOT_STYLE, save_figure
from granular_analysis.plotting.plots import add_potential_reference, smooth_energy_curve
from granular_analysis.scripts.run_energy_analysis import main as run_energy


def plot_energy_comparison(tables, output, component="kinetic", normalized=True, potential_energy=False, smoothing_window=9):
    if component not in {"kinetic", "translational", "rotational"}:
        raise ValueError("Unknown energy component")
    x = "time_norm" if normalized else "time"
    y = f"{component}_energy" + ("_norm" if normalized else "")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / (f"{component}_energy" + ("_normalized" if normalized else "") + "_comparison")
    combined = []
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 5))
        try:
            for label, table in tables.items():
                if table.empty or not {x, y}.issubset(table.columns):
                    raise ValueError(f"Missing or empty {x}/{y} data for {label}")
                curve = table[[x, y]].copy()
                if curve.isna().any().any() or not curve[x].is_monotonic_increasing or curve[x].duplicated().any():
                    raise ValueError(f"Invalid or non-increasing energy series for {label}")
                line, = ax.plot(curve[x], smooth_energy_curve(curve[y], smoothing_window), label=label, linewidth=0.8,
                                markevery=max(1, len(curve)//12), **plotstyle.case_style(label))
                if potential_energy:
                    add_potential_reference(ax, table, normalized, color=line.get_color(),
                                            label=f"{label}: initial potential (input y, y=0)")
                combined.append(table.assign(case=label))
            ax.set_xlabel(r"$t / (D_{Eq} / g)^{1/2}$" if normalized else "Simulation time")
            ax.set_ylabel(f"{component.capitalize()} energy" + (r" $E / (g\rho D_{Eq}^{4})$" if normalized else ""))
            ax.grid(alpha=.25)
            ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
            if smoothing_window > 1:
                ax.set_title(f"Centered mean: {smoothing_window} samples", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plotstyle.prepare_figure(fig)
            fig.tight_layout()
            plotstyle.export_figure(fig, prefix.with_suffix(".png"), dpi=200)
            plt.close(fig)
        finally:
            plt.close(fig)
    pd.concat(combined, ignore_index=True).to_csv(prefix.with_suffix(".csv"), index=False)
    print(f"Saved energy comparison: {prefix} (.png, .svg, .csv)")
    return prefix


def run_energy_comparison(cases, output, *, labels=None, component="kinetic", normalized=True, force=False, skip_first=None, potential_energy=False, rotation_share_threshold=0.5):
    resolved = [CasePaths.from_input(path) for path in cases]
    roots = [case.case_root.resolve() for case in resolved]
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("Energy comparison requires at least two distinct cases")
    labels = labels or [root.name for root in roots]
    if len(labels) != len(roots) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique --compare-labels entry per case")
    tables = {}
    for label, case in zip(labels, resolved):
        csv = case.case_root / "results_granular_analysis" / "energy.csv"
        print(f"Energy comparison: {label} ({case.case_root})")
        run_energy(path=str(case.trn), out=str(csv), force=force, skip_first=skip_first, potential_energy=potential_energy, rotation_share_threshold=rotation_share_threshold)
        tables[label] = pd.read_csv(csv)
    prefix = plot_energy_comparison(tables, output, component, normalized, potential_energy)
    prefix.with_suffix(".json").write_text(json.dumps(dict(
        cases=dict(zip(labels, map(str, roots))), component=component,
        normalized=normalized, potential_energy=potential_energy, rotation_share_threshold=rotation_share_threshold, display_smoothing="centered mean, 9 samples; CSV unchanged", time_alignment="native samples; no interpolation or extrapolation",
    ), indent=2) + "\n")
    return prefix
