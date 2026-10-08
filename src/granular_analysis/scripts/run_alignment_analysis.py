"""Write per-particle alignment angles and a single-frame histogram."""

from granular_analysis.plotting import plotstyle

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from granular_analysis.analysis.alignment import particle_alignment, enrich_alignment
from granular_analysis.utils.physics import load_mass_array
from granular_analysis.io.loader import load_frame
from granular_analysis.paths import latest_trn_counter
from granular_analysis.plotting.helpers import PLOT_STYLE, save_figure


def run_alignment(*, case, out_dir, counter=None, mode="prolate", min_speed=0.0,
                  cluster_eps=0.3, cluster_min_samples=20):
    counter = counter if counter is not None else latest_trn_counter(case)
    if counter is None:
        raise FileNotFoundError(f"No particle frames in {case.trn}")
    particles = load_frame(case.require_trn(), f"ell_trn_{counter:06d}.bin")
    table = particle_alignment(particles, mode=mode, min_speed=min_speed)
    mass_path = case.case_root / "input_ellipsoid_data.dat"
    masses = load_mass_array(mass_path) if mass_path.exists() else None
    mass_warning = None
    if masses is not None and len(masses) != len(particles):
        mass_warning = f"Input has {len(masses)} particles, TRN has {len(particles)}; energy weighting omitted (uncertain mass mapping)."
        print(f"WARNING: {mass_warning}")
        masses = None
    table, metadata = enrich_alignment(table, particles, masses, cluster_eps, cluster_min_samples)
    metadata.update(counter=counter, mode=mode, min_speed=min_speed,
                    mass_source=str(mass_path) if masses is not None else None, mass_warning=mass_warning)
    target = Path(out_dir) / "alignment"
    target.mkdir(parents=True, exist_ok=True)
    prefix = target / f"alignment_{mode}_{counter:06d}"
    table.to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = table.groupby("cluster").agg(count=("particle_row", "size"),
                                          mean_angle=("angle_deg", "mean"),
                                          mean_speed=("speed", "mean"),
                                          median_speed=("speed", "median"))
    summary.to_csv(str(prefix) + "_clusters.csv")
    angles = table["angle_deg"].dropna()
    lower, upper = 0, 90
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 4))
        ax.hist(angles, bins=np.linspace(lower, upper, 37), color="#238b8d", edgecolor="white")
        ax.set(xlim=(lower, upper), xlabel=("Angle to longest axis [deg]" if mode == "prolate"
                                         else "Angle to c axis [deg]"), ylabel="Particle count",
               title=f"Counter {counter:06d}, t={particles[0]['t']:.8g}" if particles else f"Counter {counter:06d}")
        plotstyle.prepare_figure(fig)
        fig.tight_layout()
        plotstyle.export_figure(fig, prefix.with_suffix(".png"), dpi=200)
        plt.close(fig)
        plot_diagnostics(table, prefix, mode)
    print(f"Alignment: {len(angles)}/{len(table)} valid particles; excluded {len(table) - len(angles)}.")
    if len(angles):
        print(f"Mean angle = {angles.mean():.3f} deg; median = {angles.median():.3f} deg")
    print(f"Saved CSV, PNG and SVG: {prefix}")
    labels = table.cluster
    print(f"Clusters: {labels[labels >= 0].nunique()}; noise: {(labels == -1).sum()}; excluded: {(labels == -2).sum()}")
    if masses is None and mass_warning is None:
        print("No input_ellipsoid_data.dat: energy-weighted distribution omitted.")
    return table


def plot_diagnostics(table, prefix, mode):
    valid = table.loc[table.status == "valid"]
    fig, axes = plt.subplots(2, 3, figsize=(plotstyle.figwidth_full, 8), constrained_layout=True)
    angle_label = "Angle [deg]" if mode == "prolate" else "Angle to c [deg]"
    edges = np.linspace(0, 90, 31)
    for weights, label in [(np.ones(len(valid)), "Count"), (valid.speed.to_numpy(), "Speed"),
                           (valid.kinetic_energy.to_numpy(), "Kinetic energy")]:
        if len(weights) and np.isfinite(weights).all() and weights.sum() > 0:
            axes[0, 0].hist(valid.angle_deg, bins=edges, weights=weights / weights.sum(),
                            histtype="step", label=label)
    theta = np.linspace(0, 90, 200)
    axes[0, 0].plot(theta, np.sin(np.radians(theta))*np.pi/180*3,
                    "k--", label="Isotropic 3D")
    axes[0, 0].set(xlabel=angle_label, ylabel="Fraction per 3 deg bin")
    axes[0, 0].legend()
    if len(valid):
        axes[0, 1].hist(valid.abs_cos_angle, bins=np.linspace(0, 1, 21), density=True, color="#238b8d")
        mesh = axes[0, 2].hexbin(valid.angle_deg, valid.speed, gridsize=35, mincnt=1)
        fig.colorbar(mesh, ax=axes[0, 2], label="Particle count")
    axes[0, 1].axhline(1, color="black", linestyle="--", label="Isotropic 3D")
    axes[0, 1].set(xlabel="|cos(angle to axis)|", ylabel="Probability density", xlim=(0, 1))
    axes[0, 1].legend()
    axes[0, 2].set(xlabel=angle_label, ylabel="Speed")
    for label in sorted(table.cluster.unique()):
        group = table.loc[table.cluster == label]
        color = "#bbbbbb" if label == -2 else "#555555" if label == -1 else plt.get_cmap("tab20")(int(label) % 20)
        name = "Excluded" if label == -2 else "Noise" if label == -1 else f"Cluster {label}"
        axes[1, 0].scatter(group.v_axis, group.v_plane, s=5, color=color, alpha=.6)
        axes[1, 1].scatter(group.angle_deg, group.speed, s=5, color=color, alpha=.6)
        axes[1, 2].scatter(group.x, group.y, s=5, color=color, alpha=.6, label=name)
    axes[1, 0].set(xlabel="Speed along selected axis", ylabel="Speed in normal plane", aspect="equal")
    axes[1, 1].set(xlabel=angle_label, ylabel="Speed")
    axes[1, 2].set(xlabel="x", ylabel="y", aspect="equal")
    if len(table):
        axes[1, 2].legend(fontsize=10, markerscale=2)
    fig.suptitle(prefix.name)
    plotstyle.export_figure(fig, str(prefix) + "_diagnostics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(plotstyle.figwidth_full, 4), constrained_layout=True)
    for ax, (x, y) in zip(axes, [("x", "y"), ("x", "z"), ("y", "z")]):
        for label in sorted(table.cluster.unique()):
            group = table.loc[table.cluster == label]
            color = "#bbbbbb" if label == -2 else "#555555" if label == -1 else plt.get_cmap("tab20")(int(label) % 20)
            ax.scatter(group[x], group[y], s=5, color=color, alpha=.6)
        ax.set(xlabel=x, ylabel=y, aspect="equal")
    plotstyle.export_figure(fig, str(prefix) + "_spatial.png", dpi=180)
    plt.close(fig)
