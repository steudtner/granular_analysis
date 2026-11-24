
from granular_analysis.plotting import plotstyle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_shape(shape, out, title=""):
    fig, ax = plt.subplots()
    if shape is not None:
        ax.plot(shape[:, 0], shape[:, 1])
        ax.fill(shape[:, 0], shape[:, 1], alpha=0.2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid()
    plotstyle.prepare_figure(plt.gcf())
    plt.tight_layout()
    plotstyle.export_figure(plt.gcf(), out, dpi=600)
    plt.close()


def plot_kinetic_energy(E, out, particle_ids=None, title="Kinetic energy per particle"):
    fig, ax = plt.subplots()
    x = np.arange(len(E)) if particle_ids is None else np.asarray(particle_ids)
    ax.plot(x, E, "-o")
    ax.set_xlabel("Particle")
    ax.set_ylabel("Kinetic energy")
    ax.set_title(title)
    ax.grid()
    plotstyle.prepare_figure(plt.gcf())
    plt.tight_layout()
    plotstyle.export_figure(plt.gcf(), out, dpi=600)
    plt.close()


def add_potential_reference(ax, df, normalized=False, *, color="black", label=None):
    column = "initial_potential_energy" + ("_norm" if normalized else "")
    if column in df and len(df) and np.isfinite(df[column].iloc[0]):
        ax.axhline(df[column].iloc[0], color=color, linestyle="--", linewidth=1.2,
                   label=label or "Initial potential energy (input y, reference y=0)")


def smooth_energy_curve(values, window=9):
    """Centered sample mean for display only; never bridge nonfinite gaps.

    Window is an odd number of output samples (not physical time). At segment
    boundaries only available samples contribute. Use window=1 to disable.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1:
        raise ValueError("energy curve must be one-dimensional")
    finite = np.isfinite(result)
    result[~finite] = np.nan
    if window == 1:
        return result
    edges = np.diff(np.r_[False, finite, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        segment = result[start:stop].copy()
        radius = window // 2
        indices = np.arange(len(segment))
        left = np.maximum(0, indices - radius)
        right = np.minimum(len(segment), indices + radius + 1)
        cumulative = np.r_[0.0, np.cumsum(segment)]
        result[start:stop] = (cumulative[right] - cumulative[left]) / (right - left)
    return result


def energy_display_points(x, y, smoothing_window=9, max_points=300):
    """Mean adjacent samples for display, with endpoints and gaps retained.

    The limit applies per finite segment. None disables point reduction.
    CSV values are never modified; bins are sample-based, not time-weighted.
    """
    if max_points is not None and (isinstance(max_points, bool) or
            not isinstance(max_points, (int, np.integer)) or max_points < 3):
        raise ValueError("max_points must be an integer >= 3 or None")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).copy()
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be one-dimensional with equal length")
    y[~np.isfinite(x)] = np.nan
    y = smooth_energy_curve(y, smoothing_window)
    finite = np.isfinite(x) & np.isfinite(y)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    xs, ys = [], []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if xs:
            xs.append(np.nan)
            ys.append(np.nan)
        if max_points is None or stop - start <= max_points:
            xs.extend(x[start:stop])
            ys.extend(y[start:stop])
        else:
            xs.append(x[start]); ys.append(y[start])
            for indices in np.array_split(np.arange(start + 1, stop - 1), max_points - 2):
                xs.append(np.mean(x[indices]))
                ys.append(np.mean(y[indices]))
            xs.append(x[stop - 1]); ys.append(y[stop - 1])
    return np.asarray(xs), np.asarray(ys)


def _energy_smoothing_note(ax, window, max_points=300):
    notes = []
    if window > 1:
        notes.append(f"Centered mean: {window} samples")
    if max_points is not None:
        notes.append(f"Binned means: up to {max_points} points/segment")
    if notes:
        ax.text(0.99, 0.98, " | ".join(notes),
                transform=ax.transAxes, ha="right", va="top", fontsize=10)


def plot_energy_time_series(df, out, use_normalized=False, potential_energy=False, smoothing_window=9, max_points=300):
    fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 4))

    if use_normalized and "time_norm" in df.columns and "kinetic_energy_norm" in df.columns:
        x = df["time_norm"]
        y = df["kinetic_energy_norm"]
        ax.set_xlabel(r"$t / (D_{\mathrm{Eq}} / g)^{1/2}$", fontsize=10)
        ax.set_ylabel(r"$\sum E_{kin}  / (g\rho D_{\mathrm{Eq}}^{4})$", fontsize=10)
        ax.set_title("Normalized kinetic energy over time", fontsize=10)
        """
         if len(y) > 0:
            E_max = y.max()
            threshold = 0.03 * E_max
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=r"$\epsilon = 3\% \sum E_{kin,max}$"
            ) """
    else:
        x = df["time"]
        y = df["kinetic_energy"]
        ax.set_xlabel("time", fontsize=10)
        ax.set_ylabel("kinetic energy", fontsize=10)
        ax.set_title("Total kinetic energy over time", fontsize=10)

    ax.plot(*energy_display_points(x, y, smoothing_window, max_points), linewidth=1.2, label=r"$\sum E_{kin}$")
    _energy_smoothing_note(ax, smoothing_window, max_points)
    if potential_energy:
        add_potential_reference(ax, df, use_normalized and "kinetic_energy_norm" in df)
    ax.grid(True)
    ax.legend(fontsize=10)
    plotstyle.prepare_figure(plt.gcf())
    plt.tight_layout()
    plotstyle.export_figure(plt.gcf(), out, dpi=300)
    plt.close()


def plot_energy_components(df, out, use_normalized=False, potential_energy=False, smoothing_window=9, max_points=300):
    """Show components separately, retaining each component's visible scale."""
    normalized = use_normalized and "time_norm" in df.columns
    x = df["time_norm" if normalized else "time"]
    suffix = "_norm" if normalized else ""
    fig, axes = plt.subplots(3, 1, figsize=(plotstyle.figwidth_full, 8), sharex=True)
    components = [("kinetic", "Total", "#222222"),
                  ("translational", "Translation", "#0072B2"),
                  ("rotational", "Rotation", "#D55E00")]
    for ax, (key, label, color) in zip(axes, components):
        ax.plot(*energy_display_points(x, df[f"{key}_energy{suffix}"], smoothing_window, max_points),
                color=color, linewidth=1.2)
        ax.set_ylabel(label + (r" $E / (g\rho D_{\mathrm{Eq}}^{4})$" if normalized else " energy"))
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)
    if potential_energy:
        add_potential_reference(axes[0], df, normalized)
        axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.3), frameon=False, fontsize=10)
    title = "Kinetic energy components (separate vertical scales)"
    if smoothing_window > 1:
        title += f" — centered mean, {smoothing_window} samples"
    if max_points is not None:
        title += f"\nBinned means: up to {max_points} points/segment"
    fig.suptitle(title, fontsize=10)
    axes[-1].set_xlabel(r"$t / (D_{\mathrm{Eq}} / g)^{1/2}$" if normalized else "time")
    plotstyle.prepare_figure(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plotstyle.export_figure(fig, out, dpi=300)
    plt.close(fig)


def plot_energy_ratio(df, out, use_normalized=False, smoothing_window=9, max_points=300):
    """Plot the dimensionless ratio of summed translational/rotational energy."""
    fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 4))
    normalized = use_normalized and "time_norm" in df.columns
    ax.plot(*energy_display_points(df["time_norm" if normalized else "time"],
            df["translational_rotational_ratio"], smoothing_window, max_points), linewidth=1.2)
    _energy_smoothing_note(ax, smoothing_window, max_points)
    ax.set_xlabel(r"$t / (D_{\mathrm{Eq}} / g)^{1/2}$" if normalized else "time")
    ax.set_ylabel(r"$\sum E_{\mathrm{trans}} / \sum E_{\mathrm{rot}}$")
    ax.axhline(1.0, color="gray", linestyle="--", label="Equal energies")
    ax.set_title("Translational / rotational energy over time")
    ax.grid(True)
    ax.legend()
    plotstyle.prepare_figure(fig)
    fig.tight_layout()
    plotstyle.export_figure(fig, out, dpi=300)
    plt.close(fig)


def plot_energy_histogram(particle_energies, timestep_label, out_path, normalization=1.0):
    E_norm = particle_energies / normalization

    fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 4))
    ax.hist(E_norm, bins=60)
    ax.set_yscale("log")
    ax.set_xlabel(r"$E^* = E  / (g\rho D_{eq}^{4})$", fontsize=10)
    ax.set_ylabel("count (log)", fontsize=10)
    ax.set_title(f"Normalized Energy Distribution\n{timestep_label}", fontsize=10)
    ax.grid(True, which="both")
    plotstyle.prepare_figure(plt.gcf())
    plt.tight_layout()
    plotstyle.export_figure(plt.gcf(), out_path, dpi=300)
    plt.close()


def plot_profile(df, out):
    fig, ax = plt.subplots()
    ax.plot(df["z"], df["porosity"], "-o")
    ax.set_xlabel("z")
    ax.set_ylabel("porosity")
    ax.grid()
    plotstyle.prepare_figure(plt.gcf())
    plt.tight_layout()
    plotstyle.export_figure(plt.gcf(), out, dpi=600)
    plt.close()
