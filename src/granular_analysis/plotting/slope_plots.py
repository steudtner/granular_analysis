"""Plotting routines for segmented slope analysis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_single_model(
    x, z, model, out_path, xlim=(0, 60), ylim=(0, 20),
    color="blue", dpi=600
):
    """
    Create a single publication-quality plot for one model.

    Parameters
    ----------
    x : array-like
        Data x-values.
    z : array-like
        Data z-values.
    model : dict
        Model dict from ``fit_segmented_models``.
    out_path : str
        Output file path.
    xlim, ylim : tuple
        Axis limits.
    color : str
        Line color for the model fit.
    dpi : int
        Resolution.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3))

    # --- scatter ---
    ax.scatter(x, z, s=10, c="black", zorder=1)

    # --- grid ---
    xticks = np.arange(xlim[0], xlim[1] + 0.01, 5)
    yticks = np.arange(ylim[0], ylim[1] + 0.01, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, which="major", linestyle=":", color="lightgray",
            linewidth=0.5, zorder=0)

    # --- axis spines: only left and bottom (bty = "l") ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- model line ---
    n_bp = len(model["breakpoints"])
    if n_bp == 0:
        # Simple linear: draw straight line across full x-range
        x_line = np.linspace(xlim[0], xlim[1], 100)
        y_line = model["intercept"] + model["slopes"][0] * x_line
        ax.plot(x_line, y_line, color=color, linewidth=3, zorder=2)
    else:
        # Predict on a fine grid
        x_line = np.linspace(xlim[0], xlim[1], 500)
        y_line = model["intercept"] + model["slopes"][0] * x_line
        for i, bp in enumerate(model["breakpoints"]):
            y_line = y_line + (model["slopes"][i + 1] - model["slopes"][i]) * \
                     np.maximum(0, x_line - bp)
        ax.plot(x_line, y_line, color=color, linewidth=3, zorder=2)

        # Breakpoint vertical lines
        for bp in model["breakpoints"]:
            ax.axvline(x=bp, color="black", linestyle="--",
                       linewidth=1.5, zorder=1)

    # --- slope / angle annotation (upper right) ---
    slopes = model["slopes"]
    angles = model["angles"]
    y_start = ylim[1]
    y_gap = (ylim[1] - ylim[0]) * 0.07
    for i, (m, a) in enumerate(zip(slopes, angles)):
        label = (
            rf"$m_{{{i+1}}} = {m:.2f},\; "
            rf"\alpha_{{{i+1}}} = {a:.1f}^\circ$"
        )
        ax.text(
            xlim[1], y_start - i * y_gap,
            label,
            ha="right", va="top", fontsize=10,
        )

    # --- AIC / BIC (left side) ---
    ax.text(
        xlim[0], np.mean(ylim),
        f"AIC={model['aic']:.1f}, BIC={model['bic']:.1f}",
        ha="left", va="center", color="red", fontsize=9,
    )

    # --- labels ---
    ax.set_xlabel(r"$x / D_{Eq}$", fontsize=14)
    ax.set_ylabel(r"$z / D_{Eq}$", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
