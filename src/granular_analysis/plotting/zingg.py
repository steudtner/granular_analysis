"""Scatter plot of input-particle shape in the Zingg diagram."""

from granular_analysis.plotting import plotstyle
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def plot_zingg(table, output_path, *, color_by="Sf"):
    """Plot every particle at x=I/L, y=S/I; color by the selected shape metric."""
    if color_by not in ("Sf", "wadell_sphericity"):
        raise ValueError("color_by must be Sf or wadell_sphericity")
    figure = Figure(figsize=(plotstyle.figwidth_full, 6), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    options = dict(s=12, alpha=0.6, edgecolors="none")
    if color_by in table:
        options.update(c=table[color_by], cmap="viridis", vmin=0, vmax=1)
    points = ax.scatter(table["I_over_L"], table["S_over_I"], **options)
    if color_by in table:
        label = "Corey $S_f = S / (L I)^{1/2}$" if color_by == "Sf" else "Wadell sphericity"
        figure.colorbar(points, ax=ax, label=label)
    ax.axvline(2 / 3, color="0.4", linestyle="--", linewidth=1)
    ax.axhline(2 / 3, color="0.4", linestyle="--", linewidth=1)
    for x, y, label in [(0.33, 0.94, "Rod / prolate"), (0.83, 0.94, "Equant"),
                        (0.33, 0.08, "Blade"), (0.83, 0.08, "Disk / oblate")]:
        ax.text(x, y, label, ha="center", va="center", color="0.3")
    ax.set(xlabel="Intermediate / long axis (I/L)", ylabel="Short / intermediate axis (S/I)",
           xlim=(0, 1.02), ylim=(0, 1.02), title=f"Zingg diagram — {len(table)} input particles")
    ax.set_aspect("equal")
    ax.set_xticks([0, 1/3, 2/3, 1], ["0", "1/3", "2/3", "1"])
    ax.set_yticks([0, 1/3, 2/3, 1], ["0", "1/3", "2/3", "1"])
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plotstyle.export_figure(figure, target)
    return target
