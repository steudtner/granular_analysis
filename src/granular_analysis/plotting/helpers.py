from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from granular_analysis.config import RESULTS_DIR


def save_figure(
    fig: plt.Figure,
    out_path: str | Path | None = None,
    *,
    default_name: str = "plot.png",
    dpi: int = 600,
) -> str:
    """Save a Matplotlib figure using Noto Sans and German decimal commas.

    - If *out_path* is None, a default path under ``RESULTS_DIR`` is used.
    - Ensures the parent directory exists.
    - Uses Noto Sans for regular text.
    - Uses Matplotlib's standard sans-serif math font for math expressions.
    - Formats numerical axis tick labels with German decimal commas.
    - Applies a tight bounding box.
    - Uses high DPI for raster formats.
    - Keeps text as text in SVG/PDF/EPS where possible.

    Returns
    -------
    str
        Path to the saved figure.
    """

    # ------------------------------------------------------------
    # Font settings
    # ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "Noto Sans",

        # General text sizes
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Use standard Matplotlib sans-serif math rendering.
        # Do NOT use custom Noto Sans mathtext mappings here.
        "mathtext.fontset": "dejavusans",

        # Keep fonts as text in vector formats
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ------------------------------------------------------------
    # German decimal formatter
    # ------------------------------------------------------------
    def german_decimal(value, _):
        """Format numerical tick labels with a decimal comma."""
        if value == 0:
            return "0"

        return f"{value:g}".replace(".", ",")

    formatter = FuncFormatter(german_decimal)

    # ------------------------------------------------------------
    # Apply decimal formatter to all axes
    # ------------------------------------------------------------
    for ax in fig.get_axes():
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    # ------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------
    out_path = (
        Path(out_path)
        if out_path is not None
        else Path(RESULTS_DIR) / default_name
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Output format
    # ------------------------------------------------------------
    out_format = (
        out_path.suffix.lower().lstrip(".")
        or "eps"
    )

    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0,
    }

    if out_format in {"png", "jpg", "jpeg"}:
        save_kwargs["dpi"] = dpi

    if out_format == "svg":
        save_kwargs["transparent"] = False

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    fig.savefig(
        out_path,
        format=out_format,
        **save_kwargs,
    )

    plt.close(fig)

    return str(out_path)