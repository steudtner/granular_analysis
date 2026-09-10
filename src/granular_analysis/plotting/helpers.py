from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
from granular_analysis.config import RESULTS_DIR


def save_figure(fig: plt.Figure, out_path: str | Path | None = None, *, default_name: str = "plot.png", dpi: int = 600) -> str:
    """Save a Matplotlib figure with sane project defaults.

    - If *out_path* is None, a default path under ``RESULTS_DIR`` is used.
    - Ensures parent directory exists.
    - Applies tight bbox and high DPI for raster formats.

    Returns the string path to the saved file.
    """
    out_path = Path(out_path) if out_path is not None else Path(RESULTS_DIR) / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_format = out_path.suffix.lower().lstrip(".") or "eps"
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0}
    if out_format in {"png", "jpg", "jpeg"}:
        save_kwargs["dpi"] = dpi
    if out_format == "svg":
        save_kwargs["transparent"] = False

    fig.savefig(out_path, format=out_format, **save_kwargs)
    plt.close(fig)
    return str(out_path)
