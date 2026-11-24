
from granular_analysis.plotting import plotstyle
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from granular_analysis.paths import CasePaths
from granular_analysis.geometry.alpha_logging import summarize_alpha_warnings

from granular_analysis.config import (
    RESULTS_DIR,
    Z_LEVELS,
    LZ,
    ALPHA_2D,
    FOLDER,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
)
from granular_analysis.io.loader import load_frame, load_last_frame
from granular_analysis.analysis.porosity2d import (
    compute_slice_porosity,
)
from granular_analysis.utils.state_manager import build_fingerprint, load_state, save_state


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_particles(data_folder, counter=None):
    """Load the selected TRN frame without applying hidden filtering."""
    if counter is None:
        return load_last_frame(data_folder)
    return load_frame(data_folder, f"ell_trn_{int(counter):06d}.bin")


# --------------------------------------------------
# 2D DATA
# --------------------------------------------------

@summarize_alpha_warnings()
def compute_2d_slice_summary(
    particles,
    z_levels,
    output_dir=None,
    *,
    alpha=ALPHA_2D,
    lz=LZ,
    skip_first=0,
    densify=None,
    upper_edge_bins=50,
):
    records = []
    for z in z_levels:
        result = compute_slice_porosity(
            particles,
            z,
            alpha,
            lz,
            densify_max_segment_length=densify,
            n_bins=upper_edge_bins,
            skip_first=skip_first,
            output_dir=output_dir,
            write_upper_edge=False,
        )
        if result is None:
            records.append({
                "z": z,
                "A_ell": np.nan,
                "A_alpha": np.nan,
                "porosity": np.nan,
                "n": 0,
            })
            continue
        records.append({
            "z": z,
            "A_ell": result["A_ell"],
            "A_alpha": result["A_alpha"],
            "porosity": result["porosity"],
            "n": result["n"],
        })
    return pd.DataFrame(records)


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def _plot_profile(
    df,
    columns,
    labels,
    linestyles,
    title,
    ylabel,
    save_path,
    ylim=None,
    linewidth=1.5,
):

    fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 5))

    for col, label, linestyle in zip(
        columns,
        labels,
        linestyles,
    ):

        if col in df.columns:

            ax.plot(
                df["z"],
                df[col],
                linestyle,
                linewidth=linewidth,
                label=label,
            )

    ax.set_xlabel(
        r"$z / D_{\mathrm{Eq}}$"
    )

    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plotstyle.legend_above(ax)

    plotstyle.prepare_figure(fig)

    fig.tight_layout()

    plotstyle.export_figure(fig,
        save_path,
        dpi=200,
    )

    plt.close(fig)

    return fig


def plot_area_profile(df, save_path):

    return _plot_profile(
        df,
        columns=[
            "A_ell",
            "A_alpha",
        ],
        labels=[
            r"$A_{\mathrm{ell}}$",
            r"$A_{\alpha}$",
        ],
        linestyles=[
            "-",
            "-",
        ],
        linewidth=1.2,
        title="Area Profile",
        ylabel="Area",
        save_path=save_path,
    )


def plot_porosity_profile(df, save_path):

    return _plot_profile(
        df,
        columns=[
            "porosity",
        ],
        labels=[
            "2D porosity",
        ],
        linestyles=[
            "-",
        ],
        linewidth=1.2,
        title="Porosity Profile",
        ylabel=r"$\Phi$",
        save_path=save_path,
        ylim=(0.3, 0.6),
    )


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

def print_statistics(df):

    if "A_alpha" in df.columns:

        p = df["A_alpha"].dropna()

        if len(p):

            print(
                f"A_alpha: mean={p.mean():.4f}, "
                f"std={p.std():.4f}"
            )

    if "porosity" in df.columns:

        p = df["porosity"].dropna()

        if len(p):

            print(
                f"Porosity: mean={p.mean():.4f}, "
                f"std={p.std():.4f}"
            )

# --------------------------------------------------
# PATH RESOLUTION
# --------------------------------------------------

def resolve_data_folder(data_folder):
    """Resolve and validate the canonical transient-data directory."""
    return str(CasePaths.from_input(data_folder).require_trn())

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main(
    force: bool = False,
    path: str = None,
    out: str = None,
    z_value: float = None,
    *,
    alpha: float = ALPHA_2D,
    skip_first: int = None,
    densify: float = None,
    upper_edge_bins: int = 50,
    counter: int = None,
):

    # --------------------------------------------------
    # INPUT PATH
    # --------------------------------------------------

    data_folder = (
        path
        if path
        else os.environ.get(
            "GRANULAR_DATA_FOLDER"
        )
        or FOLDER
    )

    data_folder = resolve_data_folder(
        data_folder
    )

    # --------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------

    results_dir = (
        os.path.abspath(out)
        if out
        else os.path.abspath(RESULTS_DIR)
    )

    outdir = os.path.join(
        results_dir,
        "slices",
    )

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Z LEVELS
    # --------------------------------------------------

    if z_value is not None:

        z_levels = [
            float(z_value)
        ]

    else:

        z_levels = Z_LEVELS

    print(f"2D slices: {len(z_levels)} level(s), alpha={alpha:g}")

    # --------------------------------------------------
    # CACHE
    # --------------------------------------------------

    effective_skip = (
        skip_first
        if skip_first is not None
        else SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0
    )
    frame = (
        Path(data_folder) / f"ell_trn_{int(counter):06d}.bin"
        if counter is not None
        else max(Path(data_folder).glob("ell_trn_*.bin"))
    )
    frame_stat = frame.stat()
    cache_fingerprint = build_fingerprint(
        frame=str(frame.resolve()),
        frame_size=frame_stat.st_size,
        frame_mtime_ns=frame_stat.st_mtime_ns,
        z_levels=[float(value) for value in z_levels],
        alpha=alpha,
        lz=LZ,
        skip_first=effective_skip,
        densify=densify,
        upper_edge_bins=upper_edge_bins,
    )
    cached = None if force else load_state(
        "slices", results_dir, fingerprint=cache_fingerprint
    )

    csv_path = os.path.join(
        outdir,
        "slice_summary.csv",
    )

    if (
        cached
        and os.path.exists(csv_path)
    ):

        print("Using cached slice results.")

        df = pd.read_csv(
            csv_path
        )

    else:

        # --------------------------------------------------
        # LOAD PARTICLES
        # --------------------------------------------------

        particles = load_particles(data_folder, counter=counter)

        # --------------------------------------------------
        # COMPUTE
        # --------------------------------------------------

        print(f"Computing slices for {len(particles)} particles...")

        df = compute_2d_slice_summary(
            particles,
            z_levels,
            output_dir=outdir,
            alpha=alpha,
            skip_first=effective_skip,
            densify=densify,
            upper_edge_bins=upper_edge_bins,
        )

        # --------------------------------------------------
        # SAVE CSV
        # --------------------------------------------------

        df.to_csv(
            csv_path,
            index=False,
        )

        # --------------------------------------------------
        # CACHE
        # --------------------------------------------------

        stats = {
            "n_levels": len(df),
            "porosity_mean": float(
                df["porosity"].mean()
            ),
            "porosity_std": float(
                df["porosity"].std()
            ),
            "A_alpha_mean": (
                float(
                    df["A_alpha"].mean()
                )
                if "A_alpha" in df.columns
                else None
            ),
        }

        save_state(
            "slices",
            stats,
            results_dir,
            fingerprint=cache_fingerprint,
        )

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------

    area_png = os.path.join(
        outdir,
        "area_profile.png",
    )

    porosity_png = os.path.join(
        outdir,
        "porosity_profile.png",
    )

    plot_area_profile(
        df,
        area_png,
    )

    plot_porosity_profile(
        df,
        porosity_png,
    )

    print_statistics(df)

    print(f"Saved slice summary and profiles to: {outdir}")


if __name__ == "__main__":
    main()
