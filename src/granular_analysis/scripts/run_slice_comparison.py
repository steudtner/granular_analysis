import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from granular_analysis.config import (
    RESULTS_DIR,
    Z_LEVELS,
    LZ,
    ALPHA_2D,
    FOLDER,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
)
from granular_analysis.io.loader import (
    load_last_frame,
    filter_particles,
)
from granular_analysis.analysis.porosity2d import (
    compute_slice_porosity,
)
from granular_analysis.utils.state_manager import load_state, save_state


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_particles(data_folder):
    ellipsoids = load_last_frame(data_folder)

    return filter_particles(
        ellipsoids,
        USE_PARTICLE_SKIP,
        SKIP_FIRST_PARTICLES,
    )


# --------------------------------------------------
# 2D DATA
# --------------------------------------------------

def compute_2d_slice_summary(
    particles,
    z_levels,
    output_dir=None,
):
    records = []

    for z in z_levels:

        result = compute_slice_porosity(
            particles,
            z,
            ALPHA_2D,
            LZ,
            output_dir=output_dir,
        )

        if result is None:

            records.append(
                {
                    "z": z,
                    "A_ell": np.nan,
                    "A_alpha": np.nan,
                    "porosity": np.nan,
                    "n": 0,
                }
            )

            continue

        records.append(
            {
                "z": z,
                "A_ell": result["A_ell"],
                "A_alpha": result["A_alpha"],
                "porosity": result["porosity"],
                "n": result["n"],
            }
        )

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

    fig, ax = plt.subplots(figsize=(8, 5))

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
    ax.legend()

    fig.tight_layout()

    fig.savefig(
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
        linewidth=1.5,
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
        linewidth=1.5,
        title="Porosity Profile",
        ylabel=r"$\Phi$",
        save_path=save_path,
        ylim=(0.3, 0.6),
    )


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

def print_statistics(df):

    print("\n=== AREA STATISTICS ===")

    if "A_alpha" in df.columns:

        p = df["A_alpha"].dropna()

        if len(p):

            print(
                f"A_alpha: mean={p.mean():.4f}, "
                f"std={p.std():.4f}"
            )

    print("\n=== POROSITY STATISTICS ===")

    if "porosity" in df.columns:

        p = df["porosity"].dropna()

        if len(p):

            print(
                f"2D: mean={p.mean():.4f}, "
                f"std={p.std():.4f}"
            )


# --------------------------------------------------
# PATH RESOLUTION
# --------------------------------------------------

def resolve_data_folder(data_folder):
    """
    Resolve the supplied case/results path to the directory
    containing the ellipsoid TRN binary files.

    Accepted input forms:

        .../results
        .../results/ellipsoid
        .../results/ellipsoid/ellipsoid_trn
        .../ellipsoid_trn
    """

    data_folder = os.path.abspath(data_folder)

    candidates = [
        data_folder,
        os.path.join(
            data_folder,
            "ellipsoid_trn",
        ),
        os.path.join(
            data_folder,
            "ellipsoid",
            "ellipsoid_trn",
        ),
        os.path.join(
            data_folder,
            "ellipsoid",
        ),
    ]

    for candidate in candidates:

        if not os.path.isdir(candidate):
            continue

        try:
            files = os.listdir(candidate)
        except OSError:
            continue

        if any(
            filename.endswith(".bin")
            for filename in files
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find ellipsoid TRN .bin files. "
        f"Checked starting from: {data_folder}"
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main(
    force: bool = False,
    path: str = None,
    out: str = None,
    z_value: float = None,
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

    print("=" * 60)
    print("2D Slice Analysis")
    print("=" * 60)

    print(
        f"DATA   = {data_folder}"
    )

    print(
        f"OUTPUT = {outdir}"
    )

    if z_value is not None:

        print(
            f"Z      = {z_value}"
        )

    else:

        print(
            f"Z LEVELS = {len(z_levels)}"
        )

    # --------------------------------------------------
    # CACHE
    # --------------------------------------------------

    cached = (
        None
        if force
        else load_state(
            "slices",
            results_dir,
        )
    )

    csv_path = os.path.join(
        outdir,
        "slice_summary.csv",
    )

    if (
        cached
        and os.path.exists(csv_path)
    ):

        print(
            "\n2D Slice Analysis (cached)"
        )

        for k, v in cached.items():

            if k != "timestamp":

                print(
                    f"  {k} = {v}"
                )

        df = pd.read_csv(
            csv_path
        )

    else:

        # --------------------------------------------------
        # LOAD PARTICLES
        # --------------------------------------------------

        print(
            "\nLoading particles..."
        )

        particles = load_particles(
            data_folder
        )

        print(
            f"Loaded particles: "
            f"{len(particles)}"
        )

        # --------------------------------------------------
        # COMPUTE
        # --------------------------------------------------

        print(
            "\nComputing slices..."
        )

        df = compute_2d_slice_summary(
            particles,
            z_levels,
            output_dir=outdir,
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

    print(
        "\nCreating plots..."
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

    print("\nSaved:")

    print(csv_path)
    print(area_png)
    print(porosity_png)


if __name__ == "__main__":
    main()
