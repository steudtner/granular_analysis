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


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_particles():
    ellipsoids = load_last_frame(FOLDER)
    return filter_particles(ellipsoids, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES)


# --------------------------------------------------
# 2D DATA
# --------------------------------------------------

def compute_2d_slice_summary(particles):
    records = []

    for z in Z_LEVELS:

        result = compute_slice_porosity(
            particles,
            z,
            ALPHA_2D,
            LZ,
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


def load_or_compute_2d(particles):

    csv_path = os.path.join(
        RESULTS_DIR,
        "2d",
        "porosity.csv",
    )

    if os.path.exists(csv_path):
        print("Loading existing 2D results...")
        return pd.read_csv(csv_path)

    print("Recomputing 2D slice results...")
    return compute_2d_slice_summary(particles)


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def _plot_profile(df, columns, labels, markers, title, ylabel, save_path):
    """Generic depth-profile plotter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label, marker in zip(columns, labels, markers):
        if col in df.columns:
            ax.plot(df["z"], df[col], marker, label=label)
    ax.set_xlabel("z")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    return fig


def plot_area_profile(df, save_path):
    return _plot_profile(
        df,
        columns=["A_ell", "A_alpha"],
        labels=["A_ell", "A_alpha"],
        markers=["o-", "s-"],
        title="Area Profile",
        ylabel="Area",
        save_path=save_path,
    )


def plot_porosity_profile(df, save_path):
    return _plot_profile(
        df,
        columns=["porosity"],
        labels=["2D porosity"],
        markers=["o-"],
        title="Porosity Profile",
        ylabel="Porosity",
        save_path=save_path,
    )


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

def print_statistics(df):

    print("\n=== AREA STATISTICS ===")

    if "A_alpha" in df.columns:
        p = df["A_alpha"].dropna()
        if len(p):
            print(f"A_alpha: mean={p.mean():.4f}, std={p.std():.4f}")

    print("\n=== POROSITY STATISTICS ===")

    if "porosity" in df.columns:
        p = df["porosity"].dropna()

        if len(p):
            print(
                f"2D: mean={p.mean():.4f}, "
                f"std={p.std():.4f}"
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("2D Slice Analysis")
    print("=" * 60)

    particles = load_particles()
    print(f"Loaded particles: {len(particles)}")

    df = load_or_compute_2d(particles)

    outdir = os.path.join(
        RESULTS_DIR,
        "slices",
    )

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    csv_path = os.path.join(
        outdir,
        "slice_summary.csv",
    )

    area_png = os.path.join(
        outdir,
        "area_profile.png",
    )

    porosity_png = os.path.join(
        outdir,
        "porosity_profile.png",
    )

    df.to_csv(
        csv_path,
        index=False,
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
