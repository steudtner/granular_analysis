import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh

from granular_analysis.config import (
    RESULTS_DIR,
    Z_LEVELS,
    LZ,
    ALPHA_2D,
    FOLDER,
)

from granular_analysis.io.loader import (
    load_last_frame,
    filter_particles,
)

from granular_analysis.analysis.porosity2d import (
    compute_slice_porosity,
)

from granular_analysis.utils.mesh import (
    extract_3d_slice_areas,
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_mesh():
    mesh_path = os.path.join(RESULTS_DIR, "3d", "alpha.stl")

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"3D mesh not found:\n{mesh_path}"
        )

    return trimesh.load(mesh_path)


def load_particles():
    particles = load_last_frame(FOLDER)
    return filter_particles(particles)


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
# MERGE
# --------------------------------------------------

def build_dataframe(df_2d, areas_3d):

    df = pd.DataFrame(
        {
            "z": Z_LEVELS,
            "A_3d": areas_3d,
        }
    )

    df = df.merge(
        df_2d,
        on="z",
        how="left",
    )

    if "A_alpha" in df.columns:

        df["porosity_3d"] = np.where(
            df["A_3d"] > 0,
            (df["A_3d"] - df["A_alpha"]) / df["A_3d"],
            np.nan,
        )

    return df


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def plot_area_profile(df, save_path):

    fig, ax = plt.subplots(figsize=(8, 5))

    if "A_ell" in df.columns:
        ax.plot(
            df["z"],
            df["A_ell"],
            "o-",
            label="A_ell",
        )

    if "A_alpha" in df.columns:
        ax.plot(
            df["z"],
            df["A_alpha"],
            "s-",
            label="A_alpha",
        )

    if "A_3d" in df.columns:
        ax.plot(
            df["z"],
            df["A_3d"],
            "^-",
            label="A_3D",
        )

    ax.set_xlabel("z")
    ax.set_ylabel("Area")
    ax.set_title("Area Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)

    return fig


def plot_porosity_profile(df, save_path):

    fig, ax = plt.subplots(figsize=(8, 5))

    if "porosity" in df.columns:
        ax.plot(
            df["z"],
            df["porosity"],
            "o-",
            label="2D porosity",
        )

    if "porosity_3d" in df.columns:
        ax.plot(
            df["z"],
            df["porosity_3d"],
            "s-",
            label="3D porosity",
        )

    ax.set_xlabel("z")
    ax.set_ylabel("Porosity")
    ax.set_title("Porosity Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)

    return fig


# --------------------------------------------------
# STATISTICS
# --------------------------------------------------

def print_statistics(df):

    print("\n=== AREA STATISTICS ===")

    if "A_3d" in df.columns:
        print(df["A_3d"].describe())

    print("\n=== POROSITY STATISTICS ===")

    if "porosity" in df.columns:
        p = df["porosity"].dropna()

        if len(p):
            print(
                f"2D: mean={p.mean():.4f}, "
                f"std={p.std():.4f}"
            )

    if "porosity_3d" in df.columns:
        p = df["porosity_3d"].dropna()

        if len(p):
            print(
                f"3D: mean={p.mean():.4f}, "
                f"std={p.std():.4f}"
            )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("2D / 3D Slice Comparison")
    print("=" * 60)

    mesh = load_mesh()

    particles = load_particles()

    print(f"Loaded particles: {len(particles)}")

    areas_3d = extract_3d_slice_areas(
        mesh,
        Z_LEVELS,
    )

    df_2d = load_or_compute_2d(
        particles,
    )

    df = build_dataframe(
        df_2d,
        areas_3d,
    )

    outdir = os.path.join(
        RESULTS_DIR,
        "comparison",
    )

    os.makedirs(
        outdir,
        exist_ok=True,
    )

    csv_path = os.path.join(
        outdir,
        "comparison.csv",
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