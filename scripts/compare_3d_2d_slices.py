import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trimesh

from granular_analysis.config import RESULTS_DIR, Z_LEVELS, LZ, ALPHA_2D, FOLDER
from granular_analysis.io.loader import load_last_frame, filter_particles
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.utils.mesh import (
    extract_3d_slice_areas,
    section_area_from_mesh,
)


# -------------------------
# IO helpers
# -------------------------

def load_mesh():
    mesh_path = os.path.join(RESULTS_DIR, "3d", "alpha.stl")

    if not os.path.exists(mesh_path):
        print(f"[ERROR] Mesh not found: {mesh_path}")
        return None

    return trimesh.load(mesh_path)


def load_particles():
    particles = load_last_frame(FOLDER)
    return filter_particles(particles)


# -------------------------
# 2D analysis
# -------------------------

def compute_2d_slice_summary(particles, z_levels, alpha):
    records = []

    for z in z_levels:
        res = compute_slice_porosity(particles, z, alpha, LZ)

        if res is None:
            records.append({
                "z": z,
                "A_ell": 0.0,
                "A_alpha": 0.0,
                "porosity": np.nan,
                "n": len(particles),
            })
        else:
            records.append({
                "z": z,
                "A_ell": res["A_ell"],
                "A_alpha": res["A_alpha"],
                "porosity": res["porosity"],
                "n": res["n"],
            })

    return pd.DataFrame(records)


def load_2d_csv():
    path = os.path.join(RESULTS_DIR, "2d", "porosity.csv")

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


# -------------------------
# merging
# -------------------------

def merge(z_levels, areas_3d, df_2d):
    df = pd.DataFrame({
        "z": z_levels,
        "A_3d": areas_3d,
    })

    if df_2d is not None:
        df = df.merge(df_2d, on="z", how="left")

    if "A_alpha" in df.columns:
        df["porosity_3d"] = np.where(
            df["A_3d"] > 0,
            (df["A_3d"] - df["A_alpha"]) / df["A_3d"],
            np.nan,
        )
    else:
        df["porosity_3d"] = np.nan

    return df


# -------------------------
# plotting
# -------------------------

def plot(df, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(df["z"], df["A_3d"], label="3D area")
    if "A_alpha" in df:
        ax[0].plot(df["z"], df["A_alpha"], "--", label="2D area")

    ax[0].set_title("Area comparison")
    ax[0].legend()
    ax[0].grid()

    if "porosity" in df:
        ax[1].plot(df["z"], df["porosity"], label="2D")
    if "porosity_3d" in df:
        ax[1].plot(df["z"], df["porosity_3d"], label="3D")

    ax[1].set_title("Porosity")
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    return fig


# -------------------------
# main
# -------------------------

def main():
    print("=" * 60)
    print("3D / 2D Slice Comparison")
    print("=" * 60)

    mesh = load_mesh()
    if mesh is None:
        return

    particles = load_particles()
    print(f"Particles: {len(particles)}")

    areas_3d = extract_3d_slice_areas(mesh, Z_LEVELS)

    df_2d = load_2d_csv()
    if df_2d is None:
        print("Recomputing 2D slices...")
        df_2d = compute_2d_slice_summary(particles, Z_LEVELS, ALPHA_2D)

    df = merge(Z_LEVELS, areas_3d, df_2d)

    out_dir = os.path.join(RESULTS_DIR, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)

    plot(df, save_path=os.path.join(out_dir, "comparison.png"))

    print("\nSummary:")
    print(df[["A_3d"]].describe())


if __name__ == "__main__":
    main()