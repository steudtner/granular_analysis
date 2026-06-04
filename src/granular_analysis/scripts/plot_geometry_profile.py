import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from granular_analysis.config import RESULTS_DIR


def load_data():
    csv_path = os.path.join(
        RESULTS_DIR,
        "3d_visualization",
        "3d_2d_slice_comparison.csv",
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")

    return pd.read_csv(csv_path)


def plot_areas(df):
    z = df["z"]

    fig, ax = plt.subplots(figsize=(8, 5))

    if "A_ell" in df.columns:
        ax.plot(z, df["A_ell"], "o-", label="A_ell (ellipsoid area)")

    if "A_alpha" in df.columns:
        ax.plot(z, df["A_alpha"], "s-", label="A_alpha (alpha shape area)")

    if "A_3d" in df.columns:
        ax.plot(z, df["A_3d"], "^-", label="A_3D slice area")

    ax.set_xlabel("Z depth")
    ax.set_ylabel("Area")
    ax.set_title("Area components over depth (2D slices + 3D projection)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def plot_volumes(df):
    """
    Volumes are constant over z in your current setup,
    but we still plot them for completeness.
    """

    z = df["z"]

    fig, ax = plt.subplots(figsize=(8, 5))

    if "V_ell" in df.columns:
        ax.plot(z, df["V_ell"], label="V_ell (constant)", linewidth=2)

    if "V_alpha" in df.columns:
        ax.plot(z, df["V_alpha"], label="V_alpha (constant)", linewidth=2)

    ax.set_xlabel("Z depth (dummy axis)")
    ax.set_ylabel("Volume")
    ax.set_title("3D volume metrics (replicated over z for comparison)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def main():
    df = load_data()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig1 = plot_areas(df)
    fig2 = plot_volumes(df)

    out1 = os.path.join(RESULTS_DIR, "area_profile.png")
    out2 = os.path.join(RESULTS_DIR, "volume_profile.png")

    fig1.tight_layout()
    fig1.savefig(out1, dpi=200)

    fig2.tight_layout()
    fig2.savefig(out2, dpi=200)

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()