import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from granular_analysis.config import RESULTS_DIR, OUTDIR_3D


def load_data():
    """
    Lädt 3D/2D Vergleichsdaten, falls vorhanden.
    Erwartet CSV aus run_3d/compare step.
    """
    csv_path = os.path.join(RESULTS_DIR, "3d_visualization", "3d_2d_slice_comparison.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing CSV: {csv_path}\n"
            "Run run_all.py or comparison script first."
        )

    return pd.read_csv(csv_path)


def plot_porosity(df):
    z = df["z"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # 2D porosity
    if "porosity" in df.columns:
        ax.plot(z, df["porosity"], "o-", label="2D porosity")

    # 3D porosity
    #if "porosity_3d" in df.columns:
        #ax.plot(z, df["porosity_3d"], "s-", label="3D porosity")

    #ax.set_xlabel("Z depth")
    #ax.set_ylabel("Porosity")
    #ax.set_title("Porosity profile over depth")
    #ax.grid(True, alpha=0.3)
    #ax.legend()

    return fig


def print_statistics(df):
    print("\n=== POROSITY STATISTICS ===")

    if "porosity" in df.columns:
        p2d = df["porosity"].dropna()
        print("2D porosity:")
        print(f"  mean = {p2d.mean():.4f}")
        print(f"  std  = {p2d.std():.4f}")

    if "porosity_3d" in df.columns:
        p3d = df["porosity_3d"].dropna()
        print("3D porosity:")
        print(f"  mean = {p3d.mean():.4f}")
        print(f"  std  = {p3d.std():.4f}")


def main():
    df = load_data()

    fig = plot_porosity(df)

    print_statistics(df)

    outpath = os.path.join(RESULTS_DIR, "porosity_profile.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)

    print(f"\nSaved plot: {outpath}")


if __name__ == "__main__":
    main()