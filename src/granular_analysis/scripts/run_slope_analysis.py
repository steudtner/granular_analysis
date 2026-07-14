"""Run segmented slope analysis using R's segmented package via subprocess."""

import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from granular_analysis.config import RESULTS_DIR

# Fallback paths for HPC where PATH gets messed up by virtualenvs
RSCRIPT_CANDIDATES = [
    "/software/rome/r2026/R/4.5.1-gfbf-2025a/bin/Rscript",
    "/software/rome/r2026/R/4.5.1/bin/Rscript",
    "Rscript"
]

def _plot_single_model(x, z, model_row, out_path, xlim=(0, 60), ylim=(0, 20), color="blue", dpi=600):
    """Create a single plot from R results."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(x, z, s=10, c="black", zorder=1)

    xticks = np.arange(xlim[0], xlim[1] + 0.01, 5)
    yticks = np.arange(ylim[0], ylim[1] + 0.01, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, which="major", linestyle=":", color="lightgray", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_pred = [float(v) for v in str(model_row["x_pred"]).split(";")]
    y_pred = [float(v) for v in str(model_row["y_pred"]).split(";")]
    ax.plot(x_pred, y_pred, color=color, linewidth=3, zorder=2)

    if model_row["n_breakpoints"] > 0:
        bps = [float(v) for v in str(model_row["breakpoints"]).split(";")]
        for bp in bps:
            ax.axvline(x=bp, color="black", linestyle="--", linewidth=1.5, zorder=1)

    slopes = [float(v) for v in str(model_row["slopes"]).split(";")]
    angles = [float(v) for v in str(model_row["angles_deg"]).split(";")]
    y_start = ylim[1]
    y_gap = (ylim[1] - ylim[0]) * 0.07
    for i, (m, a) in enumerate(zip(slopes, angles)):
        label = (rf"$m_{{{i+1}}} = {m:.2f},\; \alpha_{{{i+1}}} = {a:.1f}^\circ$")
        ax.text(xlim[1], y_start - i * y_gap, label, ha="right", va="top", fontsize=10)

    ax.text(xlim[0], np.mean(ylim), f"AIC={model_row['aic']:.1f}, BIC={model_row['bic']:.1f}",
            ha="left", va="center", color="red", fontsize=9)

    ax.set_xlabel(r"$x / D_{Eq}$", fontsize=14)
    ax.set_ylabel(r"$z / D_{Eq}$", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main(csv_path=None, xlim=None, ylim=None, force=False):
    """Run the R slope analysis and generate Python plots."""
    out_dir = os.path.join(RESULTS_DIR, "slope_analysis")
    os.makedirs(out_dir, exist_ok=True)

    if csv_path is None:
        csv_path = os.path.join(RESULTS_DIR, "alphashape_last_timestep.csv")
    if xlim is None:
        xlim = (0.0, 60.0)
    if ylim is None:
        ylim = (0.0, 20.0)

    print(f"Slope Analysis (R): Input CSV = {csv_path}")
    print(f"Slope Analysis (R): Output dir = {out_dir}")

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        return

    # --- Find Rscript ---
    rscript_path = None
    for candidate in RSCRIPT_CANDIDATES:
        try:
            test = subprocess.run([candidate, "--version"], capture_output=True)
            if test.returncode == 0:
                rscript_path = candidate
                break
        except FileNotFoundError:
            continue
            
    if not rscript_path:
        print("ERROR: Could not find Rscript. Please load R module first: module load R/4.5.1", file=sys.stderr)
        return

    # --- Run R script via subprocess ---
    r_script = os.path.join(os.path.dirname(__file__), "slope_analysis.R")
    
    print(f"Running R script with: {rscript_path}")
    try:
        result = subprocess.run(
            [rscript_path, r_script, csv_path, out_dir],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        if result.stderr:
            print("R warnings:", result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"ERROR running R script: {e}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return

    # --- Read R results ---
    r_results_path = os.path.join(out_dir, "slope_results_R.csv")
    if not os.path.exists(r_results_path):
        print(f"ERROR: R results not found: {r_results_path}")
        return

    r_df = pd.read_csv(r_results_path)
    print(f"\nLoaded R results: {len(r_df)} models")

    print("\n" + "=" * 55)
    print("SEGMENTED REGRESSION RESULTS (R)")
    print("=" * 55)
    print(f"{'Model':<15} {'AIC':>10} {'BIC':>10} {'Slopes':>20}")
    print("-" * 55)
    for _, row in r_df.iterrows():
        slopes_str = str(row["slopes"]).replace(";", ", ")
        print(f"{str(row['model']):<15} {row['aic']:>10.1f} {row['bic']:>10.1f}   {slopes_str}")

    best_aic = r_df.loc[r_df["aic"].idxmin()]
    best_bic = r_df.loc[r_df["bic"].idxmin()]
    print(f"\nBest model (AIC): {best_aic['model']}")
    print(f"Best model (BIC): {best_bic['model']}")

    # --- Load data for scatter plot ---
    data_df = pd.read_csv(csv_path, names=["x", "z_max"], header=None)
    data_df = data_df[np.isfinite(data_df["x"]) & np.isfinite(data_df["z_max"])]

    # --- Plot individual models ---
    colors = ["blue", "orange", "purple", "brown"]
    for i, (_, row) in enumerate(r_df.iterrows()):
        out_path = os.path.join(out_dir, f"model_{i}.png")
        _plot_single_model(data_df["x"].values, data_df["z_max"].values, row, out_path, xlim=xlim, ylim=ylim, color=colors[i])
        print(f"  Saved: {out_path}")

    summary_path = os.path.join(out_dir, "slope_summary.csv")
    r_df_to_save = r_df.drop(columns=["x_pred", "y_pred"])
    r_df_to_save.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmented slope analysis using R.")
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args()
    main(csv_path=args.csv_path)
