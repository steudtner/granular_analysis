"""Plotting routines for segmented slope analysis."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm  # for colour cycling (kept from previous fix)


# ----------------------------------------------------------------------
# Helper: draw a single model
# ----------------------------------------------------------------------
def _plot_single_model(
    x, z, model, out_path, xlim=(0, 60), ylim=(0, 20),
    color="blue", dpi=600
):
    """
    Create a publication‑quality plot for one model.

    * Scatter of the raw data (black points)
    * Piece‑wise (or simple linear) fit in *color*
    * Breakpoint vertical lines (if any)
    * Upper‑right annotation with **slope** and **angle** for each segment
    * Left‑hand annotation with **AIC** and **BIC**
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 3))

    # --------------------------------------------------------------
    # 1️⃣  Raw data points
    # --------------------------------------------------------------
    ax.scatter(x, z, s=10, c="black", zorder=1)

    # --------------------------------------------------------------
    # 2️⃣  Grid & axis style
    # --------------------------------------------------------------
    xticks = np.arange(xlim[0], xlim[1] + 0.01, 5)
    yticks = np.arange(ylim[0], ylim[1] + 0.01, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, which="major", linestyle=":", color="lightgray",
            linewidth=0.5, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------------
    # 3️⃣  Model line (linear or segmented)
    # --------------------------------------------------------------
    n_bp = model["n_breakpoints"]
    if n_bp == 0:                                 # simple linear
        x_line = np.linspace(xlim[0], xlim[1], 100)
        y_line = model["intercept"] + model["slopes"][0] * x_line
        ax.plot(x_line, y_line, color=color, linewidth=3, zorder=2)
    else:                                          # segmented regression
        x_line = np.linspace(xlim[0], xlim[1], 500)
        y_line = model["intercept"] + model["slopes"][0] * x_line

        for i, bp in enumerate(model["breakpoints"]):
            # change in slope after each breakpoint
            if i < len(model["slopes"]) - 1:
                delta = model["slopes"][i + 1] - model["slopes"][i]
                y_line = y_line + delta * np.maximum(0, x_line - bp)

        ax.plot(x_line, y_line, color=color, linewidth=3, zorder=2)

        # vertical lines marking the breakpoints
        for bp in model["breakpoints"]:
            ax.axvline(x=bp, color="black", linestyle="--",
                       linewidth=1.5, zorder=1)

    # --------------------------------------------------------------
    # 4️⃣  Slope / angle annotation (upper right)
    # --------------------------------------------------------------
    #slopes = model["slopes"]
    #angles = model["angles_deg"]
    #y_start = ylim[1]
    #y_gap = (ylim[1] - ylim[0]) * 0.07

    # for i, (m, a) in enumerate(zip(slopes, angles)):
    #    label = (
    #        rf"$m_{{{i+1}}} = {m:.2f},\; "
    #        rf"\alpha_{{{i+1}}} = {a:.1f}^\circ$"
    #    )
    #    ax.text(
    #        xlim[1], y_start - i * y_gap,
    #        label,
    #        ha="right", va="top", fontsize=10,
    #    )

    # --------------------------------------------------------------
    # 5️⃣  AIC / BIC annotation (left side)
    # --------------------------------------------------------------
    #ax.text(
     #   xlim[0], np.mean(ylim),
     #   f"AIC={model['aic']:.1f}, BIC={model['bic']:.1f}",
     #   ha="left", va="center", color="red", fontsize=9,
    #)

    # --------------------------------------------------------------
    # 6️⃣  Axis labels
    # --------------------------------------------------------------
    ax.set_xlabel(r"$x / D_{Eq}$", fontsize=14)
    ax.set_ylabel(r"$z / D_{Eq}$", fontsize=14)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


# ----------------------------------------------------------------------
# Helper: plot *all* models and also dump a richer console table
# ----------------------------------------------------------------------
def plot_all_models_from_csv(csv_path, out_dir, x_data, z_data,
                             xlim=(0, 60), ylim=(0, 20)):
    """
    Read the R‑generated CSV, plot each model, and print a full table that
    includes:

    * model name
    * number of breakpoints
    * break‑point x‑values
    * slopes
    * angles (degrees)
    * AIC / BIC

    The colour palette is automatically enlarged so we never hit an
    “index out of range” error.
    """
    df = pd.read_csv(csv_path)

    # --------------------------------------------------------------
    # Colour palette – ensure we have at least as many colours as models
    # --------------------------------------------------------------
    n_models = len(df)
    base = cm.tab10(np.linspace(0, 1, 10))          # 10 distinct colours
    colours = np.resize(base, (n_models, 4))       # repeat if >10 models

    # --------------------------------------------------------------
    # Header for the console table
    # --------------------------------------------------------------
    header = (
        f"{'Model':<15} {'#BP':>3}   {'Breakpoints':<30} "
        f"{'Slopes':<30} {'Angles[°]':<30} {'AIC':>7} {'BIC':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    # --------------------------------------------------------------
    # Loop over each model row
    # --------------------------------------------------------------
    for idx, row in df.iterrows():
        model_info = {
            'model': row['model'],
            'n_breakpoints': int(row['n_breakpoints']),
            'breakpoints': ([float(x) for x in row['breakpoints'].split(';')]
                            if pd.notna(row['breakpoints']) and row['breakpoints'] != ''
                            else []),
            'slopes': [float(x) for x in row['slopes'].split(';')],
            'angles_deg': [float(x) for x in row['angles_deg'].split(';')],
            'aic': float(row['aic']),
            'bic': float(row['bic']),
            'intercept': float(row['intercept'])
        }

        # ----------------------------------------------------------
        # 1️⃣  Print the row in a compact, human‑readable form
        # ----------------------------------------------------------
        bp_str   = "; ".join(f"{bp:.4f}" for bp in model_info['breakpoints'])
        sl_str   = "; ".join(f"{s:.4f}"  for s in model_info['slopes'])
        ang_str  = "; ".join(f"{a:.2f}"  for a in model_info['angles_deg'])
        print(
            f"{model_info['model']:<15} "
            f"{model_info['n_breakpoints']:>3}   "
            f"{bp_str:<30} "
            f"{sl_str:<30} "
            f"{ang_str:<30} "
            f"{model_info['aic']:>7.1f} {model_info['bic']:>7.1f}"
        )

        # ----------------------------------------------------------
        # 2️⃣  Plot the figure
        # ----------------------------------------------------------
        colour = colours[idx]                     # safe indexing
        clean_name = model_info['model'].replace(' ', '_').replace('/', '_')
        out_path = f"{out_dir}/{clean_name}_plot.png"
        try:
            _plot_single_model(
                x_data, z_data, model_info,
                out_path, xlim, ylim, color=colour
            )
            print(f"✅  Saved plot for {model_info['model']} → {out_path}")
        except Exception as exc:
            print(f"⚠️  Plot failed for {model_info['model']}: {exc}")

    # --------------------------------------------------------------
    # Footer separator
    # --------------------------------------------------------------
    print("=" * len(header))


# ----------------------------------------------------------------------
# Optional: a tiny helper if you ever want to dump the table only
# ----------------------------------------------------------------------
def print_breakpoint_summary(csv_path: str):
    """Print just the breakpoint / AIC / BIC table (no plotting)."""
    df = pd.read_csv(csv_path)

    header = (
        f"{'Model':<15} {'#BP':>3}   {'Breakpoints':<30} "
        f"{'Slopes':<30} {'Angles[°]':<30} {'AIC':>7} {'BIC':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        n_bp = int(row['n_breakpoints'])
        bp = ([float(x) for x in row['breakpoints'].split(';')]
               if pd.notna(row['breakpoints']) and row['breakpoints'] != ''
               else [])
        sl = [float(x) for x in row['slopes'].split(';')]
        ang = [float(x) for x in row['angles_deg'].split(';')]
        aic = float(row['aic'])
        bic = float(row['bic'])

        print(
            f"{row['model']:<15} {n_bp:>3}   "
            f"{'; '.join(f'{b:.4f}' for b in bp):<30} "
            f"{'; '.join(f'{s:.4f}' for s in sl):<30} "
            f"{'; '.join(f'{a:.2f}' for a in ang):<30} "
            f"{aic:>7.1f} {bic:>7.1f}"
        )
    print("=" * len(header))