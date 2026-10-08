"""Plot AIC and BIC model performance over an alpha-parameter sweep."""

from __future__ import annotations

from granular_analysis.plotting import plotstyle

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from granular_analysis.plotting.helpers import viridis_palette


def plot_model_selection(input_root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    """Collect slope-model scores from ``alpha_*`` directories and plot them."""
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for alpha_dir in sorted(input_root.glob("alpha_*"), key=lambda path: float(path.name[6:])):
        csv_path = alpha_dir / "slope_analysis" / "slope_results_R.csv"
        if not csv_path.is_file():
            continue
        table = pd.read_csv(csv_path)
        table["alpha"] = float(alpha_dir.name[6:])
        tables.append(table[["alpha", "model", "n_breakpoints", "aic", "bic"]])
    if not tables:
        raise FileNotFoundError(f"No slope_results_R.csv files found under {input_root}")

    data = pd.concat(tables, ignore_index=True).sort_values(["alpha", "model"])
    data.to_csv(output_dir / "slope_model_selection.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.figwidth_full, 4.5), sharex=True, constrained_layout=True)
    model_groups = list(data.groupby("model", sort=False))
    colours = viridis_palette()
    for metric, axis in zip(("aic", "bic"), axes):
        for colour, (model, group) in zip(colours, model_groups):
            axis.plot(
                group["alpha"], group[metric], "o-", linewidth=1.2,
                color=colour, label=model,
            )
        axis.set(
            xlabel=r"Alpha-Parameter, $\alpha$",
            ylabel=metric.upper(),
            title=f"{metric.upper()} der Steigungsmodelle",
        )
        axis.grid(True, alpha=0.3)
    axes[1].legend(title="Modell", loc="best")
    fig.suptitle("Modellauswahl der Steigungsanalyse über Alpha")
    plotstyle.export_figure(fig, output_dir / "slope_model_selection.png", dpi=200)
    plt.close(fig)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", help="Directory containing alpha_* run directories")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    data = plot_model_selection(args.input_root, args.out)
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
