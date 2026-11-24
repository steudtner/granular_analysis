#!/usr/bin/env python3
"""Compare R/segmented and Python/SciPy fits on one upper-edge profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from granular_analysis.analysis.slope import fit_slope_models
from granular_analysis.plotting.slope_plots import plot_all_models_from_csv
from granular_analysis.scripts.run_slope_analysis import (
    _find_input_csv,
    _load_and_normalise_slice,
    rscript_command,
)


def _serialise(values, digits=6):
    return ";".join(f"{float(value):.{digits}f}" for value in values)


def _python_results(data: pd.DataFrame, max_breakpoints: int) -> pd.DataFrame:
    x = data["x"].to_numpy(dtype=float)
    z = data["z_max"].to_numpy(dtype=float)
    grid = np.linspace(float(x.min()), float(x.max()), 500)
    records = []
    for model in fit_slope_models(x, z, max_breakpoints=max_breakpoints):
        records.append({
            "model": model.name,
            "n_breakpoints": model.n_breakpoints,
            "breakpoints": _serialise(model.breakpoints),
            "slopes": _serialise(model.slopes),
            "angles_deg": _serialise(model.angles_deg, digits=4),
            "aic": model.aic,
            "bic": model.bic,
            "rss": model.rss,
            "intercept": model.intercept,
            "x_pred": _serialise(grid),
            "y_pred": _serialise(model.predict(grid)),
        })
    return pd.DataFrame.from_records(records)


def _run_r(csv_path: Path, output_dir: Path, rscript: str | None) -> Path:
    script = Path(__file__).with_name("slope_analysis.R")
    command = rscript_command(str(script), str(csv_path), str(output_dir), executable=rscript)
    subprocess.run(command, check=True)
    result = output_dir / "slope_results_R.csv"
    if not result.is_file():
        raise FileNotFoundError(f"R slope analysis produced no {result}")
    return result


def _comparison(r_results: pd.DataFrame, python_results: pd.DataFrame):
    columns = [
        "n_breakpoints", "breakpoints", "slopes", "angles_deg", "aic", "bic"
    ]
    r = r_results[columns].rename(columns={c: f"r_{c}" for c in columns[1:]})
    py = python_results[columns].rename(
        columns={c: f"python_{c}" for c in columns[1:]}
    )
    result = r.merge(py, on="n_breakpoints", how="outer").sort_values(
        "n_breakpoints"
    )
    result["python_minus_r_aic"] = result["python_aic"] - result["r_aic"]
    result["python_minus_r_bic"] = result["python_bic"] - result["r_bic"]
    return result


def compare(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    rscript: str | None = None,
    max_breakpoints: int = 3,
) -> Path:
    csv_path = Path(csv_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_and_normalise_slice(csv_path)

    python_path = output_dir / "slope_results_python.csv"
    python_results = _python_results(data, max_breakpoints=max_breakpoints)
    python_results.to_csv(python_path, index=False)

    r_path = _run_r(csv_path, output_dir, rscript)
    r_results = pd.read_csv(r_path)
    comparison = _comparison(r_results, python_results)
    comparison_path = output_dir / "slope_method_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    for label, result_path in (("r", r_path), ("python", python_path)):
        plot_dir = output_dir / label
        plot_dir.mkdir(exist_ok=True)
        plot_all_models_from_csv(
            str(result_path),
            str(plot_dir),
            data["x"].to_numpy(),
            data["z_max"].to_numpy(),
        )

    print("\nR/Python comparison (negative delta favours Python):")
    print(comparison[[
        "n_breakpoints", "python_minus_r_aic", "python_minus_r_bic"
    ]].to_string(index=False))
    print(f"\nFull comparison: {comparison_path}")
    return comparison_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Upper-edge CSV; inferred from --results if omitted.")
    parser.add_argument(
        "--results",
        default="results_granular_analysis",
        help="Analysis results directory (default: ./results_granular_analysis).",
    )
    parser.add_argument("--out", help="Comparison output directory.")
    parser.add_argument("--rscript", help="Explicit Rscript executable.")
    parser.add_argument("--max-breakpoints", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    results_dir = Path(args.results).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else _find_input_csv(results_dir)
    output_dir = Path(args.out).expanduser().resolve() if args.out else results_dir / "slope_comparison"
    compare(
        csv_path,
        output_dir,
        rscript=args.rscript,
        max_breakpoints=args.max_breakpoints,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
