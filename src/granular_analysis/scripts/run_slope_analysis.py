#!/usr/bin/env python3
"""Run segmented-regression slope analysis using the packaged R script."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from granular_analysis.config import RESULTS_DIR
from granular_analysis.plotting.slope_plots import plot_all_models_from_csv
from granular_analysis.utils.state_manager import build_fingerprint, load_state, save_state


ROMEO_R_COMMAND = (
    'module load release/2026 GCC/14.2.0 R/4.5.1 >/dev/null '
    '&& exec Rscript "$@"'
)


def rscript_command(*arguments: str, executable: str = None) -> list[str]:
    """Build a portable Rscript command, with a Romeo module fallback."""
    selected = executable or os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if selected:
        return [selected, *map(str, arguments)]

    # A login shell exposes Environment Modules/Lmod on Romeo. Loading R in
    # the child supplies Rscript and its libraries without altering Python.
    if os.environ.get("LMOD_CMD") or Path("/software/rome").is_dir():
        bash = shutil.which("bash") or "/bin/bash"
        return [
            bash,
            "-lc",
            ROMEO_R_COMMAND,
            "granular-rscript",
            *map(str, arguments),
        ]

    raise FileNotFoundError(
        "Rscript was not found. Install/load R, set RSCRIPT=/path/to/Rscript, "
        "or load the cluster R module."
    )


def _candidate_slice_csvs(results_dir: Path):
    for directory in (results_dir / "2d", results_dir):
        yield directory / "alphashape_last_timestep.csv"
        yield from sorted(directory.glob("alphashape_slice_*.csv"))


def _find_input_csv(results_dir: Path) -> Path:
    for candidate in _candidate_slice_csvs(results_dir):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No upper-edge CSV found under {results_dir}. Run the 2d pipeline first."
    )


def _load_and_normalise_slice(csv_path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(csv_path)
        if {"x", "z_max"}.issubset(frame.columns):
            return frame[["x", "z_max"]]
    except (OSError, pd.errors.ParserError):
        pass

    frame = pd.read_csv(csv_path, header=None, names=["x", "z_max"])
    if frame.shape[1] != 2 or frame.empty:
        raise ValueError(f"Expected two non-empty columns in {csv_path}")
    return frame


def main(force: bool = False, out: str = None, alpha: float = None,
         xlim=(0, 60), ylim=(0, 20)):
    results_dir = Path(out or RESULTS_DIR).expanduser().resolve()
    csv_path = _find_input_csv(results_dir)
    data = _load_and_normalise_slice(csv_path)
    analysis_dir = results_dir / "slope_analysis"
    results_path = analysis_dir / "slope_results_R.csv"
    r_script = Path(__file__).with_name("slope_analysis.R")
    csv_stat = csv_path.stat()
    script_stat = r_script.stat() if r_script.is_file() else None
    cache_fingerprint = build_fingerprint(
        csv=str(csv_path.resolve()),
        csv_size=csv_stat.st_size,
        csv_mtime_ns=csv_stat.st_mtime_ns,
        r_script_mtime_ns=script_stat.st_mtime_ns if script_stat else None,
    )

    cached = None if force else load_state(
        "slope", str(results_dir), fingerprint=cache_fingerprint
    )
    if cached and results_path.is_file():
        plot_all_models_from_csv(
            str(results_path),
            str(analysis_dir),
            data["x"].to_numpy(),
            data["z_max"].to_numpy(),
            alpha=alpha,
            xlim=xlim,
            ylim=ylim,
        )
        return

    if not r_script.is_file():
        raise FileNotFoundError(f"Packaged R script not found: {r_script}")


    analysis_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        rscript_command(str(r_script), str(csv_path), str(analysis_dir)),
        check=True,
    )
    if not results_path.is_file():
        raise FileNotFoundError(f"R analysis did not create {results_path}")

    plot_all_models_from_csv(
        str(results_path),
        str(analysis_dir),
        data["x"].to_numpy(),
        data["z_max"].to_numpy(),
        alpha=alpha,
        xlim=xlim,
        ylim=ylim,
    )
    save_state(
        "slope",
        {"status": "completed"},
        str(results_dir),
        fingerprint=cache_fingerprint,
    )


if __name__ == "__main__":
    main()
