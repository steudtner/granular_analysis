#!/usr/bin/env python3
"""Run the segmented‑regression (slope) analysis.

* Locate the CSV that contains the (x, z_max) data produced by the
  α‑shape step.
* Normalise its column names to `x` and `z_max` (the rest of the code
  expects exactly these names).
* Call the R script that fits the segmented models.
* Plot all models (including breakpoints) using the existing
  `slope_plots.py`.
"""
import os
import glob
import subprocess
import pandas as pd

from granular_analysis.config import RESULTS_DIR
from granular_analysis.utils.state_manager import load_state, save_state
from granular_analysis.plotting.slope_plots import plot_all_models_from_csv


# ----------------------------------------------------------------------
# Helper: read the z‑value that was used for the exact‑slice plot
# ----------------------------------------------------------------------
def _read_last_z() -> float:
    """
    Try to infer the slice‑z value from an existing CSV.
    If we can’t find it we simply return the default (2.5) – the
    caller will overwrite it if needed.
    """
    # The α‑shape stage writes either
    #   alphashape_last_timestep.csv
    # or a slice‑specific file:
    #   alphashape_slice_<z>.csv
    patterns = [
        os.path.join(RESULTS_DIR, "alphashape_last_timestep.csv"),
        os.path.join(RESULTS_DIR, "alphashape_slice_*.csv"),
    ]

    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            try:
                df = pd.read_csv(p, nrows=5)   # read a few rows only
                # Possible column names for the slice coordinate
                if "z" in df.columns:
                    return float(df["z"].iloc[0])
                if "z_center" in df.columns:
                    return float(df["z_center"].iloc[0])
                if "z_max" in df.columns:
                    return float(df["z_max"].iloc[0])
            except Exception:
                # If the file is malformed, just skip it
                continue

    # Fallback – the default that the CLI uses when nothing is found
    return 2.5


# ----------------------------------------------------------------------
# Helper: locate the CSV that contains the (x, z_max) data
# ----------------------------------------------------------------------
def _find_input_csv() -> str:
    """
    The α‑shape step can produce two different CSV names.
    We try both and return the first one that exists.
    """
    # 1️⃣  The generic “last timestep” file
    candidate1 = os.path.join(RESULTS_DIR, "alphashape_last_timestep.csv")
    if os.path.isfile(candidate1):
        return candidate1

    # 2️⃣  The slice‑specific file that matches the exact‑slice z value
    #     (e.g. alphashape_slice_1.25.csv)
    z_val = _read_last_z()
    candidate2 = os.path.join(
        RESULTS_DIR,
        f"alphashape_slice_{z_val:.2f}.csv"
    )
    if os.path.isfile(candidate2):
        return candidate2

    # If we reach this point we couldn't locate any file – raise a clear error
    raise FileNotFoundError(
        "Could not locate a CSV with slice data. "
        f"Tried: {candidate1}, {candidate2}"
    )


# ----------------------------------------------------------------------
# Helper: rename columns so the rest of the pipeline can use ``x`` and ``z_max``
# ----------------------------------------------------------------------
def _load_and_normalise_slice(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV that comes from the α‑shape step.
    The file is *header‑less* and contains exactly two columns:
        column 0 → x
        column 1 → z_max
    This function reads it accordingly and guarantees that the
    resulting DataFrame has the canonical column names ``x`` and ``z_max``.
    """
    # -------------------------------------------------------------
    # 1️⃣  Try the ordinary way first (in case a header *does* exist)
    # -------------------------------------------------------------
    try:
        df = pd.read_csv(csv_path)
        # If we got *any* column named “x” or “z” we can keep it.
        # Otherwise we fall through to the header‑less branch.
        if ("x" in df.columns) and ("z_max" in df.columns):
            return df
    except Exception:
        # If pandas raises (e.g. empty file) we simply go to the
        # header‑less branch – the fallback will still work.
        pass

    # -------------------------------------------------------------
    # 2️⃣  Header‑less file → read with `header=None` and assign names
    # -------------------------------------------------------------
    try:
        df = pd.read_csv(csv_path, header=None, names=["x", "z_max"])
        # Verify that we really got two columns
        if df.shape[1] != 2:
            raise ValueError(
                f"Expected 2 columns after forced rename, got {df.shape[1]}"
            )
        print(
            f"🔧  CSV had no header – forced column names: ['x', 'z_max'] "
            f"(rows={len(df)})"
        )
        return df
    except Exception as exc:
        # Re‑raise a clear error that will be caught by the caller
        raise RuntimeError(
            f"Failed to read/normalise CSV '{csv_path}': {exc}"
        ) from exc


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def main(force: bool = False):
    # ------------------------------------------------------------------
    # 1️⃣  Locate the slice CSV and normalise its column names
    # ------------------------------------------------------------------
    try:
        csv_path = _find_input_csv()
    except FileNotFoundError as e:
        print(f"❌  {e}")
        return

    try:
        data_df = _load_and_normalise_slice(csv_path)
    except Exception as exc:
        print(f"❌  Failed to read/normalise CSV '{csv_path}': {exc}")
        return

    print(
        f"✅  Loaded slice data from {csv_path} "
        f"({len(data_df)} rows, columns={list(data_df.columns)})"
    )

    # ------------------------------------------------------------------
    # 2️⃣  Cache handling (unchanged logic, just using the new DataFrame)
    # ------------------------------------------------------------------
    cached = None if force else load_state("slope", RESULTS_DIR)
    if cached:
        print("Slope analysis result (cached):")
        for k, v in cached.items():
            if k != "timestamp":
                print(f"  {k} = {v}")

        results_path = os.path.join(
            RESULTS_DIR, "slope_analysis", "slope_results_R.csv"
        )
        if os.path.isfile(results_path):
            plot_all_models_from_csv(
                results_path,
                os.path.join(RESULTS_DIR, "slope_analysis"),
                data_df["x"].values,
                data_df["z_max"].values,
            )
        return

    # ------------------------------------------------------------------
    # 3️⃣  Run the R script
    # ------------------------------------------------------------------
    out_dir = os.path.join(RESULTS_DIR, "slope_analysis")
    os.makedirs(out_dir, exist_ok=True)

    r_script = os.path.join(
        os.path.dirname(__file__), "slope_analysis.R"
    )
    if not os.path.isfile(r_script):
        print(f"❌  R script not found at {r_script}")
        return

    cmd = [
        "/software/rome/r2026/R/4.5.1-gfbf-2025a/bin/Rscript",
        r_script,
        csv_path,
        out_dir,
    ]

    print("\n🔧  Calling R script:")
    print("   " + " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print("\nR stdout:")
        print(proc.stdout)
        if proc.stderr:
            print("\nR warnings / stderr:")
            print(proc.stderr)
    except subprocess.CalledProcessError as e:
        print(f"\n❌  Rscript failed (rc={e.returncode})")
        print(e.stdout)
        print(e.stderr)
        return

    # ------------------------------------------------------------------
    # 4️⃣  Plot the results that the R script just created
    # ------------------------------------------------------------------
    results_path = os.path.join(out_dir, "slope_results_R.csv")
    if not os.path.isfile(results_path):
        print(f"❌  Expected results CSV not found: {results_path}")
        return

    print(f"\n📊  Plotting {len(pd.read_csv(results_path))} models …")
    plot_all_models_from_csv(
        results_path,
        out_dir,
        data_df["x"].values,
        data_df["z_max"].values,
    )

    # ------------------------------------------------------------------
    # 5️⃣  Save a tiny state file so the next run can skip the heavy work
    # ------------------------------------------------------------------
    save_state("slope", {"status": "completed"}, RESULTS_DIR)


if __name__ == "__main__":
    main()