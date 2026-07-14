#!/usr/bin/env python3
"""Orchestrator to run all or selected granular analysis pipelines."""
import os

# Set the data folder to be relative to the current directory
current_dir = os.getcwd()
data_folder = os.path.join(current_dir, "results", "ellipsoid", "ellipsoid_trn")
os.environ['GRANULAR_DATA_FOLDER'] = data_folder

import argparse
import sys
import traceback
from granular_analysis.config import RESULTS_DIR, FOLDER
from granular_analysis.scripts.run_2d_analysis import main as run_2d
from granular_analysis.scripts.run_3d_porosity import main as run_3d
from granular_analysis.scripts.run_slice_comparison import main as run_slices
from granular_analysis.scripts.run_energy_analysis import main as run_energy
from granular_analysis.scripts.run_slope_analysis import main as run_slope
from granular_analysis.utils.logging_utils import suppress_noise

PIPELINES = {
    "2d": run_2d,
    #"3d": run_3d,
    "slices": run_slices,
    "energy": run_energy,
    "slope": run_slope,
}

def run_pipeline(name: str, func, force: bool = False) -> bool:
    """Run a single pipeline, catch and report errors."""
    print("\n" + "=" * 60)
    print(f"Running: {name}")
    print("=" * 60)
    try:
        # Pass force argument if the function accepts it
        try:
            func(force=force)
        except TypeError:
            # If the function doesn't accept 'force', call it without
            func()
        return True
    except Exception as e:
        print(f"\nERROR in pipeline '{name}': {e}", file=sys.stderr)
        traceback.print_exc()
        return False

def main():
    suppress_noise()
    
    print("=" * 60)
    print("PATHS")
    print("=" * 60)
    print(f"DATA    = {FOLDER}")
    print(f"RESULTS = {RESULTS_DIR}")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=list(PIPELINES.keys()),
        nargs="+",
        metavar="NAME",
        help="Run only the named pipeline(s). Omit to run all.",
    )
    parser.add_argument(
        "--exclude",
        choices=list(PIPELINES.keys()),
        nargs="+",
        metavar="NAME",
        help="Skip the named pipeline(s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-computation, ignoring cached results.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining pipelines if one fails.",
    )
    args = parser.parse_args()

    # Determine which pipelines to run
    names = list(PIPELINES.keys())
    if args.only:
        names = [n for n in names if n in args.only]
    if args.exclude:
        names = [n for n in names if n not in args.exclude]
    if not names:
        print("No pipelines selected.", file=sys.stderr)
        sys.exit(1)

    # Execute
    results = {}
    for name in names:
        success = run_pipeline(name, PIPELINES[name], force=args.force)
        results[name] = success
        if not success and not args.continue_on_error:
            sys.exit(1)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:12s} {status}")
    print("=" * 60)

if __name__ == "__main__":
    main()
