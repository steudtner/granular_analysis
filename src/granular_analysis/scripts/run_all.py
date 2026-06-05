#!/usr/bin/env python3
"""Orchestrator to run all or selected granular analysis pipelines.

Available pipelines
-------------------
  2d       : Run 2D porosity and ellipse area analysis.
  3d       : Run 3D porosity analysis (includes PLY export).
  slices   : Run 2D depth-slice analysis with area/porosity profiles.
  energy   : Compute kinetic energy time series.

The script caches intermediate results (e.g. CSV files). Re-running without
--force therefore skips expensive re-computation and only regenerates plots.
"""

import argparse
import sys
import traceback

from granular_analysis.scripts.run_2d_analysis import main as run_2d
from granular_analysis.scripts.run_3d_porosity import main as run_3d
from granular_analysis.scripts.run_slice_comparison import main as run_slices
from granular_analysis.scripts.run_energy_analysis import main as run_energy
from granular_analysis.utils.logging_utils import suppress_noise


PIPELINES = {
    "2d": run_2d,
    "3d": run_3d,
    "slices": run_slices,
    "energy": run_energy,
}


def run_pipeline(name: str, func) -> bool:
    """Run a single pipeline, catch and report errors."""
    print("\n" + "=" * 60)
    print(f"Running: {name}")
    print("=" * 60)
    try:
        func()
        return True
    except Exception as e:
        print(f"\nERROR in pipeline '{name}': {e}", file=sys.stderr)
        traceback.print_exc()
        return False


def main():
    suppress_noise()

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  Run everything:
    python -m granular_analysis.scripts.run_all

  Run only the 2D analysis:
    python -m granular_analysis.scripts.run_all --only 2d

  Run only energy analysis:
    python -m granular_analysis.scripts.run_all --only energy

  Run everything except 3D:
    python -m granular_analysis.scripts.run_all --exclude 3d

  Continue on error (don't stop if one pipeline fails):
    python -m granular_analysis.scripts.run_all --continue-on-error
""",
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
        success = run_pipeline(name, PIPELINES[name])
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