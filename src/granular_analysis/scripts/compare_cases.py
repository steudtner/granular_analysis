#!/usr/bin/env python3
"""CLI to compare multiple simulation cases for selected metrics.

Usage example:
  python -m granular_analysis.scripts.compare_cases --metric porosity_z caseA:/path/to/A caseB:/path/to/B
"""
import argparse
import os
from typing import Dict, Optional
import glob

from granular_analysis.analysis.compare import compare_porosity_z, compare_kinetic_energy


def _find_bin_directory(root: str) -> Optional[str]:
    """Try to find a directory containing .bin files starting from *root*.

    Checks the given root, then common subpaths used in this project and
    returns the first directory that contains any ``*.bin`` files.
    """
    cand = [root,
            os.path.join(root, "results", "ellipsoid", "ellipsoid_trn"),
            os.path.join(root, "results", "ellipsoid"),
            os.path.join(root, "results"),
            os.path.join(root, "results", "ellipsoid", "ellipsoid_trn"),
    ]
    for c in cand:
        if not os.path.isdir(c):
            continue
        files = glob.glob(os.path.join(c, "*.bin"))
        if files:
            return c
    return None


def parse_case_arg(s: str) -> Dict[str, str]:
    # expected format name:path
    if ":" not in s:
        raise argparse.ArgumentTypeError("Case must be NAME:PATH")
    name, path = s.split(":", 1)
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Not a directory: {path}")

    # If the given directory doesn't contain .bin files, try common subpaths
    files = glob.glob(os.path.join(path, "*.bin"))
    if not files:
        found = _find_bin_directory(path)
        if found:
            path = found
        else:
            raise argparse.ArgumentTypeError(f"No .bin files found under: {path}")

    return {name: path}


def build_cases(list_args):
    out = {}
    for s in list_args:
        out.update(parse_case_arg(s))
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare cases for selected metrics")
    parser.add_argument("--metric", choices=["porosity_z", "kinetic_energy"], required=True)
    parser.add_argument("--out", default=None, help="Output folder root (optional)")
    parser.add_argument("--slice-z", type=float, default=None, help="Z value for single-slice alpha-shape comparison (default: LZ/2)")
    parser.add_argument("--include-alpha", action="store_true", help="Also compute and plot alpha-shape overlay at --slice-z")
    parser.add_argument("--densify", type=float, default=None, help="If set, densify alpha-shape polygon segments to max length")
    parser.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is installed")
    parser.add_argument("--frame", default=None, help="Frame index (int) or bin filename to use instead of the last frame")
    parser.add_argument("--time", type=float, default=None, help="Physical time to select the frame by (selects nearest unless --time-mode set)")
    parser.add_argument("--time-mode", choices=["nearest", "before", "after"], default="nearest", help="Behavior when selecting frame by time")
    parser.add_argument("--latest", action="store_true", help="Use the latest time that is available in all provided cases (common latest time)")
    parser.add_argument("cases", nargs="+", help="Cases as NAME:PATH")
    args = parser.parse_args()

    cases = build_cases(args.cases)

    # If --latest is set and no explicit --time provided, compute the common latest time
    if args.latest and args.time is None:
        from granular_analysis.io.loader import get_all_bin_paths, _read_bin_time

        times = []
        for name, path in cases.items():
            try:
                files = get_all_bin_paths(path)
            except Exception as e:
                parser.error(f"No .bin files for case {name}: {e}")
            last = files[-1]
            try:
                t = _read_bin_time(last)
            except Exception as e:
                parser.error(f"Failed to read time from {last}: {e}")
            times.append(t)

        # choose the minimum of last times so the chosen time exists (or can be matched) in all cases
        common_latest = min(times)
        print(f"Using common latest time across cases: {common_latest}")
        args.time = common_latest

    if args.metric == "porosity_z":
        frame_arg = None
        if args.frame is not None:
            try:
                frame_arg = int(args.frame)
            except Exception:
                frame_arg = args.frame
        compare_porosity_z(cases, out_dir=args.out, slice_z=args.slice_z, include_alpha=args.include_alpha, densify_max_segment_length=args.densify, show_progress=args.progress, frame=frame_arg, time=args.time, time_mode=args.time_mode)
    elif args.metric == "kinetic_energy":
        frame_arg = None
        if args.frame is not None:
            try:
                frame_arg = int(args.frame)
            except Exception:
                frame_arg = args.frame
        compare_kinetic_energy(cases, out_dir=args.out, show_progress=args.progress, frame=frame_arg)


if __name__ == "__main__":
    main()
