#!/usr/bin/env python3
"""
Plot a xy slice of particles from the configured ellipsoid dataset.

This script uses the package loader and plotting pipeline, while preserving
existing config-based input/output locations.
"""

import argparse
from pathlib import Path

import numpy as np

from granular_analysis.config import FOLDER, RESULTS_DIR, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES
from granular_analysis.io.loader import load_last_frame, filter_particles
from granular_analysis.plotting.z_slice_plotter import plot_z_slice


def _load_particles(timestep: str) -> np.ndarray:
    """Load particle centroids from the configured data folder."""
    if timestep != "latest":
        raise ValueError("Only 'latest' is supported for --timestep.")

    ellipsoids = load_last_frame(FOLDER)
    if USE_PARTICLE_SKIP:
        ellipsoids = filter_particles(ellipsoids, skip_first=SKIP_FIRST_PARTICLES)

    pts = np.array([e["x"] for e in ellipsoids])
    return pts[np.isfinite(pts).all(axis=1)]


def _build_output_path(z: float, timestep: str, alpha: bool) -> Path:
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "true" if alpha else "false"
    return out_dir / f"slice_z{z:.2f}_t{timestep}_alpha_{suffix}.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a xy slice of particles at a given z value."
    )
    parser.add_argument(
        "--z",
        type=float,
        required=True,
        help="Z coordinate of the slice",
    )
    parser.add_argument(
        "--timestep",
        type=str,
        default="latest",
        help="Time step identifier (only 'latest' is supported)",
    )
    parser.add_argument(
        "--alphashape",
        action="store_true",
        help="Draw an alpha shape around the slice",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path",
    )
    args = parser.parse_args()

    particles = _load_particles(args.timestep)
    output_path = Path(args.output) if args.output else _build_output_path(
        args.z, args.timestep, args.alphashape
    )

    plot_z_slice(
        particles,
        z_value=args.z,
        with_alpha_shape=args.alphashape,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
