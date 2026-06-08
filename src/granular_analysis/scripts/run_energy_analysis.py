#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from granular_analysis.config import (
    FOLDER,
    RESULTS_DIR,
    D_EQ,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
)
from granular_analysis.io.loader import get_all_bin_files
from granular_analysis.io.read_ellipsoid_data import read_ellipsoid_trn
from granular_analysis.analysis.energy.kinetic_energy import (
    kinetic_energy,
    total_kinetic_energy,
)
from granular_analysis.plotting.plots import (
    plot_energy_time_series,
    plot_energy_histogram,
)
from granular_analysis.utils.physics import (
    read_gravity,
    read_density_first_particle,
    load_mass_array,
)


def _auto_input_paths():
    project_root = Path(FOLDER).resolve().parent.parent.parent
    return {
        "global": project_root / "input.dat",
        "ellipsoid": project_root / "input_ellipsoid_data.dat",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute kinetic energy time series and histograms "
                    "from ellipsoid transient data."
    )
    parser.add_argument(
        "--out",
        default=os.path.join(RESULTS_DIR, "energy.csv"),
        help="CSV output filename for the energy time series.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the energy CSV even if it already exists.",
    )
    parser.add_argument(
        "--histogram-timesteps",
        default=None,
        help="Comma-separated frame indices for histograms. "
             "Use -1 for last frame. Example: '500,1000,-1'",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do not skip any particles (ignore USE_PARTICLE_SKIP).",
    )
    parser.add_argument(
        "--no-physical-mass",
        action="store_true",
        help="Use unit masses instead of physical ellipsoid masses.",
    )
    return parser.parse_args()


def _parse_histogram_timesteps(arg, total_files):
    if arg is None:
        return []
    indices = []
    for s in arg.split(","):
        s = s.strip()
        if s == "-1":
            indices.append(total_files - 1)
        else:
            idx = int(s)
            if idx < 0:
                idx = total_files + idx
            indices.append(idx)
    return sorted(set(i for i in indices if 0 <= i < total_files))


def compute_energy_dataframe(folder, use_physical_mass=True,
                               skip_first=0, input_paths=None):
    files = get_all_bin_files(folder)
    energies, energies_norm, times, times_norm = [], [], [], []

    if use_physical_mass:
        if input_paths is None:
            input_paths = _auto_input_paths()
        g = read_gravity(input_paths["global"])
        rho_ref = read_density_first_particle(input_paths["ellipsoid"])
        mass_arr = load_mass_array(input_paths["ellipsoid"])
        normalization = rho_ref * g
    else:
        g = 0.0
        mass_arr = None
        normalization = 1.0

    for f in files:
        path = os.path.join(folder, f)
        ellipsoids = read_ellipsoid_trn(path)

        if skip_first > 0:
            ellipsoids = ellipsoids[skip_first:]

        n_particles = len(ellipsoids)

        if mass_arr is not None:
            if n_particles > len(mass_arr):
                raise RuntimeError(
                    f"More particles in binary ({n_particles}) than "
                    f"mass entries ({len(mass_arr)}). "
                    f"Check input_ellipsoid_data.dat."
                )
            masses = mass_arr[:n_particles]
        else:
            masses = np.ones(n_particles)

        velocities = np.array([e["u"] for e in ellipsoids])
        ke_particles = kinetic_energy(masses, velocities)
        ke_total = np.sum(ke_particles)

        energies.append(ke_total)
        energies_norm.append(ke_total / normalization if normalization != 0 else ke_total)

        t = ellipsoids[0]["t"]
        times.append(t)
        times_norm.append(t * np.sqrt(g) if g > 0 else t)

    return pd.DataFrame({
        "time": times,
        "time_norm": times_norm,
        "kinetic_energy": energies,
        "kinetic_energy_norm": energies_norm,
    })


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    energy_png = os.path.splitext(args.out)[0] + ".png"

    skip_first = 0 if args.no_skip else (SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0)
    use_physical = not args.no_physical_mass
    input_paths = _auto_input_paths()

    if use_physical:
        for key, path in input_paths.items():
            if not path.exists():
                print(f"WARNING: Input file not found: {path}")
                print("Falling back to unit masses.")
                use_physical = False
                break

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} exists. Use --force to recompute.")
        df = pd.read_csv(args.out)
    else:
        df = compute_energy_dataframe(
            FOLDER,
            use_physical_mass=use_physical,
            skip_first=skip_first,
            input_paths=input_paths,
        )
        df.to_csv(args.out, index=False)
        print(f"Wrote energy data to {args.out}")

    plot_energy_time_series(df, energy_png, use_normalized=use_physical)
    print(f"Wrote energy plot to {energy_png}")

    all_files = get_all_bin_files(FOLDER)
    hist_indices = _parse_histogram_timesteps(args.histogram_timesteps, len(all_files))

    if hist_indices:
        hist_dir = Path(RESULTS_DIR) / "energy_histograms"
        hist_dir.mkdir(parents=True, exist_ok=True)

        if use_physical:
            g = read_gravity(input_paths["global"])
            rho_ref = read_density_first_particle(input_paths["ellipsoid"])
            mass_arr = load_mass_array(input_paths["ellipsoid"])
            normalization = rho_ref * g
        else:
            mass_arr = None
            normalization = 1.0

        print(f"\nGenerating {len(hist_indices)} histograms...")
        for idx in hist_indices:
            f = all_files[idx]
            path = os.path.join(FOLDER, f)
            ellipsoids = read_ellipsoid_trn(path)

            if skip_first > 0:
                ellipsoids = ellipsoids[skip_first:]

            n_particles = len(ellipsoids)
            if mass_arr is not None:
                masses = mass_arr[:n_particles]
            else:
                masses = np.ones(n_particles)

            velocities = np.array([e["u"] for e in ellipsoids])
            ke_particles = kinetic_energy(masses, velocities)

            t_label = f"t={ellipsoids[0]['t']:.4f}"
            out_path = hist_dir / f"particle_energy_hist_{f.replace('.bin', '')}.png"

            plot_energy_histogram(ke_particles, t_label, out_path, normalization=normalization)
            print(f"  {out_path.name}")

        print(f"\nSaved histograms to {hist_dir}")

    print("\nEnergy analysis done")


if __name__ == "__main__":
    main()
