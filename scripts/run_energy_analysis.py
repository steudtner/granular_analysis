import argparse
import os
import numpy as np
import pandas as pd

from granular_analysis.config import FOLDER, RESULTS_DIR
from granular_analysis.io.loader import get_all_bin_files
from granular_analysis.read_ellipsoid_data import read_ellipsoid_trn
from granular_analysis.analysis.energy.kinetic_energy import total_kinetic_energy
from granular_analysis.plotting.plots import plot_energy_time_series


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute kinetic energy time series from ellipsoid transient data."
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
    return parser.parse_args()


def compute_energy_dataframe(folder):
    files = get_all_bin_files(folder)
    energies, times = [], []

    for f in files:
        path = os.path.join(folder, f)
        ellipsoids = read_ellipsoid_trn(path)
        masses = np.ones(len(ellipsoids))
        velocities = np.array([e["u"] for e in ellipsoids])
        energies.append(total_kinetic_energy(masses, velocities))
        times.append(ellipsoids[0]["t"])

    return pd.DataFrame({"time": times, "kinetic_energy": energies})


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    energy_png = os.path.splitext(args.out)[0] + ".png"

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} already exists. Use --force to recompute.")
        df = pd.read_csv(args.out)
    else:
        df = compute_energy_dataframe(FOLDER)
        df.to_csv(args.out, index=False)
        print(f"Wrote energy data to {args.out}")

    plot_energy_time_series(df, energy_png)
    print(f"Wrote energy plot to {energy_png}")
    print("Energy analysis done")


if __name__ == "__main__":
    main()
