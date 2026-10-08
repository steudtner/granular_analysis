"""Print resolved case paths and physical-input diagnostics."""

from pathlib import Path

from granular_analysis.config import FOLDER, RESULTS_DIR
from granular_analysis.utils.physics import (
    load_mass_array,
    read_density_first_particle,
    read_gravity,
)


def main():
    project_root = Path(FOLDER).resolve().parent.parent.parent
    input_global = project_root / "input.dat"
    input_ellipsoid = project_root / "input_ellipsoid_data.dat"
    print(f"FOLDER: {FOLDER}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"input.dat: {input_global}")
    print(f"input_ellipsoid_data.dat: {input_ellipsoid}")
    gravity = read_gravity(input_global)
    density = read_density_first_particle(input_ellipsoid)
    masses = load_mass_array(input_ellipsoid)
    print(f"gravity: {gravity}")
    print(f"reference density: {density}")
    print(f"mass entries: {len(masses)}")


if __name__ == "__main__":
    main()
