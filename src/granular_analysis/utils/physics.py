"""Physical constants and input-file parsers for granular simulations."""

import re
from pathlib import Path

import numpy as np


def read_gravity(input_global_path):
    """Read gravity vector magnitude from input.dat-style file."""
    grx = gry = grz = None
    with open(input_global_path, "r") as f:
        for line in f:
            if "grx" in line:
                grx = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            elif "gry" in line:
                gry = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            elif "grz" in line:
                grz = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
    if grx is None or gry is None or grz is None:
        raise RuntimeError("Could not read gravity (grx/gry/grz) from input file")
    return np.sqrt(grx**2 + gry**2 + grz**2)


def read_density_first_particle(input_ellipsoid_path):
    """Read density from the first line of input_ellipsoid_data.dat."""
    with open(input_ellipsoid_path, "r") as f:
        line = f.readline()
    parts = line.strip().split(",")
    return float(parts[1])


def load_mass_array(input_ellipsoid_path):
    """Build mass array from ellipsoid geometry data.
    
    File format per line: index,rho,a,b,c
    Returns: m = rho * (4/3) * pi * a * b * c  for each entry.
    """
    _, masses, _, _, _ = load_ellipsoid_properties(input_ellipsoid_path)
    return masses


def load_ellipsoid_properties(input_ellipsoid_path):
    """Load IDs, masses, and principal semi-axes from ellipsoid input data.

    The input file may contain human-readable header lines. Only rows
    whose first five comma-separated fields are numeric are processed.

    Returns
    -------
    ids : np.ndarray
        Rigid-body IDs.
    masses : np.ndarray
        Particle masses computed from rho, a, b, c.
    a, b, c : np.ndarray
        Semi-principal axes.
    """
    ids = []
    rho_list = []
    a_list = []
    b_list = []
    c_list = []

    with open(input_ellipsoid_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            # Ignore empty lines.
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]

            # We need at least:
            # ID, rho, a, b, c
            if len(parts) < 5:
                continue

            # Skip header / non-numeric lines.
            try:
                particle_id = int(parts[0])
                rho = float(parts[1])
                a = float(parts[2])
                b = float(parts[3])
                c = float(parts[4])
            except ValueError:
                continue

            ids.append(particle_id)
            rho_list.append(rho)
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)

    if not ids:
        raise RuntimeError(
            "No valid ellipsoid data rows found in "
            f"{input_ellipsoid_path}"
        )

    ids = np.asarray(ids, dtype=int)
    rho = np.asarray(rho_list, dtype=float)
    a = np.asarray(a_list, dtype=float)
    b = np.asarray(b_list, dtype=float)
    c = np.asarray(c_list, dtype=float)

    masses = rho * (4.0 / 3.0) * np.pi * a * b * c

    return ids, masses, a, b, c