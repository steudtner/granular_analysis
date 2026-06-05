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
    rho_list, a_list, b_list, c_list = [], [], [], []
    with open(input_ellipsoid_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            rho_list.append(float(parts[1]))
            a_list.append(float(parts[2]))
            b_list.append(float(parts[3]))
            c_list.append(float(parts[4]))

    rho = np.array(rho_list)
    a = np.array(a_list)
    b = np.array(b_list)
    c = np.array(c_list)
    return rho * (4.0 / 3.0) * np.pi * a * b * c
