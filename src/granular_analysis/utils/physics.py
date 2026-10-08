"""Physical constants and input-file parsers for granular simulations."""

import re
import numpy as np


_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")


def _last_float(line):
    values = _FLOAT_RE.findall(line)
    if not values:
        raise ValueError(f"No numeric value found in line: {line.rstrip()}")
    return float(values[-1].replace("D", "E").replace("d", "e"))


def read_gravity(input_global_path):
    """Read gravity vector magnitude from input.dat-style file."""
    return float(np.linalg.norm(read_gravity_vector(input_global_path)))


def read_gravity_vector(input_global_path):
    """Read signed gravity components in simulation coordinates."""
    grx = gry = grz = None
    with open(input_global_path, "r") as f:
        for line in f:
            line = line.split("!", 1)[0]
            if "grx" in line:
                grx = _last_float(line)
            elif "gry" in line:
                gry = _last_float(line)
            elif "grz" in line:
                grz = _last_float(line)
    if grx is None or gry is None or grz is None:
        raise RuntimeError("Could not read gravity (grx/gry/grz) from input file")
    return np.array([grx, gry, grz], dtype=float)


def initial_potential_energy(input_global_path, input_ellipsoid_path, skip_first=0):
    """Sum m*|g|*y after skipping input rows, with zero potential at y=0.

    Uses initial input positions, not restart/monitor positions. No buoyancy
    subtraction and no removal of the final bed's residual potential energy.
    """
    import csv
    if skip_first < 0:
        raise ValueError("skip_first must be nonnegative")
    gravity = read_gravity(input_global_path)
    energies = []
    row = 0
    with open(input_ellipsoid_path) as stream:
        for parts in csv.reader(stream):
            try:
                int(parts[0])
                values = [float(v.replace("D", "E").replace("d", "e")) for v in parts[1:5]]
            except (ValueError, IndexError):
                continue
            row += 1
            if row <= skip_first:
                continue
            if len(parts) < 8:
                raise ValueError("Initial particle positions (columns 6-8) missing")
            rho, a, b, c = values
            position = np.array([float(v.replace("D", "E").replace("d", "e")) for v in parts[5:8]])
            if not np.isfinite([rho, a, b, c, *position]).all() or min(rho, a, b, c) <= 0:
                raise ValueError("Invalid initial particle geometry/density/position")
            energies.append(rho * (4/3)*np.pi*a*b*c * gravity * position[1])
    if not energies or not np.isfinite(gravity).all():
        raise ValueError("Cannot determine initial potential energy")
    return float(np.sum(energies))


def read_fluid_density(input_global_path):
    """Read rho_f (water/fluid density) from the simulation input."""
    with open(input_global_path) as stream:
        for line in stream:
            line = line.split("!", 1)[0]
            match = re.match(r"\s*rho_f\s*=\s*(" + _FLOAT_RE.pattern + r")", line, re.I)
            if match:
                density = float(match.group(1).replace("D", "E").replace("d", "e"))
                if not np.isfinite(density) or density <= 0:
                    raise ValueError("Fluid density rho_f must be finite and positive.")
                return density
    raise ValueError(f"Missing fluid density rho_f in {input_global_path}")


def read_density_first_particle(input_ellipsoid_path):
    """Read density from the first valid particle row, allowing headers."""
    with open(input_ellipsoid_path, "r") as stream:
        for line in stream:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                int(parts[0])
                return float(parts[1])
            except ValueError:
                continue
    raise RuntimeError(f"No valid particle density found in {input_ellipsoid_path}")


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
