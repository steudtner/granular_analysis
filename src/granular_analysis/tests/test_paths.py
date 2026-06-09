# Test: Welcher Input-Pfad wurde verwendet?
from granular_analysis.config import FOLDER, RESULTS_DIR
from granular_analysis.utils.physics import (
    read_gravity,
    read_density_first_particle,
    load_mass_array,
)
from pathlib import Path

project_root = Path(FOLDER).resolve().parent.parent.parent
input_global = project_root / "input.dat"
input_ellipsoid = project_root / "input_ellipsoid_data.dat"

print("=" * 40)
print("CONFIG PATHS")
print("=" * 40)
print(f"FOLDER:        {FOLDER}")
print(f"RESULTS_DIR:   {RESULTS_DIR}")
print(f"project_root:  {project_root}")
print(f"input.dat:     {input_global} ({'EXISTS' if input_global.exists() else 'MISSING'})")
print(f"input_ellipsoid_data.dat: {input_ellipsoid} ({'EXISTS' if input_ellipsoid.exists() else 'MISSING'})")
print("=" * 40)
print("PHYSICS PARAMS")
print("=" * 40)

try:
    g = read_gravity(str(input_global))
    print(f"gravity (g):   {g}")
except Exception as e:
    print(f"gravity:       FAILED ({e})")
    g = None

try:
    rho_ref = read_density_first_particle(str(input_ellipsoid))
    print(f"rho_ref:       {rho_ref}")
except Exception as e:
    print(f"rho_ref:       FAILED ({e})")
    rho_ref = None

try:
    m = load_mass_array(str(input_ellipsoid))
    print(f"mass entries:  {len(m)}")
    if len(m) > 0:
        print(f"first mass:    {m[0]}")
except Exception as e:
    print(f"mass array:     FAILED ({e})")
    m = None

print("=" * 40)
print(f"use_physical:  {'YES' if (g and rho_ref and m is not None and len(m) > 0) else 'NO (will use unit masses)'}")
print("=" * 40)