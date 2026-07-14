import os
import numpy as np

# Datenpfad: Umgebungsvariable ODER relatives results/ellipsoid/-Verzeichnis unter cwd
_default_data = os.path.join(os.getcwd(), "results", "ellipsoid", "ellipsoid_trn")
FOLDER = os.getenv("GRANULAR_DATA_FOLDER", _default_data)

# Ergebnisse: immer im aktuellen Arbeitsverzeichnis
RESULTS_DIR = os.path.join(os.getcwd(), "results_granular_analysis")

# Parameter
LZ = 2.5
ALPHA_2D = 0.6
ALPHA_3D = 0.6
Z_LEVELS = np.round(np.arange(0.0, LZ + 0.05, 0.05), 2)
USE_PARTICLE_SKIP = True
SKIP_FIRST_PARTICLES = 1000
D_EQ = 1.0
OUTDIR_2D = os.path.join(RESULTS_DIR, "ellipseSlices")
OUTDIR_3D = os.path.join(RESULTS_DIR, "porosity3d")