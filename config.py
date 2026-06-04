import os
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT_DIR, "results_granular_analysis")

FOLDER = "/data/horse/ws/s4713238-DA/DiscretizationGranularCollapse/primeJainSU/results/ellipsoid/ellipsoid_trn"

LZ = 2.5

ALPHA_2D = 0.6
ALPHA_3D = 0.6

Z_LEVELS = np.round(
    np.arange(0.0, 2.5 + 0.05, 0.05),
    2
)

USE_PARTICLE_SKIP = True
SKIP_FIRST_PARTICLES = 1000

OUTDIR_2D = os.path.join(RESULTS_DIR, "ellipseSlices")
OUTDIR_3D = os.path.join(RESULTS_DIR, "porosity3d")