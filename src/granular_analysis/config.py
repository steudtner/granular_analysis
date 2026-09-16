import os
import numpy as np


def _default_data_folder() -> str:
    return os.path.join(
        os.getcwd(),
        "results",
        "ellipsoid",
        "ellipsoid_trn",
    )


def _default_results_dir() -> str:
    return os.path.join(
        os.getcwd(),
        "results_granular_analysis",
    )


# ============================================================
# DATEN
# ============================================================
# Priorität:
# 1. GRANULAR_DATA_FOLDER
# 2. automatisch: ./results/ellipsoid/ellipsoid_trn
FOLDER = (
    os.getenv("GRANULAR_DATA_FOLDER")
    or _default_data_folder()
)


# ============================================================
# ERGEBNISSE
# ============================================================
# Priorität:
# 1. GRANULAR_RESULTS_DIR
# 2. automatisch: ./results_granular_analysis
RESULTS_DIR = (
    os.getenv("GRANULAR_RESULTS_DIR")
    or _default_results_dir()
)


# ============================================================
# PARAMETER
# ============================================================
LZ = 5.0

ALPHA_2D = 0.6
ALPHA_3D = 0.6

Z_LEVELS = np.round(
    np.arange(0.0, LZ + 0.1, 0.05),
    2,
)

USE_PARTICLE_SKIP = True
SKIP_FIRST_PARTICLES = 1000

D_EQ = 1.0


# ============================================================
# OUTPUT-UNTERVERZEICHNISSE
# ============================================================
OUTDIR_2D = os.path.join(
    RESULTS_DIR,
    "Slices",
)

OUTDIR_SLOPE = os.path.join(
    RESULTS_DIR,
    "slope_analysis",
)


# ============================================================
# SLOPE ANALYSIS
# ============================================================
SLOPE_XLIM = (0.0, 60.0)
SLOPE_YLIM = (0.0, LZ)

SLOPE_COLORS = [
    "blue",
    "orange",
    "purple",
    "brown",
]
# ------------------------------------------------------------
# 3D particle rendering
# ------------------------------------------------------------

RENDER_BOUNDS = (
    0.0, 60.0,   # x_min, x_max
    0.0, 20.0,   # y_min, y_max
    0.0, 5.0,    # z_min, z_max
)
