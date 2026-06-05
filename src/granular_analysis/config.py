import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Config:
    """Immutable configuration object."""
    data_folder: Path
    results_dir: Path
    lz: float = 2.5
    alpha_2d: float = 0.6
    alpha_3d: float = 0.6
    use_particle_skip: bool = True
    skip_first_particles: int = 1000
    d_eq: float = 1.0 

    def __post_init__(self):
        # frozen=True erlaubt normale Zuweisungen nicht,
        # daher object.__setattr__ für Path-Auflösung
        object.__setattr__(
            self, 'data_folder', Path(self.data_folder).expanduser().resolve()
        )
        object.__setattr__(
            self, 'results_dir', Path(self.results_dir).expanduser().resolve()
        )

    @property
    def z_levels(self) -> np.ndarray:
        """Robust z-level generation avoiding floating point drift."""
        steps = int(round(self.lz / 0.05)) + 1
        return np.round(np.linspace(0.0, self.lz, steps), 2)

    @property
    def outdir_2d(self) -> Path:
        return self.results_dir / "ellipseSlices"

    @property
    def outdir_3d(self) -> Path:
        return self.results_dir / "porosity3d"


def _env_path(var_name: str, fallback: Path) -> Path:
    """Read a path from environment variable or use fallback."""
    env = os.getenv(var_name)
    if env:
        return Path(env).expanduser().resolve()
    return fallback


# ----------------------------------------------------------------------
# Backwards-compatible module-level constants
# ----------------------------------------------------------------------
#
# A default instance is created at import time.  On the cluster you can
# simply set the environment variable before starting Python:
#
#     export GRANULAR_DATA_FOLDER=/data/horse/...
#     export GRANULAR_RESULTS_DIR=~/results_granular_analysis
#
# Existing scripts that do
#     from granular_analysis.config import FOLDER, OUTDIR_2D
# continue to work without any changes.
# ----------------------------------------------------------------------

_CONFIG = Config(
    data_folder=_env_path("GRANULAR_DATA_FOLDER", Path.cwd() / "data"),
    results_dir=_env_path("GRANULAR_RESULTS_DIR", Path(__file__).resolve().parent.parent.parent / "results_granular_analysis"),
)

FOLDER = str(_CONFIG.data_folder)
RESULTS_DIR = str(_CONFIG.results_dir)
LZ = _CONFIG.lz
ALPHA_2D = _CONFIG.alpha_2d
ALPHA_3D = _CONFIG.alpha_3d
Z_LEVELS = _CONFIG.z_levels
USE_PARTICLE_SKIP = _CONFIG.use_particle_skip
SKIP_FIRST_PARTICLES = _CONFIG.skip_first_particles
OUTDIR_2D = str(_CONFIG.outdir_2d)
OUTDIR_3D = str(_CONFIG.outdir_3d)
D_EQ = _CONFIG.d_eq          