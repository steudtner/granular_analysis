import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# Case-Root: src/granular_analysis/config.py -> parent=granular_analysis -> parent=postproc -> parent=case
_CASE_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


@dataclass(frozen=True)
class Config:
    data_folder: Path
    results_dir: Path
    lz: float = 2.5
    z_step: float = 0.1
    alpha_2d: float = 0.6
    alpha_3d: float = 0.6
    use_particle_skip: bool = True
    skip_first_particles: int = 1000
    d_eq: float = 1.0

    def __post_init__(self):
        object.__setattr__(
            self, 'data_folder', Path(self.data_folder).expanduser().resolve()
        )
        object.__setattr__(
            self, 'results_dir', Path(self.results_dir).expanduser().resolve()
        )

    @property
    def z_levels(self) -> np.ndarray:
        steps = int(round(self.lz / self.z_step)) + 1
        return np.round(np.linspace(0.0, self.lz, steps), 2)

    @property
    def outdir_2d(self) -> Path:
        return self.results_dir / "ellipseSlices"

    @property
    def outdir_3d(self) -> Path:
        return self.results_dir / "porosity3d"


_CONFIG = Config(
    data_folder=os.getenv("GRANULAR_DATA_FOLDER", 
                          _CASE_ROOT / "results" / "ellipsoid" / "ellipsoid_trn"),
    results_dir=os.getenv("GRANULAR_RESULTS_DIR", 
                           _CASE_ROOT / "results_granular_analysis"),
)

FOLDER = str(_CONFIG.data_folder)
RESULTS_DIR = str(_CONFIG.results_dir)
LZ = _CONFIG.lz
Z_LEVELS = _CONFIG.z_levels
ALPHA_2D = _CONFIG.alpha_2d
ALPHA_3D = _CONFIG.alpha_3d
USE_PARTICLE_SKIP = _CONFIG.use_particle_skip
SKIP_FIRST_PARTICLES = _CONFIG.skip_first_particles
OUTDIR_2D = str(_CONFIG.outdir_2d)
OUTDIR_3D = str(_CONFIG.outdir_3d)
D_EQ = _CONFIG.d_eq