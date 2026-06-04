import numpy as np
import warnings

from granular_analysis.geometry.ellipsoid import (
    ellipsoid_volume,
    ellipsoid_surface_points,
)
from granular_analysis.geometry.alpha3d import compute_alpha_shape_3d
from granular_analysis.utils.porosity_utils import compute_porosity


def build_surface_cloud(particles, res):
    """Build surface point cloud from ellipsoids."""
    pts = [ellipsoid_surface_points(e, res, res) for e in particles]
    if not pts:
        return None
    pts = np.vstack(pts)
    return pts[np.isfinite(pts).all(axis=1)]


def compute_ellipsoid_volume(particles):
    """Analytical ellipsoid volume sum."""
    return float(sum(ellipsoid_volume(e) for e in particles))


def compute_porosity_3d(particles, alpha=0.6, res=10):
    """Full 3D porosity pipeline.

    Steps:
    - ellipsoid volume (analytic)
    - surface sampling
    - alpha shape reconstruction
    - porosity computation
    """
    V_ell = compute_ellipsoid_volume(particles)

    pts = build_surface_cloud(particles, res)
    if pts is None:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Singular matrix.*")
        mesh, V_alpha = compute_alpha_shape_3d(pts, alpha, include_origin=True)

    V_alpha = abs(V_alpha)
    if V_alpha <= 0 or not np.isfinite(V_alpha):
        raise RuntimeError("Invalid alpha shape volume for 3D porosity")

    return {
        "mesh": mesh,
        "V_ell": V_ell,
        "V_alpha": V_alpha,
        "porosity": compute_porosity(V_alpha, V_ell),
    }
