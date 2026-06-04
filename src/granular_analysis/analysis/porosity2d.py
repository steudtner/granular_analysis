import numpy as np

from granular_analysis.geometry.ellipsoid import ellipsoid_slice, ellipse_points
from granular_analysis.geometry.alpha2d import compute_alpha_shape_2d
from granular_analysis.utils.porosity_utils import compute_porosity


def compute_slice_porosity(particles, z0, alpha, lz):
    """Compute 2D slice porosity from a particle set.

    Pipeline:
    - slice ellipsoids in z-direction (including periodic shifts)
    - project to 2D ellipses
    - compute ellipsoid area analytically
    - compute alpha shape area
    - compute porosity
    """
    pts = []
    A_ell = 0.0

    for shift in [-lz, 0.0, lz]:
        for e in particles:
            res = ellipsoid_slice(e, z0, shift)
            if res is None:
                continue
            x0, y0, ax_e, by_e = res
            pts.append(ellipse_points(x0, y0, ax_e, by_e))
            A_ell += np.pi * ax_e * by_e

    if not pts:
        return None

    pts = np.vstack(pts)
    pts = pts[np.isfinite(pts).all(axis=1)]

    shape, A_alpha = compute_alpha_shape_2d(pts, alpha)

    return {
        "shape": shape,
        "A_ell": A_ell,
        "A_alpha": A_alpha,
        "porosity": compute_porosity(A_alpha, A_ell),
        "n": len(particles),
    }
