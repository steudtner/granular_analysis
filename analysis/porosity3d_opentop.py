import numpy as np

from granular_analysis.geometry.ellipsoid import ellipsoid_volume
from granular_analysis.geometry.alpha3d_opentop_pca import alpha_shape_opentop_pca
from granular_analysis.utils.logging_utils import suppress_noise
from granular_analysis.utils.porosity_utils import compute_porosity


def compute_porosity_3d_opentop(particles, alpha=0.6, z_min=None, z_max=None):
    suppress_noise()

    pts = []
    V_ell = 0.0

    for e in particles:
        V_ell += ellipsoid_volume(e)
        pts.append(np.array(e["x"]))

    pts = np.vstack(pts)

    mesh, V_alpha = alpha_shape_opentop_pca(pts, alpha=alpha, z_min=z_min, z_max=z_max)

    if mesh is None:
        return None

    return {
        "mesh": mesh,
        "V_ell": V_ell,
        "V_alpha": V_alpha,
        "porosity": compute_porosity(V_alpha, V_ell),
    }
