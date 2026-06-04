import numpy as np

from granular_analysis.geometry.ellipsoid import (
    ellipsoid_slice,
    ellipse_points
)

from granular_analysis.geometry.alpha2d import compute_alpha_shape_2d


def compute_alpha_shape_3d_sliced(particles, z_levels, alpha, lz):
    """
    Robust pseudo-3D alpha volume via stacked 2D alpha slices.

    This replaces unstable 3D alpha shapes with a slice-integrated model:

        V ≈ ∑ A_alpha(z) * Δz

    Returns:
        dict with:
            V_alpha_sliced
            V_ell_sliced
            porosity
            slice_data
    """

    dz = float(np.mean(np.diff(z_levels)))

    V_alpha = 0.0
    V_ell = 0.0

    slice_data = []

    for z0 in z_levels:

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

        if len(pts) == 0:
            slice_data.append({
                "z": z0,
                "A_alpha": 0.0,
                "A_ell": 0.0
            })
            continue

        pts = np.vstack(pts)
        pts = pts[np.isfinite(pts).all(axis=1)]

        _, A_alpha = compute_alpha_shape_2d(pts, alpha)

        V_alpha += A_alpha * dz
        V_ell += A_ell * dz

        slice_data.append({
            "z": z0,
            "A_alpha": A_alpha,
            "A_ell": A_ell
        })

    if V_alpha <= 0:
        return {
            "V_alpha_sliced": np.nan,
            "V_ell_sliced": np.nan,
            "porosity": np.nan,
            "slice_data": slice_data
        }

    porosity = (V_alpha - V_ell) / V_alpha

    return {
        "V_alpha_sliced": V_alpha,
        "V_ell_sliced": V_ell,
        "porosity": porosity,
        "slice_data": slice_data
    }