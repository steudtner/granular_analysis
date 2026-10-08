import os
import logging
import numpy as np
import pandas as pd

from shapely.geometry import Polygon
from shapely.ops import unary_union

from granular_analysis.geometry.ellipsoid import (
    ellipsoid_slice,
    ellipse_points,
)
from granular_analysis.geometry.alpha2d import (
    compute_alpha_shape_2d,
)
from granular_analysis.utils.porosity_utils import (
    compute_porosity,
)


def compute_slice_porosity(
    particles,
    z0,
    alpha,
    lz,
    densify_max_segment_length: float = None,
    n_bins: int = 50,
    skip_first: int = 0,
    output_dir: str = None,
    write_upper_edge: bool = False,
    clip_solid_to_shape: bool = False,
):
    """Compute 2D slice porosity from a particle set.

    Slices all particles at the plane z=z0 and computes
    the 2D alpha-shape porosity. Each ellipse contour uses 720 points
    to keep the polygonal area error small (about 1.3e-5 for circles).

    Uses periodic boundary conditions in z-direction
    (slices with shifts -lz, 0, lz).

    Parameters
    ----------
    particles : list
        Particle data.
    z0 : float
        Slice position.
    alpha : float
        Alpha-shape parameter.
    lz : float
        Periodic domain length in z.
    densify_max_segment_length : float, optional
        Maximum segment length for alpha-shape densification.
    n_bins : int
        Number of bins for upper-edge extraction.
    skip_first : int
        Ignore particles with IDs <= skip_first.
    output_dir : str, optional
        Directory for generated upper-edge data.
    """

    pts = []
    alpha_pts = []
    alpha_polygons = []

    # --------------------------------------------------
    # PERIODIC Z-SLICING
    # --------------------------------------------------

    for shift in [-lz, 0.0, lz]:

        for index, e in enumerate(particles, start=1):

            res = ellipsoid_slice(
                e,
                z0,
                shift,
            )

            if res is None:
                continue

            x0, y0, ax_e, by_e, theta = res

            ellipse = ellipse_points(
                x0,
                y0,
                ax_e,
                by_e,
                theta=theta,
                n=720,
            )

            pts.append(ellipse)

            particle_id = int(
                e.get(
                    "id",
                    e.get("idx", index),
                )
            )

            if particle_id > skip_first:
                alpha_pts.append(ellipse)
                alpha_polygons.append(
                    Polygon(ellipse)
                )

    # --------------------------------------------------
    # NO SLICE DATA
    # --------------------------------------------------

    if not pts:
        return None

    # --------------------------------------------------
    # CLEAN POINTS
    # --------------------------------------------------

    pts = np.vstack(pts)

    pts = pts[
        np.isfinite(pts).all(axis=1)
    ]

    if alpha_pts:
        alpha_pts = np.vstack(alpha_pts)

        alpha_pts = alpha_pts[
            np.isfinite(alpha_pts).all(axis=1)
        ]
    else:
        alpha_pts = np.empty(
            (0, 2)
        )

    # --------------------------------------------------
    # ELLIPSE AREA
    # --------------------------------------------------

    A_ell = (
        float(
            unary_union(
                alpha_polygons
            ).area
        )
        if alpha_polygons
        else 0.0
    )

    logging.getLogger(__name__).debug(
        "Slice z=%s: %s points; %s alpha-shape points",
        z0, pts.shape[0], alpha_pts.shape[0],
    )

    # --------------------------------------------------
    # ALPHA SHAPE
    # --------------------------------------------------

    shape, A_alpha = compute_alpha_shape_2d(
        alpha_pts,
        alpha,
        densify_max_segment_length=(
            densify_max_segment_length
        ),
    )

    # A porosity control volume must count only solid inside that volume.
    # Keep the legacy behavior unless the caller explicitly selects clipping.
    if clip_solid_to_shape and shape is not None and len(shape) >= 3:
        A_ell = float(unary_union(alpha_polygons).intersection(Polygon(shape)).area)

    # --------------------------------------------------
    # UPPER EDGE
    # --------------------------------------------------

    if (
        shape is not None
        and write_upper_edge
        and output_dir is not None
    ):
        extract_upper_edge(
            shape,
            n_bins=n_bins,
            output_dir=output_dir,
        )

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    return {
        **surface_metrics(shape),
        "shape": shape,
        "A_ell": A_ell,
        "A_alpha": A_alpha,
        "porosity": compute_porosity(
            A_alpha,
            A_ell,
        ),
        "n": len(particles),
        "n_alpha_particles": sum(
            int(
                e.get(
                    "id",
                    e.get("idx", index),
                )
            ) > skip_first
            for index, e in enumerate(
                particles,
                start=1,
            )
        ),
    }


def surface_metrics(shape_coords):
    """Spatial extents of the slice boundary, in simulation length units.

    Runout is the full x extent, and remaining height is measured above y=0.
    These describe one frame, not a maximum over simulation time.
    """
    if shape_coords is None or len(shape_coords) < 3:
        return dict(x_min=None, x_max=None, max_runout_length=None, remaining_height=None)
    coords = np.asarray(shape_coords, dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Surface coordinates must be finite")
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    return dict(x_min=x_min, x_max=x_max, max_runout_length=x_max-x_min,
                remaining_height=float(coords[:, 1].max()))


def upper_edge_points(shape_coords, n_bins=50, max_jump=2.0):
    """Return the highest polygon point per x-bin with jump filtering."""
    if shape_coords is None or len(shape_coords) < 3:
        return None, None
    coordinates = np.asarray(shape_coords, dtype=float)
    x_values, y_values = coordinates[:, 0], coordinates[:, 1]
    bins = np.linspace(x_values.min(), x_values.max(), n_bins + 1)
    indices = np.clip(np.digitize(x_values, bins), 1, n_bins)
    raw = []
    for bin_index in range(1, len(bins)):
        mask = indices == bin_index
        if np.any(mask):
            local = np.flatnonzero(mask)
            point_index = local[np.argmax(y_values[mask])]
            raw.append((x_values[point_index], y_values[point_index]))
    if not raw:
        return None, None
    filtered = [raw[0]]
    for point in raw[1:]:
        if abs(point[1] - filtered[-1][1]) <= max_jump:
            filtered.append(point)
    values = np.asarray(filtered)
    return values[:, 0], values[:, 1]


def extract_upper_edge(shape_coords, n_bins=50, output_dir=None):
    """Extract and save the canonical upper-edge profile."""
    if output_dir is None:
        raise ValueError("output_dir must be provided to extract_upper_edge().")
    x_values, y_values = upper_edge_points(shape_coords, n_bins=n_bins)
    if x_values is None:
        return None
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    frame = pd.DataFrame({"x": x_values, "z_max": y_values})
    out_path = os.path.join(output_dir, "alphashape_last_timestep.csv")
    frame.to_csv(out_path, index=False)
    print(f"Saved {len(frame)} upper-edge points to: {out_path}")
    return frame
