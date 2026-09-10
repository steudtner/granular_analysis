import os
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
):
    """Compute 2D slice porosity from a particle set.

    Slices all particles at the plane z=z0 and computes
    the 2D alpha-shape porosity.

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

    print(
        f"Slice points: {pts.shape[0]}; "
        f"alpha-shape points: {alpha_pts.shape[0]}"
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

    # --------------------------------------------------
    # UPPER EDGE
    # --------------------------------------------------

    if (
        shape is not None
        and abs(z0 - 2.5) < 1e-6
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


def extract_upper_edge(
    shape_coords,
    n_bins=50,
    output_dir=None,
):
    """Extract the upper edge from a closed 2D alpha-shape polygon.

    Bins the x-axis into n_bins intervals and takes
    the maximum y-value in each bin.

    Bins with a deviation greater than 2.0 from
    the previous valid bin are skipped.
    """

    if output_dir is None:
        raise ValueError(
            "output_dir must be provided to "
            "extract_upper_edge()."
        )

    # --------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------

    output_dir = os.path.abspath(
        output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # --------------------------------------------------
    # COORDINATES
    # --------------------------------------------------

    x = shape_coords[:, 0]
    y = shape_coords[:, 1]

    x_min = np.min(x)
    x_max = np.max(x)

    bins = np.linspace(
        x_min,
        x_max,
        n_bins + 1,
    )

    indices = np.digitize(
        x,
        bins,
    )

    # --------------------------------------------------
    # BINNING
    # --------------------------------------------------

    x_raw = []
    y_raw = []

    for i in range(
        1,
        len(bins),
    ):

        mask = indices == i

        if np.any(mask):

            y_max = np.max(
                y[mask]
            )

            y_raw.append(
                y_max
            )

            x_raw.append(
                x[mask][
                    np.argmax(y[mask])
                ]
            )

    if not x_raw:
        return

    # --------------------------------------------------
    # FILTER
    # --------------------------------------------------

    x_out = [x_raw[0]]
    y_out = [y_raw[0]]

    for i in range(
        1,
        len(x_raw),
    ):

        if abs(
            y_raw[i] - y_out[-1]
        ) <= 2.0:

            x_out.append(
                x_raw[i]
            )

            y_out.append(
                y_raw[i]
            )

    # --------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------

    df_out = pd.DataFrame(
        {
            "x": x_out,
            "z_max": y_out,
        }
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    out_path = os.path.join(
        output_dir,
        "alphashape_last_timestep.csv",
    )

    df_out.to_csv(
        out_path,
        index=False,
        header=False,
    )

    print(
        f"Extracted upper edge points: "
        f"{len(x_out)} (n_bins={n_bins})"
    )

    print(
        f"Saved upper edge (z0=2.5) to: "
        f"{out_path}"
    )
