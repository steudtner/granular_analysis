import pathlib
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from granular_analysis.config import RESULTS_DIR
from granular_analysis.plotting.helpers import save_figure
from granular_analysis.geometry.alpha2d import compute_alpha_shape_2d
from granular_analysis.geometry.ellipsoid import ellipsoid_slice, ellipse_points

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Noto Sans",
    "font.size": 10,

    "axes.labelsize": 10,
    "axes.titlesize": 10,

    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    "legend.fontsize": 9,

    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _extract_upper_edge(
    shape_coords: np.ndarray,
    n_bins: int = 50
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract a simple upper-edge profile from a closed alpha-shape polygon."""
    if shape_coords is None or len(shape_coords) < 3:
        return None, None

    x = shape_coords[:, 0]
    y = shape_coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    indices = np.digitize(x, bins)

    x_raw, y_raw = [], []
    for i in range(1, len(bins)):
        mask = indices == i
        if np.any(mask):
            x_raw.append(np.mean(x[mask]))
            y_raw.append(np.max(y[mask]))

    if not x_raw:
        return None, None

    x_out, y_out = [x_raw[0]], [y_raw[0]]
    for i in range(1, len(x_raw)):
        if abs(y_raw[i] - y_out[-1]) <= 2.0:
            x_out.append(x_raw[i])
            y_out.append(y_raw[i])

    return np.asarray(x_out), np.asarray(y_out)


def plot_exact_slice_from_ellipsoids(
    ellipsoids,
    z_plane: float,
    *,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    alphashape: bool = False,
    alpha: float = 0.6,
    upper_edge: bool = False,
    upper_edge_bins: int = 50,
    densify_max_segment_length: Optional[float] = None,
    skip_points: int = 0,
    xlim: Tuple[float, float] = (0, 60),
    ylim: Tuple[float, float] = (0, 20),
):
    """Plot exact ellipsoid slices at a single z-plane and optionally an alpha-shape.

    Ellipses are stored for visualization only as their geometric parameters
    (x0, y0, ax, by, theta). Boundary points are generated separately and only
    for the alpha-shape analysis.

    `skip_points` excludes particles with `particle_id <= skip_points` from the
    geometric analysis (alpha-shape / upper-edge extraction) only. All particles
    are always shown in the resulting plot.
    """

    # ------------------------------------------------------------------
    # Visualization data:
    # Only the geometric parameters of each ellipse are stored.
    # No discretized ellipse points are generated here.
    # ------------------------------------------------------------------
    ellipse_geometries = []
    ellipse_areas = []

    # ------------------------------------------------------------------
    # Alpha-shape data:
    # These are the discretized ellipse boundary points.
    # ------------------------------------------------------------------
    alpha_segments = []
    alpha_areas = []

    for index, e in enumerate(ellipsoids, start=1):

        res = ellipsoid_slice(e, z_plane)

        if res is None:
            continue

        x0, y0, ax_e, by_e, theta = res

        # --------------------------------------------------------------
        # Visualization:
        # Store only ellipse geometry.
        # --------------------------------------------------------------
        ellipse_geometries.append(
            (x0, y0, ax_e, by_e, theta)
        )

        ellipse_areas.append(np.pi * ax_e * by_e)

        # --------------------------------------------------------------
        # Analysis:
        # Generate boundary points ONLY for particles that are actually
        # used in the alpha-shape calculation.
        # --------------------------------------------------------------
        particle_id = int(e.get("id", e.get("idx", index)))

        if particle_id > skip_points:

            segment = ellipse_points(
                x0,
                y0,
                ax_e,
                by_e,
                theta=theta
            )

            alpha_segments.append(segment)

            # Area of the intersection ellipse
            alpha_areas.append(np.pi * ax_e * by_e)

    if not ellipse_geometries:
        raise RuntimeError(
            f"Keine Ellipsoide schneiden die Ebene z = {z_plane:.3f}"
        )

    # ------------------------------------------------------------------
    # Alpha-shape input point cloud
    # ------------------------------------------------------------------
    alpha_xy = (
        np.vstack(alpha_segments)
        if alpha_segments
        else np.empty((0, 2))
    )

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    suffix = "_upperEdge" if upper_edge else ""

    if output_path is not None:
        out_path = pathlib.Path(output_path)
    else:
        out_path = (
            pathlib.Path(RESULTS_DIR)
            / f"alphashape_slice_{z_plane:.2f}{suffix}.svg"
        )

    if upper_edge and out_path.suffix.lower() == ".svg":
        stem = out_path.stem

        if not stem.endswith("_upperEdge"):
            out_path = out_path.with_name(
                f"{stem}_upperEdge{out_path.suffix}"
            )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 4))

    area = float(sum(ellipse_areas))

    # ------------------------------------------------------------------
    # Alpha shape
    # ------------------------------------------------------------------
    if alphashape and alpha_xy.shape[0] >= 5:

        print(
            f"Using {alpha_xy.shape[0]} points to compute alpha-shape"
        )

        shape_coords, alpha_area = compute_alpha_shape_2d(
            alpha_xy,
            alpha=alpha,
            densify_max_segment_length=densify_max_segment_length
        )

        if shape_coords is not None:
            print(
                f"Alpha-shape vertices after "
                f"densify={densify_max_segment_length}: "
                f"{len(shape_coords)}"
            )

        if shape_coords is not None and shape_coords.size:

            ax.fill(
                shape_coords[:, 0],
                shape_coords[:, 1],
                color="tab:blue",
                alpha=0.4,
                label="Alpha-shape"
            )

            ax.plot(
                shape_coords[:, 0],
                shape_coords[:, 1],
                linewidth=2,
                color="tab:blue"
            )

            area = float(alpha_area)

            # ----------------------------------------------------------
            # Upper edge
            # ----------------------------------------------------------
            if upper_edge:

                x_edge, y_edge = _extract_upper_edge(
                    shape_coords,
                    n_bins=upper_edge_bins
                )

                if x_edge is not None and y_edge is not None:

                    print(
                        f"Upper-edge: bins={upper_edge_bins}, "
                        f"extracted_points={len(x_edge)}"
                    )

                    ax.plot(
                        x_edge,
                        y_edge,
                        color="red",
                        linewidth=2,
                        linestyle="-",
                        label="Oberkante"
                    )

    # ------------------------------------------------------------------
    # Draw all ellipses
    #
    # IMPORTANT:
    # No ellipse_points() here.
    #
    # The ellipse is reconstructed directly from:
    #     x0, y0, ax, by, theta
    #
    # The drawing resolution is therefore completely independent of
    # the alpha-shape point resolution.
    # ------------------------------------------------------------------
    t_plot = np.linspace(0.0, 2.0 * np.pi, 100)

    cos_t_plot = np.cos(t_plot)
    sin_t_plot = np.sin(t_plot)

    for x0, y0, ax_e, by_e, theta in ellipse_geometries:

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_local = ax_e * cos_t_plot
        y_local = by_e * sin_t_plot

        x = (
            x0
            + cos_theta * x_local
            - sin_theta * y_local
        )

        y = (
            y0
            + sin_theta * x_local
            + cos_theta * y_local
        )

        ax.plot(
            x,
            y,
            color="black",
            linewidth=1.0
        )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------
    ax.set_xlabel(r"x/$D_{\mathrm{Eq}}$")
    ax.set_ylabel(r"y/$D_{\mathrm{Eq}}$")

    ax.set_xticks(
        np.arange(
            xlim[0],
            xlim[1] + 5,
            5
        )
    )

    ax.set_yticks(
        np.arange(
            ylim[0],
            ylim[1] + 2.5,
            2.5
        )
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.grid(True)

    if alphashape:
        ax.legend(fontsize=10)

    plt.tight_layout()

    saved = save_figure(
        fig,
        out_path,
        default_name=f"alphashape_slice_{z_plane:.2f}{suffix}.svg"
    )

    return {
        "output_path": str(saved),
        "area": area,
        "num_ellipses": len(ellipse_geometries),
        "num_alpha_points": len(alpha_xy),
    }