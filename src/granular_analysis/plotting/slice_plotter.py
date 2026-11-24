"""Exact 2D sections through ellipsoid frames."""

from __future__ import annotations

from granular_analysis.plotting import plotstyle

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, SymLogNorm
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from granular_analysis.analysis.porosity2d import upper_edge_points
from granular_analysis.config import RESULTS_DIR
from granular_analysis.geometry.alpha2d import compute_alpha_shape_2d
from granular_analysis.geometry.ellipsoid import ellipse_points, ellipsoid_slice
from granular_analysis.plotting.helpers import PLOT_STYLE, save_figure


@matplotlib.rc_context(PLOT_STYLE)
def plot_exact_slice_from_ellipsoids(
    ellipsoids,
    z_plane: float,
    *,
    output_path: Optional[Union[str, Path]] = None,
    alphashape: bool = False,
    alpha: float = 0.6,
    upper_edge: bool = False,
    upper_edge_bins: int = 50,
    densify_max_segment_length: Optional[float] = None,
    skip_points: int = 0,
    xlim: Tuple[float, float] = (0, 60),
    ylim: Tuple[float, float] = (0, 20),
    periodic_length: Optional[float] = None,
    fluid=None,
    fluid_grid: bool = False,
    velocity_arrows: bool = True,
    velocity_colors: Optional[bool] = None,
    velocity_directions: bool = False,
):
    """Plot exact sections, coloured white to red by full particle speed.

    ``velocity_arrows`` is a compatibility alias for enabling speed colours.
    ``velocity_colors`` overrides it when provided.
    Exports matching PNG, SVG and PDF files by default.
    """
    if fluid_grid and fluid is None:
        raise ValueError("fluid_grid requires a loaded fluid slice")
    color_velocity = velocity_arrows if velocity_colors is None else velocity_colors
    ellipse_geometries = []
    ellipse_velocities = []
    occupied_polygons = []
    alpha_segments = []
    shifts = (-periodic_length, 0.0, periodic_length) if periodic_length else (0.0,)

    for shift in shifts:
        for index, ellipsoid in enumerate(ellipsoids, start=1):
            section = ellipsoid_slice(ellipsoid, z_plane, shift)
            if section is None:
                continue
            x0, y0, axis_a, axis_b, theta = section
            ellipse_geometries.append((x0, y0, axis_a, axis_b, theta))
            velocity = np.asarray(ellipsoid.get("u", [0., 0., 0.]), dtype=float)
            ellipse_velocities.append(velocity if velocity.shape == (3,) else [np.nan] * 3)
            particle_id = int(ellipsoid.get("id", ellipsoid.get("idx", index)))
            if particle_id <= skip_points:
                continue
            segment = ellipse_points(x0, y0, axis_a, axis_b, theta=theta)
            alpha_segments.append(segment)
            occupied_polygons.append(Polygon(segment))

    if not ellipse_geometries:
        raise RuntimeError(f"No ellipsoids intersect z={z_plane:.3f}")

    alpha_xy = np.vstack(alpha_segments) if alpha_segments else np.empty((0, 2))
    occupied_area = (
        float(unary_union(occupied_polygons).area) if occupied_polygons else 0.0
    )
    suffix = "_upperEdge" if upper_edge else ""
    out_path = Path(output_path) if output_path else (
        Path(RESULTS_DIR) / f"alphashape_slice_{z_plane:.2f}{suffix}.svg"
    )
    if upper_edge and not out_path.stem.endswith("_upperEdge"):
        out_path = out_path.with_name(f"{out_path.stem}_upperEdge{out_path.suffix}")

    colorbar_count = int(fluid is not None) + int(color_velocity)
    legend_rows = int(alphashape)
    slice_height = max(2.3, plotstyle.figwidth_full * (ylim[1]-ylim[0])/(xlim[1]-xlim[0]))
    fig = plt.figure(figsize=(plotstyle.figwidth_full, slice_height + .5*colorbar_count + .35*legend_rows),
                     layout="constrained")
    fig.get_layout_engine().set(w_pad=.02, h_pad=.02, hspace=0, wspace=0)
    grid = fig.add_gridspec(1 + legend_rows + colorbar_count, 1,
                           height_ratios=[.4]*colorbar_count + [.6]*legend_rows + [5])
    ax = fig.add_subplot(grid[-1])
    colorbars = []
    if fluid is not None:
        values = np.asarray(fluid["values"])
        if fluid["component"] in {"magnitude", "u", "v", "w"}:
            values = values / plotstyle.VELOCITY_SCALE
        elif fluid["component"] == "q":
            values = values / plotstyle.Q_SCALE
        limits = {}
        if fluid["component"] == "magnitude":
            low, high = plotstyle.visible_fluid_range(
                values, fluid["x"], fluid["y"], (*xlim, *ylim))
            limits = {"vmin": low, "vmax": high}
        else:
            finite = np.abs(np.asarray(values)[np.isfinite(values)])
            limit = float(finite.max()) if finite.size else 0.0
            if fluid.get('vortex_reference_value', 0) > 0:
                limit = fluid['vortex_reference_value'] / (plotstyle.Q_SCALE if fluid['component'] == 'q' else 1)
            limit = limit if limit > 0 else 1.0
            limits = {"vmin": -limit, "vmax": limit}
            if fluid["component"] in {"lambda2", "q"}:
                # Preserve both signs, with a linear neighborhood around zero.
                limits = {"norm": SymLogNorm(linthresh=max(limit * 1e-3, np.finfo(float).tiny),
                                             vmin=-limit, vmax=limit, base=10)}
        mesh = ax.pcolormesh(fluid["x_edges"] if fluid_grid else fluid["x"],
                             fluid["y_edges"] if fluid_grid else fluid["y"],
                             values, shading="flat" if fluid_grid else "nearest",
                             cmap=plotstyle.FLUID_MAGNITUDE_CMAP if fluid["component"] == "magnitude" else "RdBu_r",
                             rasterized=True, zorder=0, **limits)
        colorbars.append((mesh, (plotstyle.velocity_label("f") if fluid["component"] == "magnitude"
                                        else r"$p/(\rho_w g D_{\mathrm{Eq}})$" if fluid["component"] == "pressure"
                                        else r"$\lambda_2$ [simulation time$^{-2}$]" if fluid["component"] == "lambda2"
                                        else plotstyle.Q_LABEL if fluid["component"] == "q"
                                        else plotstyle.velocity_label("f", fluid["component"]))))
        ax.set_title(f"t={fluid['time']:.8g}, fluid counter {fluid['counter']:06d}")
    if fluid_grid:
        plotstyle.draw_fluid_grid(ax, fluid, (*xlim, *ylim))
    reported_area = occupied_area
    if alphashape and len(alpha_xy) >= 3:
        shape, alpha_area = compute_alpha_shape_2d(
            alpha_xy,
            alpha=alpha,
            densify_max_segment_length=densify_max_segment_length,
        )
        if shape is not None and shape.size:
            if fluid is None:
                ax.fill(shape[:, 0], shape[:, 1], color="tab:blue", alpha=0.4, label="Alpha-shape")
            ax.plot(shape[:, 0], shape[:, 1], linewidth=1.2, color="tab:blue")
            reported_area = float(alpha_area)
            if upper_edge:
                edge_x, edge_y = upper_edge_points(shape, n_bins=upper_edge_bins)
                if edge_x is not None:
                    ax.plot(edge_x, edge_y, color="red", linewidth=1.2, label="Oberkante")

    speeds = np.linalg.norm(np.asarray(ellipse_velocities), axis=1) / plotstyle.VELOCITY_SCALE
    finite_speeds = speeds[np.isfinite(speeds)]
    max_speed = float(finite_speeds.max()) if finite_speeds.size else 0.0
    speed_bounds = np.linspace(0., max_speed if max_speed > 0 else 1.,
                               plotstyle.PARTICLE_SPEED_STEPS + 1)
    speed_cmap = LinearSegmentedColormap.from_list(
        "particle_speed", ["white", "red"], N=plotstyle.PARTICLE_SPEED_STEPS)
    speed_norm = BoundaryNorm(speed_bounds, speed_cmap.N, clip=True)
    if color_velocity:
        colorbars.append((matplotlib.cm.ScalarMappable(norm=speed_norm, cmap=speed_cmap),
                          plotstyle.velocity_label("p")))

    angles = np.linspace(0.0, 2.0 * np.pi, 100)
    for (x0, y0, axis_a, axis_b, theta), speed in zip(ellipse_geometries, speeds):
        x_local = axis_a * np.cos(angles)
        y_local = axis_b * np.sin(angles)
        cosine, sine = np.cos(theta), np.sin(theta)
        if fluid is not None or color_velocity:
            # Opaque exact sections hide the raster inside every particle,
            # including particles excluded from the alpha-shape calculation.
            ax.fill(x0 + cosine * x_local - sine * y_local,
                    y0 + sine * x_local + cosine * y_local,
                    facecolor=(speed_cmap(speed_norm(speed))
                               if color_velocity and np.isfinite(speed) else "white"),
                    edgecolor="none", zorder=3)
        ax.plot(
            x0 + cosine * x_local - sine * y_local,
            y0 + sine * x_local + cosine * y_local,
            color="black",
            linewidth=0.5,
            zorder=4,
        )

    if velocity_directions:
        velocities = np.asarray(ellipse_velocities)[:, :2]
        geometry = np.asarray(ellipse_geometries)
        in_plane_speed = np.linalg.norm(velocities, axis=1)
        moving = np.isfinite(velocities).all(axis=1) & (in_plane_speed > 0)
        if moving.any():
            # Unit directions; length follows particle size, not velocity magnitude.
            lengths = 1.4 * np.min(geometry[moving, 2:4], axis=1)
            directions = velocities[moving] / in_plane_speed[moving, None]
            vectors = directions * lengths[:, None]
            ax.quiver(geometry[moving, 0], geometry[moving, 1], vectors[:, 0], vectors[:, 1],
                      angles="xy", scale_units="xy", scale=1, pivot="middle",
                      color="black", edgecolor="black", linewidth=.2, width=.0025,
                      headwidth=3.5, headlength=4.5, minlength=0, zorder=5)

    ax.set_xlabel(r"$x / D_{\mathrm{Eq}}$")
    ax.set_ylabel(r"$y / D_{\mathrm{Eq}}$")
    tick_step = 2.5 if fluid_grid else 5
    ax.set_xticks(np.arange(np.ceil(xlim[0]/tick_step)*tick_step, xlim[1] + 1e-9, tick_step))
    ax.set_yticks(np.arange(np.ceil(ylim[0]/tick_step)*tick_step, ylim[1] + 1e-9, tick_step))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.grid(not fluid_grid)
    legend_axes = []
    if alphashape:
        legend_axis = fig.add_subplot(grid[colorbar_count])
        legend_axes.append(legend_axis)
        legend_axis.set_axis_off()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend_axis.legend(handles, labels, loc="center", ncol=len(handles), frameon=False)
    for index, (mappable, label) in enumerate(colorbars):
        colorbar_axis = fig.add_subplot(grid[index])
        legend_axes.append(colorbar_axis)
        bar = fig.colorbar(mappable, cax=colorbar_axis, orientation="horizontal", label=label)
        bar.ax.xaxis.set_ticks_position("top")
        bar.ax.xaxis.set_label_position("bottom")
    plotstyle.align_slice_legends(ax, legend_axes)

    # Request a raster export as well as the helper's automatic SVG/PDF pair.
    if out_path.suffix.lower() not in (".png", ".svg", ".pdf"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=600)
    save_figure(fig, out_path.with_suffix(".png"))
    saved = str(out_path)
    return {
        "output_path": saved,
        "area": reported_area,
        "occupied_area": occupied_area,
        "num_ellipses": len(ellipse_geometries),
        "num_alpha_points": len(alpha_xy),
    }
