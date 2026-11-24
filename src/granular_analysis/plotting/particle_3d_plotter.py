from __future__ import annotations

import os

# ============================================================
# HEADLESS / HPC
# ============================================================

os.environ.setdefault(
    "PYVISTA_OFF_SCREEN",
    "true",
)

os.environ.setdefault(
    "VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN",
    "1",
)

import pathlib
from typing import Optional, Tuple

import numpy as np
import pyvista as pv

from granular_analysis.config import RENDER_BOUNDS
from granular_analysis.plotting import plotstyle

from granular_analysis.geometry.ellipsoid import _rotation_matrix


# ============================================================
# CAMERA DEFINITIONS
# ============================================================

CAMERA_DIRECTIONS = {
    "top": np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    ),

    "front": np.array(
        [1.0, 1.0, 0.0],
        dtype=float,
    ),

    "side": np.array(
        [0.0, 0.0, 1.0],
        dtype=float,
    ),
}


CAMERA_UP = {
    "top": np.array(
        [0.0, 0.0, -1.0],
        dtype=float,
    ),

    "front": np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    ),

    "side": np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    ),

    "isometric": np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    ),
}


# ============================================================
# ELLIPSOID MESH
# ============================================================

def _unit_ellipsoid_mesh(
    nu: int = 48,
    nv: int = 32,
) -> pv.PolyData:

    u = np.linspace(
        0.0,
        2.0 * np.pi,
        nu,
        endpoint=False,
    )

    v = np.linspace(
        0.0,
        np.pi,
        nv,
    )

    uu, vv = np.meshgrid(
        u,
        v,
        indexing="ij",
    )

    x = np.sin(vv) * np.cos(uu)
    y = np.sin(vv) * np.sin(uu)
    z = np.cos(vv)

    points = np.column_stack(
        (
            x.ravel(),
            y.ravel(),
            z.ravel(),
        )
    )

    faces = []

    for i in range(nu):

        i_next = (
            i + 1
        ) % nu

        for j in range(nv - 1):

            p0 = i * nv + j
            p1 = i_next * nv + j
            p2 = i_next * nv + (j + 1)
            p3 = i * nv + (j + 1)

            faces.append(
                [
                    4,
                    p0,
                    p1,
                    p2,
                    p3,
                ]
            )

    faces = np.asarray(
        faces,
        dtype=np.int64,
    ).ravel()

    return pv.PolyData(
        points,
        faces,
    )


# ============================================================
# PARTICLES -> POLYDATA
# ============================================================

def ellipsoids_to_polydata(
    ellipsoids,
    nu: int = 48,
    nv: int = 32,
    mobile_start: Optional[int] = None,
) -> pv.PolyData:

    unit_mesh = _unit_ellipsoid_mesh(
        nu=nu,
        nv=nv,
    )

    all_meshes = []

    for particle_index, e in enumerate(ellipsoids):

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        center = np.asarray(
            e["x"],
            dtype=float,
        )

        if center.shape != (3,):
            raise ValueError(
                "Particle position e['x'] must contain "
                "three coordinates."
            )

        # ----------------------------------------------------
        # Semi-axes
        # ----------------------------------------------------

        a = float(e["a"])
        b = float(e["b"])
        c = float(e["c"])

        if (
            not np.isfinite(a)
            or not np.isfinite(b)
            or not np.isfinite(c)
            or a <= 0.0
            or b <= 0.0
            or c <= 0.0
        ):
            continue

        # ----------------------------------------------------
        # Scale in particle-local coordinates
        # ----------------------------------------------------

        points = (
            unit_mesh.points
            * np.array(
                [a, b, c],
                dtype=float,
            )
        )

        # ----------------------------------------------------
        # Apply particle orientation
        #
        # q is scalar-first:
        # [w, x, y, z]
        #
        # Same rotation convention as
        # geometry.ellipsoid.ellipsoid_surface_points().
        # ----------------------------------------------------

        R = _rotation_matrix(e)

        points = points @ R.T

        # ----------------------------------------------------
        # Translation
        # ----------------------------------------------------

        points += center

        # ----------------------------------------------------
        # Create particle mesh
        # ----------------------------------------------------

        particle_mesh = pv.PolyData(
            points,
            unit_mesh.faces,
        )

        if mobile_start is not None:
            color = (220, 30, 30) if particle_index >= mobile_start else (128, 128, 128)
            particle_mesh.cell_data['particle_rgb'] = np.tile(
                np.asarray(color, dtype=np.uint8), (particle_mesh.n_cells, 1))

        all_meshes.append(
            particle_mesh
        )

    # --------------------------------------------------------
    # No valid particles
    # --------------------------------------------------------

    if not all_meshes:
        return pv.PolyData()

    # --------------------------------------------------------
    # Merge all particles
    # --------------------------------------------------------

    return pv.merge(
        all_meshes,
        merge_points=False,
    )

# ============================================================
# CAMERA
# ============================================================

def _set_camera(
    plotter: pv.Plotter,
    view: str,
    center: np.ndarray,
    distance: float,
    bounds: Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    elevation=None,
):

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if view == "isometric":

        direction = np.array(
            [1.0, 1.0, 1.0],
            dtype=float,
        )

    elif view in CAMERA_DIRECTIONS:

        direction = CAMERA_DIRECTIONS[
            view
        ].copy()

    else:

        raise ValueError(
            f"Unknown camera view: {view}"
        )

    if elevation is not None and view == "isometric":
        if not np.isfinite(elevation) or not 0 <= elevation < 90:
            raise ValueError("Camera elevation must be finite and in [0, 90)")
        angle = np.radians(elevation)
        direction = np.array([np.cos(angle) / np.sqrt(2), np.sin(angle),
                              np.cos(angle) / np.sqrt(2)])

    direction /= np.linalg.norm(
        direction
    )

    # --------------------------------------------------------
    # Camera position
    # --------------------------------------------------------

    position = (
        center
        + direction * distance
    )

    up = CAMERA_UP.get(
        view,
        np.array(
            [0.0, 1.0, 0.0],
            dtype=float,
        ),
    )

    plotter.camera_position = [
        position,
        center,
        up,
    ]

    plotter.camera.parallel_projection = True

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    extent_x = x_max - x_min
    extent_y = y_max - y_min
    extent_z = z_max - z_min

    # --------------------------------------------------------
    # Camera framing
    # --------------------------------------------------------

    if view == "top":

        projected_height = max(
            extent_z,
            extent_x
            / plotter.window_size[0]
            * plotter.window_size[1],
        )

    elif view == "front":

        projected_height = max(
            extent_y,
            extent_z,
        )

    elif view == "side":

        projected_height = max(
            extent_y,
            extent_x
            / plotter.window_size[0]
            * plotter.window_size[1],
        )

    else:
        # Project all box corners onto the camera axes, including the long
        # collapse direction and the viewport aspect ratio.
        right = np.cross(direction, up)
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, direction)
        corners = np.array([
            [x, y, z]
            for x in (x_min, x_max)
            for y in (y_min, y_max)
            for z in (z_min, z_max)
        ]) - center
        width = np.ptp(corners @ right)
        height = np.ptp(corners @ camera_up)
        viewport = plotter.renderer.GetViewport()
        aspect = (plotter.window_size[0] * (viewport[2] - viewport[0])
                  / (plotter.window_size[1] * (viewport[3] - viewport[1])))
        plotter.camera.parallel_scale = 0.55 * max(height, width / aspect)
        plotter.reset_camera_clipping_range()
        return

    plotter.camera.parallel_scale = 1.10 * projected_height
    plotter.reset_camera_clipping_range()


# ============================================================
# SINGLE VIEW
# ============================================================

def _vortex_log_colors(values, component, clim=None):
    """Positive vortex strength and strictly positive limits for log colors."""
    strength = np.asarray(values, dtype=float) * (-1 if component == "lambda2" else 1)
    strength = np.where(np.isfinite(strength) & (strength > 0), strength, np.nan)
    finite = strength[np.isfinite(strength)]
    if clim is None:
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (1., 10.)
        if low == high:
            low = max(low / 10, np.finfo(float).tiny)
            high = max(high, low * 10)
    else:
        low, high = sorted(abs(float(value)) for value in clim)
        if not np.isfinite([low, high]).all() or low <= 0 or low >= high:
            raise ValueError("Log vortex color limits require distinct positive strengths")
    return strength, (low, high)


def render_particle_view(
    mesh: pv.PolyData,
    output_path: pathlib.Path,
    view: str,
    *,
    domain_bounds=RENDER_BOUNDS,
    fluid=None,
    fluid_z=None,
    fit_axes=(True, True, True),
    fluid_clim=None,
    export_document=True,
    time_label=None,
    legend_scale=1.0,
    camera_elevation=None,
    window_size: Tuple[int, int] = (
        8000,
        5600,
    ),
):

    bounds = tuple(
        float(v)
        for v in domain_bounds
    )

    if len(bounds) != 6:

        raise ValueError(
            "domain_bounds must contain exactly "
            "six values: "
            "(x_min, x_max, y_min, y_max, z_min, z_max)."
        )

    if mesh.n_points:
        # Include full particle surfaces, even outside the configured domain.
        bounds = tuple(
            (min(bounds[i], mesh.bounds[i]) if i % 2 == 0
             else max(bounds[i], mesh.bounds[i])) if fit_axes[i // 2] else bounds[i]
            for i in range(6)
        )

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # --------------------------------------------------------
    # Domain center
    # --------------------------------------------------------

    center = np.array(
        [
            0.5 * (x_min + x_max),
            0.5 * (y_min + y_max),
            0.5 * (z_min + z_max),
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Domain extent
    # --------------------------------------------------------

    extent = np.array(
        [
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        ],
        dtype=float,
    )

    max_extent = float(
        np.max(extent)
    )

    distance = (
        2.0 * max_extent
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_path = pathlib.Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # PyVista
    # --------------------------------------------------------

    plotter = pv.Plotter(
        off_screen=True,
        window_size=window_size,
    )

    plotter.set_background(
        "white"
    )

    # --------------------------------------------------------
    # Complete particles
    # --------------------------------------------------------

    if mesh.n_points and not all(fit_axes) and any(
        mesh.bounds[i] < bounds[i] if i % 2 == 0 else mesh.bounds[i] > bounds[i]
        for i in range(6)
    ):
        mesh = mesh.clip_box(bounds=bounds, invert=False)

    plotter.add_mesh(
        mesh,
        **({'scalars': 'particle_rgb', 'rgb': True, 'preference': 'cell',
            'show_scalar_bar': False} if 'particle_rgb' in mesh.cell_data else {'color': 'gray'}),
        smooth_shading=True,
        specular=0.15,
        ambient=0.25,
        opacity=1.0,
    )

    if fluid is not None:
        z_plane = float(fluid_z if fluid_z is not None else (z_min + z_max) / 2)
        is_volume = "z_edges" in fluid and np.ndim(fluid["values"]) == 3
        if is_volume:
            grid = pv.RectilinearGrid(fluid["x_edges"], fluid["y_edges"], fluid["z_edges"])
        else:
            xx, yy = np.meshgrid(fluid["x_edges"], fluid["y_edges"], indexing="ij")
            grid = pv.StructuredGrid(xx, yy, np.full_like(xx, z_plane))
        values = np.asarray(fluid["values"], dtype=float)
        component = fluid["component"]
        if component in {"magnitude", "u", "v", "w"}:
            values = values / plotstyle.VELOCITY_SCALE
        elif component == "q":
            values = values / plotstyle.Q_SCALE
        log_colors = component in {"q", "lambda2"}
        if log_colors:
            if fluid_clim is None and fluid.get('vortex_reference_value', 0) > 0:
                upper = fluid['vortex_reference_value'] / (plotstyle.Q_SCALE if component == 'q' else 1)
                lower = fluid['vortex_threshold_fraction'] * upper
                if 0 < lower < upper:
                    fluid_clim = (lower, upper)
            values, vortex_clim = _vortex_log_colors(values, component, fluid_clim)
        color_title = "-lambda2" if component == "lambda2" else component
        color_map = "viridis" if log_colors else (plotstyle.FLUID_MAGNITUDE_CMAP if component == "magnitude" else "RdBu_r")
        grid.cell_data[component] = values.ravel(order="C") if is_volume else values.T.ravel(order="F")
        if is_volume and np.isfinite(grid.cell_data[component]).any():
            grid = grid.extract_cells(np.isfinite(grid.cell_data[component]))
        finite = values[np.isfinite(values)]
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
        limit = max(limit, np.finfo(float).eps)
        clim = tuple(fluid_clim) if fluid_clim is not None else ((0, limit) if component == "magnitude" else (-limit, limit))
        if log_colors:
            clim = vortex_clim
        # Reserve a separate lower strip for the horizontal colour legend.
        # Keep the camera framing tied to the remaining scene viewport.
        if view in {"top", "front"}:
            plotter.renderer.SetViewport(0.0, 0.16, 1.0, 1.0)
        if not all(fit_axes):
            clip_bounds = tuple(
                (min(bounds[i], grid.bounds[i]) if i % 2 == 0
                 else max(bounds[i], grid.bounds[i])) if fit_axes[i // 2] else bounds[i]
                for i in range(6)
            )
            if any(grid.bounds[i] < clip_bounds[i] if i % 2 == 0 else grid.bounds[i] > clip_bounds[i]
                   for i in range(6)):
                grid = grid.clip_box(bounds=clip_bounds, invert=False)
        plotter.add_mesh(
            grid, scalars=component, preference="cell", lighting=False, nan_opacity=0.0,
            cmap=color_map, log_scale=log_colors,
            clim=clim,
            scalar_bar_args={
                "title": color_title + (" / (g D_eq)^(1/2)" if component in {"magnitude", "u", "v", "w"} else " / (rho g D_eq)" if component == "pressure" else " / (g / D_eq)" if component == "q" else ""),
                "vertical": False,
                "position_x": 0.20,
                "position_y": 0.10,
                "width": 0.60,
                "height": 0.80,
                "color": "black",
                "font_family": "times",
                "title_font_size": max(12, int(window_size[1] * 0.018 * legend_scale)),
                "label_font_size": max(10, int(window_size[1] * 0.014 * legend_scale)),
                "unconstrained_font_size": True,
                "n_labels": 5,
                "fmt": "%.2g",
            },
        )
        if not is_volume and view in {"top", "front"} and np.isfinite(grid.cell_data[component]).any():
            # These cameras see the x-y plane edge-on. Draw coloured cell
            # edges as well so its location remains visible in projection.
            plotter.add_mesh(
                grid.extract_cells(np.isfinite(grid.cell_data[component])).extract_all_edges(), scalars=component, preference="cell",
                lighting=False, line_width=max(2, window_size[0] / 500),
                cmap=color_map, log_scale=log_colors,
                clim=clim,
                show_scalar_bar=False,
            )
        scalar_bar = next(iter(plotter.scalar_bars.values()))
        plotter.renderer.RemoveActor2D(scalar_bar)
        if view in {"top", "front"}:
            # A coplanar legend would be unreadable in these edge-on views.
            from vtkmodules.vtkRenderingCore import vtkRenderer
            legend_renderer = vtkRenderer()
            legend_renderer.SetViewport(0.0, 0.0, 1.0, 0.16)
            legend_renderer.SetBackground(1.0, 1.0, 1.0)
            legend_renderer.InteractiveOff()
            legend_renderer.AddActor2D(scalar_bar)
            plotter.render_window.AddRenderer(legend_renderer)
        elif grid.n_points:
            # Place the colour strip and text in the very same x-y plane,
            # just above its visible upper edge, so the camera rotates both.
            gx0, gx1, gy0, gy1, _, _ = bounds if is_volume else grid.bounds
            strip_width = (gx1 - gx0) * 0.65
            text_height = strip_width * 0.025 * legend_scale
            strip_height = text_height * 0.75
            strip_x = (gx0 + gx1 - strip_width) / 2
            strip_y = gy1 + text_height * 2.0
            xx, yy = np.meshgrid(
                np.linspace(strip_x, strip_x + strip_width, 257),
                [strip_y, strip_y + strip_height], indexing="ij",
            )
            legend = pv.StructuredGrid(xx, yy, np.full_like(xx, z_plane))
            legend.cell_data[component] = (np.geomspace if log_colors else np.linspace)(*clim, 256)
            plotter.add_mesh(
                legend, scalars=component, preference="cell", lighting=False,
                cmap=color_map, log_scale=log_colors,
                clim=clim, show_scalar_bar=False,
            )
            legend_meshes = [legend]
            title = scalar_bar.GetTitle()
            for label, x, y, height in [
                (title, (gx0 + gx1) / 2, strip_y + strip_height + text_height * 1.3, text_height),
                *[(f"{value:.2g}", strip_x + strip_width * i / 4,
                   strip_y - text_height * 0.8, text_height * 0.8)
                  for i, value in enumerate((np.geomspace if log_colors else np.linspace)(*clim, 5))],
            ]:
                text = pv.Text3D(label, depth=0, height=height,
                                 center=(x, y, z_plane), normal=(0, 0, 1))
                plotter.add_mesh(text, color="black", lighting=False)
                legend_meshes.append(text)
            # The data still obey explicit limits; leave room for annotations.
            for annotation in legend_meshes:
                bounds = tuple(min(bounds[i], annotation.bounds[i]) if i % 2 == 0
                               else max(bounds[i], annotation.bounds[i]) for i in range(6))
        if grid.n_points:
            bounds = tuple(
                (min(bounds[i], grid.bounds[i]) if i % 2 == 0
                 else max(bounds[i], grid.bounds[i])) if fit_axes[i // 2] else bounds[i]
                for i in range(6)
            )
        center = np.array([(bounds[i] + bounds[i + 1]) / 2 for i in (0, 2, 4)])

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    _set_camera(
        plotter,
        view,
        center,
        distance,
        bounds,
        **({"elevation": camera_elevation} if camera_elevation is not None else {}),
    )

    # --------------------------------------------------------
    # Render directly to final PNG
    # --------------------------------------------------------

    if time_label is not None:
        plotter.add_text(time_label, position="upper_left", color="black",
                         font_size=max(12, window_size[1] // 50))
    image = plotter.screenshot(str(output_path))
    if view == 'top':
        import json
        half_height = float(plotter.camera.parallel_scale)
        half_width = half_height * image.shape[1] / image.shape[0]
        output_path.with_suffix('.projection.json').write_text(json.dumps({
            'view': view,
            'extent': [float(center[0]-half_width), float(center[0]+half_width),
                       float(center[2]+half_height), float(center[2]-half_height)],
            'particle_bounds': list(mesh.bounds),
        }) + '\n')
    plotter.close()
    if not export_document:
        return image

    # A shaded 3D screenshot remains raster content inside the PDF/SVG pair.
    # Preserve its aspect ratio at the final 160-mm print width.
    import matplotlib.pyplot as plt
    height = plotstyle.figwidth_full * image.shape[0] / image.shape[1]
    fig = plt.figure(figsize=(plotstyle.figwidth_full, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, interpolation="none", resample=False)
    ax.set_axis_off()
    try:
        plotstyle.export_figure(
            fig, output_path.with_suffix('.pdf'),
            dpi=image.shape[1] / plotstyle.figwidth_full,
        )
    finally:
        plt.close(fig)


# ============================================================
# ALL VIEWS
# ============================================================

def render_particle_views(
    ellipsoids,
    output_dir: pathlib.Path,
    *,
    domain_bounds=RENDER_BOUNDS,
    fluid=None,
    fluid_z=None,
    fit_axes=(True, True, True),
    nu: int = 48,
    nv: int = 32,
    window_size: Tuple[int, int] = (
        8000,
        5600,
    ),
    views: Optional[Tuple[str, ...]] = None,
    filename_prefix: str = "particles_3d",
    mobile_start: Optional[int] = None,
):
    """
    Render all requested 3D views.
    """

    if views is None:

        views = (
            "isometric",
            "top",
            "front",
            "side",
        )

    output_dir = pathlib.Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bounds = tuple(
        float(v)
        for v in domain_bounds
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    print(
        "============================================================"
    )

    print(
        "3D PARTICLE RENDERING"
    )

    print(
        "============================================================"
    )

    print(
        "Render bounds:"
    )

    print(
        f"  x = {bounds[0]} ... {bounds[1]}"
    )

    print(
        f"  y = {bounds[2]} ... {bounds[3]}"
    )

    print(
        f"  z = {bounds[4]} ... {bounds[5]}"
    )

    print()

    # ========================================================
    # BUILD PARTICLE MESH
    # ========================================================

    print(
        "Building complete 3D particle mesh..."
    )

    mesh = ellipsoids_to_polydata(
        ellipsoids,
        nu=nu,
        nv=nv,
        mobile_start=mobile_start,
    )

    print(
        f"  Points: {mesh.n_points}"
    )

    print(
        f"  Cells:  {mesh.n_cells}"
    )

    print()

    # ========================================================
    # RENDER VIEWS
    # ========================================================

    results = {}

    for view in views:

        output_path = (
            output_dir
            / f"{filename_prefix}_{view}.png"
        )

        print(
            f"Rendering {view:12s} -> "
            f"{output_path}"
        )

        render_particle_view(
            mesh,
            output_path,
            view,
            domain_bounds=bounds,
            fluid=fluid,
            fluid_z=fluid_z,
            fit_axes=fit_axes,
            window_size=window_size,
        )

        results[
            view
        ] = output_path

    print()

    print(
        "3D rendering completed."
    )

    return results
