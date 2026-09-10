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
) -> pv.PolyData:

    unit_mesh = _unit_ellipsoid_mesh(
        nu=nu,
        nv=nv,
    )

    all_meshes = []

    for e in ellipsoids:

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

        projected_height = max(
            extent_y,
            extent_z,
        )

    plotter.camera.parallel_scale = (
        1.10 * projected_height
    )


# ============================================================
# DOMAIN BOX
# ============================================================

def _add_domain_box(
    plotter: pv.Plotter,
    bounds,
):

    domain = pv.Box(
        bounds=bounds,
    )

    plotter.add_mesh(
        domain,
        style="wireframe",
        color="black",
        line_width=1.0,
        opacity=0.35,
    )


# ============================================================
# SINGLE VIEW
# ============================================================

def render_particle_view(
    mesh: pv.PolyData,
    output_path: pathlib.Path,
    view: str,
    *,
    domain_bounds=RENDER_BOUNDS,
    window_size: Tuple[int, int] = (
        5000,
        3500,
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

    plotter.add_mesh(
        mesh,
        color="gray",
        smooth_shading=True,
        specular=0.15,
        ambient=0.25,
    )

    # --------------------------------------------------------
    # Domain box
    # --------------------------------------------------------

    _add_domain_box(
        plotter,
        bounds,
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    _set_camera(
        plotter,
        view,
        center,
        distance,
        bounds,
    )

    # --------------------------------------------------------
    # Render directly to final PNG
    # --------------------------------------------------------

    plotter.screenshot(
        str(output_path)
    )

    plotter.close()


# ============================================================
# ALL VIEWS
# ============================================================

def render_particle_views(
    ellipsoids,
    output_dir: pathlib.Path,
    *,
    domain_bounds=RENDER_BOUNDS,
    nu: int = 48,
    nv: int = 32,
    window_size: Tuple[int, int] = (
        5000,
        3500,
    ),
    views: Optional[Tuple[str, ...]] = None,
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
            / f"particles_3d_{view}.png"
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
