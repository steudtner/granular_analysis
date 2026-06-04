import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh

from granular_analysis.config import RESULTS_DIR, Z_LEVELS, FOLDER
from granular_analysis.io.loader import load_last_frame
from granular_analysis.geometry.ellipsoid import ellipsoid_surface_points
from granular_analysis.geometry.alpha3d_variants import (
    alpha_shape_smooth,
    alpha_shape_trimesh
)


# ----------------------------
# POINT CLOUD
# ----------------------------
def build_point_cloud():
    ell = load_last_frame(FOLDER)

    pts = np.vstack([
        ellipsoid_surface_points(e, 10, 10)
        for e in ell
    ])

    return pts


# ----------------------------
# PLOT
# ----------------------------
def plot_alpha_variants(points, alpha, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    shape = alpha_shape_smooth(points, alpha)
    mesh = alpha_shape_trimesh(points, alpha)

    fig = plt.figure(figsize=(12, 5))

    # ---- 2D boundary
    ax1 = fig.add_subplot(121)
    if shape is not None:
        x, y = shape.exterior.xy
        ax1.plot(x, y, linewidth=2)

    ax1.set_title("Smooth Alpha Shape (2D)")
    ax1.set_aspect("equal")

    # ---- 3D mesh
    ax2 = fig.add_subplot(122, projection="3d")

    if mesh is not None:
        v = mesh.vertices
        f = mesh.faces

        ax2.plot_trisurf(
            v[:, 0],
            v[:, 1],
            v[:, 2],
            triangles=f,
            alpha=0.6
        )

    ax2.set_title("Triangulated Alpha Shape (3D)")

    path = os.path.join(save_dir, "alpha_shape_variants.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")

    print(f"Saved: {path}")

    return fig


# ----------------------------
# MAIN
# ----------------------------
def main():
    from granular_analysis.utils.logging_utils import suppress_noise
    suppress_noise()

    pts = build_point_cloud()

    save_dir = os.path.join(RESULTS_DIR, "alpha_variants")
    plot_alpha_variants(pts, alpha=0.6, save_dir=save_dir)


if __name__ == "__main__":
    main()