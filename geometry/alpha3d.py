import numpy as np
import alphashape
import trimesh


def compute_alpha_shape_3d(points, alpha, include_origin=False):

    points = np.asarray(points, dtype=float)

    if include_origin and np.all(points >= 0.0):
        boundary_points = np.array([
            [0.0, 0.0, 0.0],
            [points[:, 0].max(), 0.0, 0.0],
            [0.0, points[:, 1].max(), 0.0],
            [0.0, 0.0, points[:, 2].max()]
        ])
        points = np.vstack([points, boundary_points])

    mesh = alphashape.alphashape(points, alpha)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("3D alpha failed")

    trimesh.repair.fix_normals(mesh)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)

    volume = mesh.volume
    if not np.isfinite(volume):
        raise RuntimeError("Invalid alpha shape volume")

    return mesh, abs(volume)