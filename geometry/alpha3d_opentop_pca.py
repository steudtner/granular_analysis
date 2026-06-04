import numpy as np
import alphashape
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from sklearn.decomposition import PCA

from .alpha_utils import _largest_polygon, _pca_transform, _inverse_pca


# -----------------------------
# OPEN-TOP ALPHA SHAPE (original concept preserved)
# -----------------------------
def alpha_shape_opentop_pca(points, alpha=0.6, z_min=None, z_max=None):

    pts = np.asarray(points)

    if len(pts) < 20:
        return None, 0.0

    # -------------------------
    # Z constraint
    # -------------------------
    if z_min is not None:
        pts = pts[pts[:, 2] >= z_min]
    if z_max is not None:
        pts = pts[pts[:, 2] <= z_max]

    if len(pts) < 20:
        return None, 0.0

    # -------------------------
    # PCA alignment
    # -------------------------
    pts_pca, pca = _pca_transform(pts)

    # -------------------------
    # 2D alpha shape (XY footprint)
    # -------------------------
    shape = alphashape.alphashape(pts_pca[:, :2], alpha)
    shape = _largest_polygon(shape)

    if not isinstance(shape, Polygon):
        return None, 0.0

    base = np.array(shape.exterior.coords)

    z_bot = np.min(pts_pca[:, 2])
    z_top = np.max(pts_pca[:, 2])

    # -------------------------
    # OPEN PRISM (unchanged logic)
    # -------------------------
    vertices = []
    faces = []

    n = len(base)

    for i in range(n):
        x, y = base[i]
        vertices.append([x, y, z_bot])
        vertices.append([x, y, z_top])

    vertices = np.asarray(vertices)

    for i in range(n - 1):
        b0 = 2 * i
        b1 = 2 * i + 1
        b2 = 2 * (i + 1)
        b3 = 2 * (i + 1) + 1

        faces.append([b0, b2, b1])
        faces.append([b1, b2, b3])

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces),
        process=True
    )

    # back-transform
    mesh = _inverse_pca(mesh, pca)

    # -------------------------
    # volume remains heuristic (as before)
    # -------------------------
    volume = mesh.area * (z_top - z_bot)

    return mesh, volume