import numpy as np
import alphashape
import trimesh

from shapely.geometry import Polygon, MultiPolygon
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay

from .alpha_utils import _largest_polygon, _pca_transform, _inverse_pca


def alpha_shape_opentop_pca(points, alpha=0.6, z_min=None, z_max=None):

    pts = np.asarray(points)

    if len(pts) < 20:
        return None, 0.0

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
    # alpha shape footprint
    # -------------------------

    shape = alphashape.alphashape(
        pts_pca[:, :2],
        alpha
    )

    shape = _largest_polygon(shape)

    if not isinstance(shape, Polygon):
        return None, 0.0

    footprint = np.array(shape.exterior.coords)

    # -------------------------
    # top surface
    # -------------------------

    top_vertices = []

    for x, y in footprint:

        d = np.sum(
            (pts_pca[:, :2] - np.array([x, y])) ** 2,
            axis=1
        )

        idx = np.argmin(d)

        z = pts_pca[idx, 2]

        top_vertices.append([x, y, z])

    top_vertices = np.asarray(top_vertices)

    # -------------------------
    # bottom plane z=0
    # -------------------------

    bottom_vertices = top_vertices.copy()
    bottom_vertices[:, 2] = 0.0

    n = len(top_vertices)

    vertices = np.vstack([
        top_vertices,
        bottom_vertices
    ])

    faces = []

    # -------------------------
    # side walls
    # -------------------------

    for i in range(n - 1):

        t0 = i
        t1 = i + 1

        b0 = i + n
        b1 = i + 1 + n

        faces.append([t0, b0, t1])
        faces.append([t1, b0, b1])

    # -------------------------
    # top triangulation
    # -------------------------

    tri = Delaunay(top_vertices[:, :2])

    for simplex in tri.simplices:
        faces.append(simplex.tolist())

    # -------------------------
    # bottom triangulation
    # -------------------------

    for simplex in tri.simplices:
        faces.append(
            [
                simplex[2] + n,
                simplex[1] + n,
                simplex[0] + n,
            ]
        )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces),
        process=True
    )

    mesh = _inverse_pca(mesh, pca)

    volume = abs(mesh.volume)

    return mesh, volume