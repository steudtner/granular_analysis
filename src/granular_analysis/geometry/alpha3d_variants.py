import numpy as np
import alphashape
import trimesh
from shapely.geometry import Polygon

from .alpha_utils import _largest_polygon


def alpha_shape_smooth(points, alpha):
    """Continuous / smooth alpha shape (Shapely-based)."""
    pts = np.asarray(points)
    if len(pts) < 10:
        return None
    shape = alphashape.alphashape(pts, alpha)
    shape = _largest_polygon(shape)
    return shape if isinstance(shape, Polygon) else None


def alpha_shape_trimesh(points, alpha):
    """Triangulated / mesh-based alpha shape (for 3D rendering)."""
    pts = np.asarray(points)
    if len(pts) < 10:
        return None
    shape = alphashape.alphashape(pts, alpha)
    shape = _largest_polygon(shape)
    if not isinstance(shape, Polygon):
        return None
    vertices = np.array(shape.exterior.coords)
    return trimesh.Trimesh(vertices=vertices, process=True)
