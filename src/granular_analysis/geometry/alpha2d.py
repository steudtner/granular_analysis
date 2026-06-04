import numpy as np
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import ConvexHull


def compute_alpha_shape_2d(points, alpha):
    """Compute a 2D alpha-shape from `points`.

    Returns (coords, area) where `coords` is a Nx2 numpy array of the
    exterior polygon coordinates and `area` is the polygon area. If the
    alpha-shape cannot be computed, falls back to the convex hull.
    """
    pts = np.asarray(points)
    if pts.size == 0 or pts.shape[0] < 3:
        return None, 0.0

    try:
        shape = alphashape.alphashape(pts, alpha)

        # If shapely Polygon, return exterior coords
        if isinstance(shape, Polygon):
            return np.array(shape.exterior.coords), float(shape.area)

        # If MultiPolygon or GeometryCollection, pick largest polygon
        if isinstance(shape, MultiPolygon):
            largest = max(shape.geoms, key=lambda g: g.area)
            return np.array(largest.exterior.coords), float(largest.area)

    except Exception as e:
        # Don't crash — fall through to convex hull fallback
        try:
            msg = str(e)
            # keep light debug print
            print("alpha2d: alphashape failed:", msg)
        except Exception:
            pass

    # Fallback: convex hull of input points
    try:
        pts2 = pts[:, :2]
        if len(pts2) < 3:
            return None, 0.0
        hull = ConvexHull(pts2)
        hull_pts = pts2[hull.vertices]
        # Ensure closed polygon by repeating first point
        coords = np.vstack([hull_pts, hull_pts[0]])
        # shoelace area
        area = 0.5 * abs(np.sum(coords[:-1, 0] * coords[1:, 1] - coords[1:, 0] * coords[:-1, 1]))
        return coords, float(area)
    except Exception:
        return None, 0.0