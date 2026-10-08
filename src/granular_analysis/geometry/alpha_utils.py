"""Shared geometry helpers used across alpha-shape variants."""
import numpy as np
from shapely.geometry import MultiPolygon
import numpy as np
from scipy.spatial import ConvexHull


def convex_hull_coords(points: np.ndarray) -> tuple:
    """Return closed polygon coordinates and area computed from ConvexHull.

    Returns (coords, area). If not computable returns (None, 0.0).
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None, 0.0
    try:
        pts2 = pts[:, :2]
        hull = ConvexHull(pts2)
        hull_pts = pts2[hull.vertices]
        coords = np.vstack([hull_pts, hull_pts[0]])
        area = 0.5 * abs(np.sum(coords[:-1, 0] * coords[1:, 1] - coords[1:, 0] * coords[:-1, 1]))
        return coords, float(area)
    except Exception:
        return None, 0.0


def densify_polygon(coords: np.ndarray, max_segment_length: float) -> np.ndarray:
    """Linearly interpolate polygon edges so no segment exceeds max length.

    - *coords* is an (N,2) array. The function ensures the polygon is closed
      (first point repeated at end) in the output.
    - *max_segment_length* > 0 indicates the maximum allowed distance between
      consecutive vertices; longer edges are subdivided uniformly.

    Returns a new (M,2) array with the densified, closed polygon coordinates.
    """
    if max_segment_length is None or max_segment_length <= 0:
        return np.asarray(coords)

    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 2:
        return pts

    # Ensure closed polygon
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    out_points = []
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        seg = p2 - p1
        d = np.hypot(seg[0], seg[1])
        if d <= max_segment_length or d == 0.0:
            out_points.append(p1)
            continue
        # number of subdivisions so that each subsegment <= max_segment_length
        n_sub = int(np.ceil(d / max_segment_length))
        # generate n_sub points including p1 but excluding p2
        for k in range(n_sub):
            t = k / n_sub
            out_points.append(p1 + t * seg)

    # append final closing point (same as first)
    out_points.append(out_points[0])
    return np.asarray(out_points)


def _largest_polygon(shape):
    """Return the polygon with the largest area from a (Multi)Polygon."""
    if isinstance(shape, MultiPolygon):
        return max(shape.geoms, key=lambda g: g.area)
    return shape
