import warnings

import numpy as np
from typing import Optional
import alphashape
from shapely.geometry import Polygon

from granular_analysis.geometry.alpha_utils import _largest_polygon, convex_hull_coords, densify_polygon
from granular_analysis.geometry.alpha_logging import summarize_alpha_warnings


@summarize_alpha_warnings()
def compute_alpha_shape_2d(points, alpha=None, densify_max_segment_length: Optional[float] = None, max_points: int = 8000):
    """Compute a robust 2D alpha-shape from `points`.

    - *points*: array-like of shape (N,2) or (N,3)
    - *alpha*: optional float; if None the function will attempt to auto-select
      a reasonable alpha using `alphashape.optimizealpha` (if available).

    Returns (coords, area). On failure returns (None, 0.0).
    """
    pts = np.asarray(points)

    if pts.size == 0 or pts.shape[0] < 3:
        return None, 0.0

    # reduce to XY if 3D points were given
    if pts.ndim == 2 and pts.shape[1] >= 3:
        pts2 = pts[:, :2]
    else:
        pts2 = pts

    # If there are too many points, downsample deterministically to keep
    # the Delaunay/alphashape computation bounded in time and memory.
    if pts2.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts2.shape[0], size=max_points, replace=False)
        pts2_sub = pts2[idx]
    else:
        pts2_sub = pts2

    # Try to compute alpha automatically if not provided
    try:
        if alpha is None:
            try:
                alpha = alphashape.optimizealpha(pts2_sub)
            except Exception:
                alpha = 0.6

        shape = alphashape.alphashape(pts2_sub, alpha)

        if shape is None:
            raise RuntimeError("alphashape returned None")

        if isinstance(shape, Polygon):
            coords = np.array(shape.exterior.coords)
            area = float(shape.area)
        else:
            largest = _largest_polygon(shape)
            coords = np.array(largest.exterior.coords)
            area = float(largest.area)

        if densify_max_segment_length is not None and densify_max_segment_length > 0:
            coords = densify_polygon(coords, densify_max_segment_length)

        return coords, area

    except Exception as exc:
        warnings.warn(
            f"Alpha-shape calculation failed; using the convex hull instead: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Fallback: convex hull
    coords, area = convex_hull_coords(pts2)
    if densify_max_segment_length is not None and densify_max_segment_length > 0 and coords is not None:
        coords = densify_polygon(coords, densify_max_segment_length)
    return coords, area
