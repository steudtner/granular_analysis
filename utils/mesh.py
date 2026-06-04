"""Mesh utilities for slicing and area extraction.

Centralizes robust section extraction and area computation used
by visualization and comparison scripts.
"""
import numpy as np
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union
from scipy.spatial import ConvexHull


def section_area_from_mesh(section):
    """Compute 2D area from a trimesh section object.

    This function accepts either a precomputed `section` (returned by
    `mesh.section(...)`) or a tuple-like (path2d, transform) as returned
    by some trimesh helpers. It returns a float area (0.0 if nothing).
    """
    # Allow passing through a mesh and z-level externally by callers
    # but the canonical caller should pass a `section` object.
    if section is None:
        return 0.0

    # If caller provided (path2d, transform) tuple, handle it
    try:
        # detect tuple-like path2d
        if isinstance(section, tuple) and len(section) >= 1:
            path2d = section[0]
            mesh_section = None
        else:
            # standard trimesh Section object
            mesh_section = section
            path2d = None
    except Exception:
        mesh_section = section
        path2d = None

    # Try to get a 2D path from the section if available
    if mesh_section is not None:
        try:
            path2d_obj, _ = mesh_section.to_2D()
            path2d = path2d_obj
        except Exception:
            try:
                path2d_obj, _ = mesh_section.to_planar()
                path2d = path2d_obj
            except Exception:
                path2d = None

    if path2d is not None:
        polygons = getattr(path2d, 'polygons_closed', None)
        if polygons:
            total_area = 0.0
            for polygon in polygons:
                if polygon is None:
                    continue
                if hasattr(polygon, 'is_valid') and not polygon.is_valid:
                    continue
                try:
                    total_area += float(polygon.area)
                except Exception:
                    # fallback to shoelace if exterior coords exist
                    try:
                        pts = np.array(polygon.exterior.coords)
                        if len(pts) >= 3:
                            total_area += 0.5 * abs(np.sum(pts[:, 0] * np.roll(pts[:, 1], 1) -
                                                          np.roll(pts[:, 0], 1) * pts[:, 1]))
                    except Exception:
                        continue
            if total_area > 0:
                return float(total_area)

        polygons_full = getattr(path2d, 'polygons_full', None)
        if polygons_full:
            total_area = 0.0
            for polygon in polygons_full:
                if polygon is None:
                    continue
                try:
                    total_area += float(polygon.area)
                except Exception:
                    continue
            if total_area > 0:
                return float(total_area)

    # If we have a mesh_section, try discrete segments from it
    pts_all = []
    if mesh_section is not None:
        for segment in getattr(mesh_section, 'discrete', []):
            if segment is None or len(segment) == 0:
                continue
            segment = np.asarray(segment)
            if segment.ndim == 2 and segment.shape[1] >= 2:
                pts_all.append(segment[:, :2])

    if not pts_all:
        return 0.0

    pts_2d = np.vstack(pts_all)
    if len(pts_2d) < 3:
        return 0.0

    # Try polygonize from segments
    try:
        lines = [LineString(seg) for seg in pts_all if len(seg) > 1]
        poly_candidates = list(polygonize(lines))
        if poly_candidates:
            union = unary_union(poly_candidates)
            if union.is_valid and union.area > 0:
                return float(union.area)
    except Exception:
        pass

    # Fallback to convex hull area or shoelace
    try:
        hull = ConvexHull(pts_2d)
        return float(hull.volume)
    except Exception:
        center = pts_2d.mean(axis=0)
        angles = np.arctan2(pts_2d[:, 1] - center[1], pts_2d[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        pts_sorted = pts_2d[sorted_idx]
        return float(0.5 * abs(np.sum(pts_sorted[:, 0] * np.roll(pts_sorted[:, 1], 1) -
                                np.roll(pts_sorted[:, 0], 1) * pts_sorted[:, 1])))


def extract_3d_slice_areas(mesh, z_levels):
    """Compute section areas for a sequence of z-levels from a trimesh mesh.

    Returns a numpy array of areas corresponding to `z_levels`.
    """
    areas = []
    for z in z_levels:
        plane_origin = np.array([0.0, 0.0, z])
        plane_normal = np.array([0.0, 0.0, 1.0])
        try:
            section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        except Exception:
            section = None
        area = section_area_from_mesh(section)
        areas.append(area)
    return np.array(areas)
