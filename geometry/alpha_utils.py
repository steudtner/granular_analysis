"""Shared geometry helpers used across alpha-shape variants."""
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from sklearn.decomposition import PCA


def _largest_polygon(shape):
    """Return the polygon with the largest area from a (Multi)Polygon."""
    if isinstance(shape, MultiPolygon):
        return max(shape.geoms, key=lambda g: g.area)
    return shape


def _pca_transform(points):
    """PCA-align a 3D point cloud. Returns (transformed_points, pca_object)."""
    pca = PCA(n_components=3)
    pts_t = pca.fit_transform(points)
    return pts_t, pca


def _inverse_pca(mesh, pca):
    """Back-transform mesh vertices from PCA space."""
    mesh.vertices = pca.inverse_transform(mesh.vertices)
    return mesh
