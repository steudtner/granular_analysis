import numpy as np
from granular_analysis.geometry.alpha2d import compute_alpha_shape_2d


def test_circle_alpha_shape():
    """Alpha shape of a dense circle should approximate pi*r^2."""
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    coords, area = compute_alpha_shape_2d(pts, alpha=0.5)
    expected = np.pi
    assert area > 0
    assert abs(area - expected) / expected < 0.05


def test_empty_points():
    coords, area = compute_alpha_shape_2d(np.empty((0, 2)), 0.5)
    assert coords is None
    assert area == 0.0


def test_too_few_points():
    pts = np.array([[0, 0], [1, 0]])
    coords, area = compute_alpha_shape_2d(pts, 0.5)
    assert coords is None
    assert area == 0.0


def test_rectangle():
    """With a very large alpha, result should be close to the convex hull (area ~ 1)."""
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    coords, area = compute_alpha_shape_2d(pts, alpha=1000)
    assert area > 0.9


if __name__ == "__main__":
    test_circle_alpha_shape()
    test_empty_points()
    test_too_few_points()
    test_rectangle()
    print("test_alpha2d: OK")
