
import numpy as np
import pytest

from granular_analysis.analysis import porosity2d
from granular_analysis.analysis.porosity_time import compute_porosity_profile
from granular_analysis.geometry.ellipsoid import ellipse_points
from granular_analysis.scripts.validate_porosity_six_spheres import six_spheres, analytic_slice


def test_triangular_layer_is_nonoverlapping_and_has_nine_contacts():
    centers = np.array([p['x'] for p in six_spheres()])
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)[np.triu_indices(6, 1)]
    assert np.all(distances >= 2 - 1e-14)
    assert np.count_nonzero(np.isclose(distances, 2)) == 9


@pytest.mark.parametrize('z', [1.25, 1.5, 2., 2.5, 2.75])
def test_production_porosity_converges_to_analytic_circles(z, monkeypatch):
    outer, solid, phi = analytic_slice(z)
    errors = []
    for n in (10, 72, 720):
        monkeypatch.setattr(porosity2d, 'ellipse_points', lambda *a, **kw: ellipse_points(*a, **dict(kw, n=n)))
        row = compute_porosity_profile(six_spheres(), [z], alpha=0., lz=6.).iloc[0]
        assert row.valid
        errors.append(abs(row.porosity - phi))
    assert errors[2] < errors[1] < errors[0]
    assert row.A_alpha == pytest.approx(outer, rel=2e-5)
    assert row.A_ell == pytest.approx(solid, rel=2e-5)
    assert errors[2] < 1e-5


@pytest.mark.parametrize('z', [1.25, 2., 2.75])
def test_default_contour_resolution_matches_analytic_porosity(z):
    outer, solid, phi = analytic_slice(z)
    row = compute_porosity_profile(six_spheres(), [z], alpha=0., lz=6.).iloc[0]
    assert row.valid
    assert row.A_alpha == pytest.approx(outer, rel=2e-5)
    assert row.A_ell == pytest.approx(solid, rel=2e-5)
    assert abs(row.porosity - phi) < 1e-5
