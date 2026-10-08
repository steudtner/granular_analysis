import numpy as np
import pytest
from granular_analysis.plotting.plots import smooth_energy_curve


def test_noise_reduced_without_mutating_measurements():
    raw = np.array([1., 3.] * 50)
    original = raw.copy()
    smoothed = smooth_energy_curve(raw)
    assert np.std(smoothed[4:-4]) < np.std(raw) / 5
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(smooth_energy_curve(raw, 1), raw)


def test_constant_short_series_and_gaps():
    np.testing.assert_allclose(smooth_energy_curve([5., 5.]), [5., 5.])
    np.testing.assert_allclose(smooth_energy_curve([1., 1., np.nan, 9., 9.]),
                               [1., 1., np.nan, 9., 9.], equal_nan=True)
    assert smooth_energy_curve([]).size == 0


@pytest.mark.parametrize("window", [0, -1, 2, 2.5, True])
def test_invalid_window(window):
    with pytest.raises(ValueError):
        smooth_energy_curve([1., 2.], window)


def test_components_separate_axes_and_preserve_input(tmp_path, monkeypatch):
    import pandas as pd
    import matplotlib.figure
    from granular_analysis.plotting.plots import plot_energy_components
    table = pd.DataFrame({"time": range(15), "kinetic_energy": [3.] * 15,
                          "translational_energy": [2.] * 15, "rotational_energy": [1.] * 15})
    original = table.copy(deep=True)
    captured = []
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *a, **k: captured.append(self))
    plot_energy_components(table, tmp_path / "components.png")
    assert len(captured[0].axes) == 3
    assert all(len(ax.lines) == 1 for ax in captured[0].axes)
    pd.testing.assert_frame_equal(table, original)


def test_display_reduction_preserves_endpoints_gaps_and_raw_values():
    from granular_analysis.plotting.plots import energy_display_points
    x = np.arange(1000, dtype=float)
    y = 2 * x
    original = y.copy()
    xx, yy = energy_display_points(x, y, 1, 30)
    assert len(xx) == 30
    np.testing.assert_allclose(yy, 2 * xx)
    np.testing.assert_array_equal(xx[[0, -1]], x[[0, -1]])
    np.testing.assert_array_equal(y, original)
    y[500] = np.nan
    xx, yy = energy_display_points(x, y, 1, 30)
    assert np.isnan(yy).sum() == 1
    assert len(xx) == 61
    xx, yy = energy_display_points(x, original, 1, None)
    np.testing.assert_array_equal(xx, x)
    np.testing.assert_array_equal(yy, original)
