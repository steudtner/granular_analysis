import numpy as np

from granular_analysis.analysis.slope import fit_slope_model, fit_slope_models


def test_linear_slope_fit():
    x = np.linspace(0.0, 10.0, 30)
    z = 4.0 - 0.5 * x
    model = fit_slope_model(x, z)

    assert model.n_breakpoints == 0
    assert np.isclose(model.intercept, 4.0)
    assert np.allclose(model.slopes, [-0.5])
    assert np.allclose(model.predict(x), z)


def test_one_breakpoint_fit_recovers_continuous_profile():
    x = np.linspace(0.0, 20.0, 80)
    z = 10.0 - 0.1 * x - 0.4 * np.maximum(0.0, x - 8.0)
    model = fit_slope_model(x, z, n_breakpoints=1)

    assert np.allclose(model.breakpoints, [8.0], atol=0.05)
    assert np.allclose(model.slopes, [-0.1, -0.5], atol=0.01)
    assert np.allclose(model.predict(x), z, atol=0.01)


def test_fit_slope_models_includes_requested_complexities():
    x = np.linspace(0.0, 20.0, 40)
    z = 5.0 - 0.2 * x

    models = fit_slope_models(x, z, max_breakpoints=2)

    assert [model.n_breakpoints for model in models] == [0, 1, 2]
