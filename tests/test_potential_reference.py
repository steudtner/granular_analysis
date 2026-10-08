import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from granular_analysis.utils.physics import initial_potential_energy
from granular_analysis.plotting.plots import add_potential_reference


def test_initial_potential_uses_y_and_gravity_magnitude(tmp_path):
    global_input = tmp_path / "input.dat"
    global_input.write_text("grx=0\ngry=-2\ngrz=0\n")
    particles = tmp_path / "particles.dat"
    particles.write_text("ID,rho,a,b,c,x,y,z\n8,3,1,1,1,100,4,200\n2,3,1,1,1,300,5,400\n")
    assert initial_potential_energy(global_input, particles) == pytest.approx(72*np.pi)
    assert initial_potential_energy(global_input, particles, skip_first=1) == pytest.approx(40*np.pi)
    global_input.write_text("grx=2\ngry=0\ngrz=0\n")
    assert initial_potential_energy(global_input, particles) == pytest.approx(72*np.pi)


def test_reference_normalization_and_color():
    fig, ax = plt.subplots()
    table = pd.DataFrame(dict(initial_potential_energy=[100], initial_potential_energy_norm=[5]))
    add_potential_reference(ax, table, True, color="red")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [5, 5])
    assert ax.lines[0].get_color() == "red"
    plt.close(fig)


@pytest.mark.parametrize("enabled", [False, True])
def test_reference_opt_in(tmp_path, monkeypatch, enabled):
    from granular_analysis.plotting import plots
    table = pd.DataFrame(dict(time=[0, 1], kinetic_energy=[1, 2],
                              initial_potential_energy=[5, 5]))
    captured = []
    monkeypatch.setattr(plots.plotstyle, "export_figure",
                        lambda fig, *a, **kw: captured.append(len(fig.axes[0].lines)))
    plots.plot_energy_time_series(table, tmp_path / "test.png", potential_energy=enabled)
    assert captured == [2 if enabled else 1]
