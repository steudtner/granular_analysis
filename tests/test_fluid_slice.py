import numpy as np
import pytest
from scipy.io import FortranFile

from granular_analysis.io.fluid_slice import load_fluid_slice, latest_common_counter, nearest_frame
from granular_analysis.paths import CasePaths
from granular_analysis.io.fluid_slice import vortex_criterion, spatial_derivative


@pytest.mark.parametrize("gradient,q,lambda2", [
    (np.zeros((3, 3)), 0, 0),
    ([[0, -2, 0], [2, 0, 0], [0, 0, 0]], 4, -4),
    ([[2, 0, 0], [0, -2, 0], [0, 0, 0]], -4, 4),
    ([[0, 2, 0], [0, 0, 0], [0, 0, 0]], 0, 0),
])
def test_vortex_tensors(gradient, q, lambda2):
    gradient = np.asarray(gradient, dtype=float)
    assert vortex_criterion(gradient, "q") == pytest.approx(q)
    assert vortex_criterion(gradient, "lambda2") == pytest.approx(lambda2)


def test_derivative_nonuniform_and_periodic():
    x = np.array([0., .2, .9, 2.])
    np.testing.assert_allclose(spatial_derivative(x*x, x, 0), 2*x, atol=1e-14)
    x = (np.arange(100) + .5)*2*np.pi/100
    np.testing.assert_allclose(spatial_derivative(np.sin(x), x, 0, 2*np.pi), np.cos(x), atol=.001)


@pytest.mark.parametrize("component,expected", [("q", -1.5), ("lambda2", 1)])
def test_vortex_from_full_3d_gradient(fluid_case, component, expected):
    result = load_fluid_slice(fluid_case, 10, 1., component)
    np.testing.assert_allclose(result["values"], expected)


@pytest.mark.parametrize("component,expected", [("q", 4), ("lambda2", -4)])
def test_vortex_rotation_in_xz_plane(fluid_case, component, expected):
    z, y, x = np.indices((4, 4, 4))
    velocity = np.stack((-2*(z-.5), np.zeros_like(y), 2*(x-.5)), axis=-1).astype(">f8")
    with (fluid_case.results / "fluid/uvw_000003.bin").open("wb") as stream:
        np.array([1211214, velocity.size], dtype=">i4").tofile(stream)
        velocity.tofile(stream)
    np.testing.assert_allclose(load_fluid_slice(fluid_case, 10, 1., component)["values"], expected)


@pytest.fixture
def fluid_case(tmp_path):
    case = CasePaths.from_input(tmp_path)
    (tmp_path / "input.dat").write_text(
        "rho_f=1.0D3, ! water density 1000\n"
        "grx=0.d0\ngry=-6.0D-2, ! gravity 123\ngrz=0.d0\n"
    )
    case.trn.mkdir(parents=True)
    fluid = case.results / "fluid"
    fluid.mkdir()
    for counter, time in [(10, 2.0), (11, 2.01)]:
        with FortranFile(case.trn / f"ell_trn_{counter:06d}.bin", "w") as stream:
            stream.write_record(np.array([0], dtype="i4"))
            stream.write_record(np.array([time]))
    (case.results / "rst_info.dat").write_text(
        "counter t nx ny nz sx sy sz ex ey ez xper yper zper\n"
        "3 2.0 2 2 2 0 0 0 3 3 3 F F F\n"
    )
    grid = np.column_stack((np.arange(4) - 0.5, np.arange(4)))
    for axis in "xyz":
        np.savetxt(fluid / f"{axis}.dat", grid)
    z, y, x = np.indices((4, 4, 4))
    velocity = np.stack((x, y, z), axis=-1).astype(">f8")
    with (fluid / "uvw_000003.bin").open("wb") as stream:
        np.array([1211214, velocity.size], dtype=">i4").tofile(stream)
        velocity.tofile(stream)
    pressure = (x - 0.5 + 2*(y - 0.5) + 3*(z - 0.5)).astype(">f8")
    with (fluid / "pre_000003.bin").open("wb") as stream:
        np.array([1211214, pressure.size], dtype=">i4").tofile(stream)
        pressure.tofile(stream)
    return case


@pytest.mark.parametrize("component", ["magnitude", "u", "v", "w"])
def test_staggered_components_and_z_interpolation(fluid_case, component):
    result = load_fluid_slice(fluid_case, 10, 1.0, component)
    x, y = np.meshgrid([0.5, 1.5], [0.5, 1.5])
    expected = {"u": x, "v": y, "w": np.ones_like(x),
                "magnitude": np.sqrt(x*x + y*y + 1)}
    np.testing.assert_allclose(result["values"], expected[component])
    assert result["counter"] == 3
    assert result["time"] == 2.0


def test_time_matching_never_uses_nearest_snapshot(fluid_case):
    assert latest_common_counter(fluid_case) == 10
    with pytest.raises(ValueError, match="No fluid checkpoint"):
        load_fluid_slice(fluid_case, 11, 1.0)


def test_incomplete_fluid_snapshot_is_rejected(fluid_case):
    path = fluid_case.results / "fluid/uvw_000003.bin"
    with path.open("r+b") as stream:
        stream.truncate(20)
    with pytest.raises(ValueError, match="Incomplete"):
        load_fluid_slice(fluid_case, 10, 1.0)


def test_pressure_is_cell_centered_and_needs_no_velocity_file(fluid_case, monkeypatch):
    monkeypatch.setattr("granular_analysis.io.fluid_slice.D_EQ", 2.5)
    (fluid_case.results / "fluid/uvw_000003.bin").unlink()
    assert latest_common_counter(fluid_case, "pressure") == 10
    result = load_fluid_slice(fluid_case, 10, 1.0, "pressure")
    x, y = np.meshgrid([0.5, 1.5], [0.5, 1.5])
    np.testing.assert_allclose(result["values"], (x + 2*y + 3) / (1000 * 0.06 * 2.5))
    with pytest.raises(ValueError, match="No fluid checkpoint"):
        load_fluid_slice(fluid_case, 11, 1.0, "pressure")


@pytest.mark.parametrize("diameter", [0, -1, np.nan, np.inf])
def test_pressure_rejects_invalid_diameter(fluid_case, monkeypatch, diameter):
    monkeypatch.setattr("granular_analysis.io.fluid_slice.D_EQ", diameter)
    with pytest.raises(ValueError, match="D_EQ"):
        load_fluid_slice(fluid_case, 10, 1.0, "pressure")


def test_pressure_selection_requires_pressure_file(fluid_case):
    (fluid_case.results / "fluid/pre_000003.bin").unlink()
    with pytest.raises(ValueError, match="No particle frame"):
        latest_common_counter(fluid_case, "pressure")


@pytest.mark.parametrize("text", [
    "rho_f=1000\ngrx=0\ngry=0\ngrz=0\n",
    "rho_f=0\ngrx=0\ngry=-0.06\ngrz=0\n",
    "grx=0\ngry=-0.06\ngrz=0\n",
])
def test_pressure_rejects_invalid_normalization(fluid_case, text):
    (fluid_case.case_root / "input.dat").write_text(text)
    with pytest.raises(ValueError):
        load_fluid_slice(fluid_case, 10, 1.0, "pressure")


def test_nearest_time_selects_a_joint_pair_not_nearest_particle(fluid_case):
    assert nearest_frame(fluid_case, 2.01) == (2.01, 11)
    assert nearest_frame(fluid_case, 2.01, "v") == (2.0, 10)
    assert nearest_frame(fluid_case, -10, "pressure") == (2.0, 10)
    with pytest.raises(ValueError, match="finite"):
        nearest_frame(fluid_case, float("nan"), "v")


def test_counter_mismatch_fails_before_analysis(fluid_case, monkeypatch, capsys, tmp_path):
    from granular_analysis.scripts import run_all
    monkeypatch.setattr(run_all, "run_2d", lambda **kw: pytest.fail("analysis ran"))
    output = tmp_path / "should_not_exist"
    assert run_all.main([
        "--only", "2d", "--fluid", "v", "--counter", "11",
        "--path", str(fluid_case.case_root), "--out", str(output),
    ]) == 1
    captured = capsys.readouterr()
    assert "--counter 10" in captured.out
    assert "--time 2.01" in captured.out
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_time_selection_reports_actual_pair(fluid_case, capsys):
    from granular_analysis.scripts.run_all import main
    assert main(["--only", "2d", "--fluid", "v", "--time", "2.01", "--dry",
                 "--path", str(fluid_case.case_root)]) == 0
    output = capsys.readouterr().out
    assert "COUNTER    = 10" in output
    assert "requested 2.01" in output
    assert "FLUID      = 000003" in output


@pytest.mark.parametrize('component', ['magnitude', 'pressure', 'q'])
def test_slice_preserves_actual_cell_faces(fluid_case, component):
    result = load_fluid_slice(fluid_case, 10, 1., component)
    np.testing.assert_allclose(result['x_edges'], [0., 1., 2.])
    np.testing.assert_allclose(result['y_edges'], [0., 1., 2.])
    assert len(result['x_edges']) == len(result['x']) + 1
