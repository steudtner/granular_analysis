import numpy as np
import pytest
import pandas as pd

from granular_analysis.scripts.run_energy_analysis import _load_monitor_series
from granular_analysis.utils.physics import read_density_first_particle, read_gravity


def _write_monitor(path, particle_id, speed):
    rows = np.zeros((2, 11))
    rows[:, 0] = particle_id
    rows[:, 1] = [0.0, 1.0]
    rows[:, 5] = speed
    np.savetxt(path, rows)


def test_energy_maps_missing_monitors_by_input_row(tmp_path):
    _write_monitor(tmp_path / "ell_1_monitor.dat", 1, 1.0)
    _write_monitor(tmp_path / "ell_3_monitor.dat", 3, 2.0)
    _, translational, rotational, total, _, _ = _load_monitor_series(
        tmp_path,
        particle_ids=np.array([3, 10, 20]),
        masses=np.array([4.0, 100.0, 2.0]),
        semi_axes=np.ones((3, 3)),
    )
    np.testing.assert_allclose(translational, [6.0, 6.0])
    np.testing.assert_allclose(rotational, 0.0)
    np.testing.assert_allclose(total, translational)


@pytest.mark.parametrize("monitor_number", [0, 4])
def test_energy_rejects_monitor_outside_input_rows(tmp_path, monitor_number):
    _write_monitor(tmp_path / f"ell_{monitor_number}_monitor.dat", monitor_number, 1.0)
    with pytest.raises(RuntimeError, match="input contains 3 particles"):
        _load_monitor_series(
            tmp_path,
            particle_ids=np.array([0, 4, 10]),
            masses=np.ones(3),
            semi_axes=np.ones((3, 3)),
        )


def test_energy_maps_contiguous_monitor_numbers_by_input_row(tmp_path):
    _write_monitor(tmp_path / "ell_1_monitor.dat", 1, 1.0)
    _write_monitor(tmp_path / "ell_2_monitor.dat", 2, 2.0)
    _write_monitor(tmp_path / "ell_3_monitor.dat", 3, 3.0)

    # Physical ID 2 was removed. PRIME nevertheless numbers the three monitor
    # files consecutively according to the remaining input-row positions.
    _, translational, rotational, total, _, _ = _load_monitor_series(
        tmp_path,
        particle_ids=np.array([1, 3, 4]),
        masses=np.array([2.0, 4.0, 6.0]),
        semi_axes=np.ones((3, 3)),
    )

    np.testing.assert_allclose(translational, [36.0, 36.0])
    np.testing.assert_allclose(rotational, 0.0)
    np.testing.assert_allclose(total, translational)


def test_physics_parsers_accept_headers_and_scientific_notation(tmp_path):
    global_input = tmp_path / "input.dat"
    global_input.write_text("grx = 0.d0\ngry = -6.0D-2\ngrz = -9.81e-1\n")
    particle_input = tmp_path / "input_ellipsoid_data.dat"
    particle_input.write_text("id,rho,a,b,c\n10,2.5,1,1,1\n")
    assert np.isclose(read_gravity(global_input), np.sqrt(0.06**2 + 0.981**2))
    assert read_density_first_particle(particle_input) == 2.5


def test_energy_reads_monitors_once_for_curves_and_histogram(tmp_path, monkeypatch):
    from granular_analysis.scripts import run_energy_analysis as energy

    (tmp_path / "input.dat").write_text("grx=0.d0\ngry=-0.06d0\ngrz=0.d0\n")
    (tmp_path / "input_ellipsoid_data.dat").write_text(
        "10,2.5,1,1,1\n20,5,1,1,1\n"
    )
    monitors = tmp_path / "results" / "ellipsoid"
    monitors.mkdir(parents=True)
    _write_monitor(monitors / "ell_1_monitor.dat", 1, 1.0)
    _write_monitor(monitors / "ell_2_monitor.dat", 2, 2.0)
    reads = []
    original_read = energy.read_monitor_file

    def read(path):
        reads.append(path)
        return original_read(path)

    histogram_values = []
    original_plot = energy.plot_energy_histogram

    def plot(values, *args, **kwargs):
        histogram_values.append(values.copy())
        return original_plot(values, *args, **kwargs)

    monkeypatch.setattr(energy, "read_monitor_file", read)
    monkeypatch.setattr(energy, "plot_energy_histogram", plot)
    output = tmp_path / "analysis" / "energy.csv"
    energy.main(path=str(tmp_path), out=str(output), histogram_timesteps="0,-1", skip_first=0)
    assert len(reads) == 2
    df = pd.read_csv(output)
    assert len(histogram_values) == 2
    np.testing.assert_allclose(
        [values.sum() for values in histogram_values], df["kinetic_energy"]
    )

    assert df["translational_rotational_ratio"].isna().all()
    ratio_plot = output.parent / "energy_ratio.png"
    assert ratio_plot.is_file()
    ratio_plot.unlink()

    # A missing ratio plot must bypass the cache shortcut and reuse the CSV.
    reads.clear()
    energy.main(path=str(tmp_path), out=str(output), histogram_timesteps="0,-1", skip_first=0)
    assert len(reads) == 2
    assert output.with_suffix(".png").is_file()
    assert ratio_plot.is_file()


def test_energy_skip_by_row_and_histogram(tmp_path):
    for row in (1, 2, 3):
        _write_monitor(tmp_path / f"ell_{row}_monitor.dat", row, float(row))
    _, _, _, total, per_particle, files = _load_monitor_series(
        tmp_path, np.array([50, 1, 2]), np.ones(3)*2, np.ones((3, 3)), skip_first=2)
    np.testing.assert_allclose(total, 9)
    np.testing.assert_allclose(per_particle, 9)
    assert [number for number, _ in files] == [3]


def test_translational_rotational_ratio_handles_undefined_values():
    from granular_analysis.analysis.energy.kinetic_energy import translational_rotational_ratio
    with np.errstate(divide="raise", invalid="raise"):
        actual = translational_rotational_ratio(
            [6, 0, 2, 0, 1, np.nan, 1], [2, 2, 0, 0, np.nan, 1, 1e-12]
        )
    np.testing.assert_allclose(actual, [3, 0, np.nan, np.nan, np.nan, np.nan, 1e12])


@pytest.mark.parametrize('threshold, expected', [(0.5, [0.2, 0.4]), (1.0, [20.2, 1.2])])
def test_dominant_rotation_filter_keeps_translation_and_records_raw(tmp_path, threshold, expected):
    for number, omega in [(1, [10.0, 1.0]), (2, [1.0, 1.0]), (3, [0.0, 2.0])]:
        rows = np.zeros((2, 11))
        rows[:, 0] = [1, 2]
        rows[:, 1] = [0, 1]
        rows[:, 5] = 2
        rows[:, 8] = omega
        np.savetxt(tmp_path / f'ell_{number}_monitor.dat', rows)
    diagnostics = {}
    _, translation, rotation, total, particles, _ = _load_monitor_series(
        tmp_path, np.arange(1, 4), np.ones(3), np.ones((3, 3)),
        rotation_share_threshold=threshold, diagnostics=diagnostics,
    )
    np.testing.assert_allclose(rotation, expected)
    np.testing.assert_allclose(translation, [6, 6])
    np.testing.assert_allclose(total, translation + rotation)
    np.testing.assert_allclose(particles.sum(axis=0), total)
    np.testing.assert_allclose(diagnostics['rotational_energy_raw'], [20.2, 1.2])
    np.testing.assert_array_equal(diagnostics['rotation_excluded_monitor'], [1, 3] if threshold == 0.5 else [0, 0])


@pytest.mark.parametrize('threshold', [0.49, 1.01, float('nan')])
def test_invalid_rotation_threshold(tmp_path, threshold):
    with pytest.raises(ValueError, match='between 0.5 and 1'):
        _load_monitor_series(tmp_path, [], [], [], rotation_share_threshold=threshold)


def test_dataframe_uses_case_inputs_and_keeps_threshold_boundary(tmp_path):
    from granular_analysis.scripts.run_energy_analysis import compute_energy_dataframe
    (tmp_path / 'input.dat').write_text('grx=0.d0\ngry=-0.06d0\ngrz=0.d0\n')
    (tmp_path / 'input_ellipsoid_data.dat').write_text('1,2.5,1,1,1\n2,2.5,1,1,1\n')
    monitors = tmp_path / 'results' / 'ellipsoid'
    monitors.mkdir(parents=True)
    for number in (1, 2):
        rows = np.zeros((2, 11))
        rows[:, 0] = [1, 2]
        rows[:, 1] = [0, 1]
        rows[:, 8] = 1
        np.savetxt(monitors / f'ell_{number}_monitor.dat', rows)
    df = compute_energy_dataframe(tmp_path, skip_first=0)
    assert (df.rotation_share_threshold == 0.5).all()
    assert (df.rotation_excluded_monitor == 0).all()
    np.testing.assert_allclose(df.rotational_energy, df.rotational_energy_raw)
    np.testing.assert_allclose(df.rotational_energy_raw_norm, df.rotational_energy_norm)
