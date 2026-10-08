import numpy as np

from granular_analysis.io.monitor_loader import read_monitor_file, monitor_source_paths
from granular_analysis.io.fluid_slice import restart_records


def test_history_is_joined_and_current_overlap_wins(tmp_path):
    path = tmp_path / "ell_0001_monitor.dat"
    history = tmp_path / "history"
    history.mkdir()
    old = np.zeros((3, 11))
    old[:, 1] = [0, 1, 2]
    old[:, 5] = 10
    current = np.zeros((2, 11))
    current[:, 1] = [2, 3]
    current[:, 5] = 20
    np.savetxt(history / path.name, old)
    np.savetxt(path, current)
    data = read_monitor_file(path)
    np.testing.assert_array_equal(data[:, 1], [0, 1, 2, 3])
    np.testing.assert_array_equal(data[:, 5], [10, 10, 20, 20])
    assert monitor_source_paths(path) == [history / path.name, path]
    np.testing.assert_array_equal(np.loadtxt(path), current)


def test_restart_history_prefers_current_records(tmp_path):
    (tmp_path / "history").mkdir()
    (tmp_path / "history/rst_info.dat").write_text("counter t\n1 0.1\n2 0.2\n")
    (tmp_path / "rst_info.dat").write_text("counter t\n2 0.25\n3 0.3\n")
    assert restart_records(tmp_path) == [
        {"counter": "1", "t": "0.1"},
        {"counter": "2", "t": "0.25"},
        {"counter": "3", "t": "0.3"},
    ]
