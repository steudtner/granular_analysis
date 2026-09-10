import importlib
import os


def test_run_all_prefers_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRANULAR_DATA_FOLDER", raising=False)
    monkeypatch.delenv("GRANULAR_RESULTS_DIR", raising=False)

    import granular_analysis.config as config
    import granular_analysis.scripts.run_all as run_all

    importlib.reload(config)
    importlib.reload(run_all)

    expected_data = os.path.join(str(tmp_path), "results", "ellipsoid", "ellipsoid_trn")
    expected_results = os.path.join(str(tmp_path), "results_granular_analysis")

    assert run_all._resolve_data_folder() == expected_data
    assert config.FOLDER == expected_data
    assert config.RESULTS_DIR == expected_results
