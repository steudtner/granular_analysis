from granular_analysis.utils.state_manager import (
    build_fingerprint,
    load_state,
    save_state,
)


def test_cache_fingerprint_rejects_changed_parameters(tmp_path):
    first = build_fingerprint(path="case-a", alpha=0.6)
    second = build_fingerprint(path="case-a", alpha=0.7)
    save_state("2d", {"porosity": 0.4}, str(tmp_path), fingerprint=first)
    assert load_state("2d", str(tmp_path), fingerprint=first)["porosity"] == 0.4
    assert load_state("2d", str(tmp_path), fingerprint=second) is None
    assert not (tmp_path / "analysis_state.json.tmp").exists()
