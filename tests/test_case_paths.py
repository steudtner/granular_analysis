from pathlib import Path

from granular_analysis.paths import (
    CasePaths,
    latest_common_collision_counter,
    latest_trn_counter,
)


def test_case_paths_resolve_every_supported_level(tmp_path):
    case = tmp_path / "case"
    trn = case / "results" / "ellipsoid" / "ellipsoid_trn"
    trn.mkdir(parents=True)
    expected = CasePaths.from_input(case)
    for supplied in (case, case / "results", case / "results" / "ellipsoid", trn):
        assert CasePaths.from_input(supplied) == expected


def test_counter_resolution_requires_matching_collision_frame(tmp_path):
    case = CasePaths.from_input(tmp_path / "case")
    case.trn.mkdir(parents=True)
    case.coll.mkdir(parents=True)
    for counter in (2, 10):
        (case.trn / f"ell_trn_{counter:06d}.bin").touch()
    for stage in (1, 2, 3):
        (case.coll / f"ell_coll_000002_{stage}.bin").touch()
    (case.coll / "ell_coll_000010_1.bin").touch()
    assert latest_trn_counter(case) == 10
    assert latest_common_collision_counter(case) == 2


def test_case_root_uses_configured_results_even_when_results_exists(tmp_path, monkeypatch):
    case = tmp_path / "case"
    (case / "results").mkdir(parents=True)
    (case / "input.dat").write_text("&general\n res_dir = './results_restart27', ! output\n/\n")
    trn = case / "results_restart27" / "ellipsoid" / "ellipsoid_trn"
    trn.mkdir(parents=True)
    (trn / "ell_trn_000001.bin").touch()
    assert CasePaths.from_input(case).require_trn() == trn
    monkeypatch.chdir(case)
    monkeypatch.delenv("GRANULAR_DATA_FOLDER", raising=False)
    assert CasePaths.from_input().trn == trn
    assert CasePaths.from_input(case / "results").results == case / "results"
    assert CasePaths.from_input(trn).results == trn.parent.parent


def test_configured_absolute_output_and_commented_setting(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    target = tmp_path / "output"
    (case / "input.dat").write_text(f'! res_dir="ignored"\n RES_DIR = "{target}"\n')
    assert CasePaths.from_input(case).results == target


def test_input_without_res_dir_keeps_default(tmp_path):
    (tmp_path / "input.dat").write_text("&general\n rst_dir='./restart'\n/\n")
    assert CasePaths.from_input(tmp_path).results == tmp_path / "results"


def test_single_stage_collision_counter(tmp_path):
    case = CasePaths.from_input(tmp_path)
    case.trn.mkdir(parents=True)
    case.coll.mkdir(parents=True)
    for counter in (0, 2, 10):
        (case.trn / f"ell_trn_{counter:06d}.bin").touch()
        (case.coll / f"ell_coll_{counter:06d}_1.bin").touch()
    assert latest_common_collision_counter(case) == 10
    # A partially written three-stage case must not become single-stage.
    (case.coll / "ell_coll_000002_2.bin").touch()
    assert latest_common_collision_counter(case) is None
