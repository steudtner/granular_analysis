import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import FortranFile

from granular_analysis.paths import CasePaths
from granular_analysis.scripts import run_geometry_comparison as comparison


def make_case(path, times):
    case = CasePaths.from_input(path)
    case.trn.mkdir(parents=True)
    for counter, time in times:
        with FortranFile(case.trn / f"ell_trn_{counter:06d}.bin", "w") as stream:
            stream.write_record(np.array([0], dtype="i4"))
            stream.write_record(np.array([time], dtype=float))
    return path


def test_case_time_selection(tmp_path):
    a = make_case(tmp_path / "a", [(1, 1), (2, 3)])
    b = make_case(tmp_path / "b", [(10, 2), (20, 4)])
    frames = comparison.select_comparison_frames([a, b], time=2.8)
    assert [(row[2], row[3]) for row in frames] == [(2, 3), (10, 2)]
    with pytest.raises(ValueError, match="distinct"):
        comparison.select_comparison_frames([a, a])


def test_all_geometry_outputs(tmp_path, monkeypatch):
    a = make_case(tmp_path / "a", [(1, 1)])
    b = make_case(tmp_path / "b", [(1, 2)])
    shape = np.array([[0, 0], [0, 1], [1, 1], [2, .5], [2, 0], [0, 0]])
    monkeypatch.setattr(comparison, "compute_slice_porosity", lambda *a, **kw:
                        dict(shape=shape, porosity=.4, A_ell=1, A_alpha=2, n=3))
    monkeypatch.setattr(comparison, "compute_2d_slice_summary", lambda *a, **kw:
                        pd.DataFrame(dict(z=[0, 1], porosity=[.3, .4], A_ell=[1, 1], A_alpha=[2, 2])))
    def slope(**kwargs):
        out = Path(kwargs["out"]) / "slope_analysis"
        out.mkdir()
        pd.DataFrame([dict(n_breakpoints=0, slopes="-0.25", intercept=1., breakpoints="",
                           aic=1., bic=2.)]).to_csv(out / "slope_results_R.csv", index=False)
    monkeypatch.setattr(comparison, "run_slope", slope)
    target = comparison.run_geometry_comparison([a, b], tmp_path / "out", ["2d", "slices", "slope"], skip_first=0)
    for name in ["alphashapes_comparison", "upper_edges_comparison", "porosity_profile_comparison", "slope_0bp_comparison"]:
        assert (target / f"{name}.png").is_file()
        assert (target / f"{name}.svg").is_file()
    manifest = json.loads((target / "comparison.json").read_text())
    assert [row["time"] for row in manifest["cases"]] == [1, 2]
    assert set(pd.read_csv(target / "porosity_2d.csv")["case"]) == {"a", "b"}


def test_slope_predictions():
    row = dict(slopes="-1;-2", intercept=3, breakpoints="1")
    np.testing.assert_allclose(comparison.slope_predictions(row, np.array([0., 1., 2.])), [3, 2, 0])
