import pandas as pd
import pytest

from granular_analysis.scripts.run_energy_comparison import plot_energy_comparison, run_energy_comparison
from granular_analysis.scripts.run_all import build_parser


def test_native_times_and_outputs(tmp_path):
    tables = {"first": pd.DataFrame({"time": [0., 1., 2.], "kinetic_energy": [1., 4., 2.]}),
              "second": pd.DataFrame({"time": [5., 7.], "kinetic_energy": [3., 6.]})}
    prefix = plot_energy_comparison(tables, tmp_path, normalized=False)
    result = pd.read_csv(prefix.with_suffix(".csv"))
    assert list(result.time) == [0., 1., 2., 5., 7.]
    assert list(result.kinetic_energy) == [1., 4., 2., 3., 6.]
    assert prefix.with_suffix(".png").is_file()
    assert prefix.with_suffix(".svg").is_file()


def test_normalized_component(tmp_path):
    table = pd.DataFrame({"time_norm": [0., 2.], "rotational_energy_norm": [1., 3.]})
    prefix = plot_energy_comparison({"a": table, "b": table}, tmp_path, "rotational")
    assert "rotational_energy_normalized" in prefix.name


def test_duplicate_cases_rejected(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        run_energy_comparison([tmp_path, tmp_path], tmp_path)


def test_compare_cli():
    args = build_parser().parse_args(["--only", "energy", "--compare", "other", "--energy-raw"])
    assert args.compare == ["other"]
    assert args.energy_raw
