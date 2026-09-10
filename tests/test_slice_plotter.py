from pathlib import Path

import numpy as np

from granular_analysis.plotting.slice_plotter import plot_exact_slice_from_ellipsoids


def test_plot_exact_slice_creates_output(tmp_path):
    ellipsoids = [
        {"x": np.array([1.0, 2.0, 0.0]), "a": 1.5, "b": 1.0, "c": 1.0},
    ]

    out_path = tmp_path / "slice.png"
    result = plot_exact_slice_from_ellipsoids(
        ellipsoids,
        z_plane=0.0,
        output_path=out_path,
        alphashape=False,
    )

    assert out_path.exists()
    assert result["output_path"] == str(out_path)
    assert result["num_ellipses"] == 1


def test_upper_edge_plot_uses_upperEdge_suffix(tmp_path):
    ellipsoids = [
        {"x": np.array([1.0, 2.0, 0.0]), "a": 1.5, "b": 1.0, "c": 1.0},
    ]

    out_path = tmp_path / "slice.svg"
    result = plot_exact_slice_from_ellipsoids(
        ellipsoids,
        z_plane=0.0,
        output_path=out_path,
        alphashape=True,
        upper_edge=True,
    )

    expected_path = tmp_path / "slice_upperEdge.svg"
    assert expected_path.exists()
    assert result["output_path"] == str(expected_path)
