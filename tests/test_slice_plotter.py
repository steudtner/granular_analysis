from pathlib import Path

import numpy as np
import pytest

from granular_analysis.plotting.slice_plotter import plot_exact_slice_from_ellipsoids


@pytest.mark.parametrize("enabled", [True, False])
def test_particle_speed_colours(tmp_path, monkeypatch, enabled):
    from granular_analysis.plotting import slice_plotter
    from matplotlib.quiver import Quiver
    import matplotlib.pyplot as plt

    def inspect(fig, path):
        ax = fig.axes[0]
        assert not any(isinstance(item, Quiver) for item in ax.collections)
        assert all(line.get_linewidth() == .5 for line in ax.lines)
        if enabled:
            colours = np.array([patch.get_facecolor() for patch in ax.patches])
            np.testing.assert_allclose(colours[0], [1., 0., 0., 1.])
            np.testing.assert_allclose(colours[2], [1., 4/9, 4/9, 1.])
            from matplotlib.colors import BoundaryNorm
            colourbar = fig.axes[1]._colorbar
            assert isinstance(colourbar.norm, BoundaryNorm)
            assert colourbar.cmap.N == 10
            assert len(colourbar.norm.boundaries) == 11
            np.testing.assert_allclose(colours[3:], [[1., 1., 1., 1.]] * 2)
            assert len(fig.axes) == 2
            assert fig.axes[1].get_position().y0 > ax.get_position().y1
            assert fig.axes[1].get_xlim()[1] == pytest.approx(4 / np.sqrt(.06))
            assert fig.axes[1].get_xlabel() == slice_plotter.plotstyle.velocity_label("p")
        else:
            assert not ax.patches
            assert len(fig.axes) == 1
        fig.canvas.draw()
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(slice_plotter, "save_figure", inspect)
    particles = [dict(x=np.array([i+1., 1., 0.]), a=.5, b=.5, c=.5, u=v)
                 for i, v in enumerate([[0, 0, 4], [-1, 0, 0], [0, 0, 2],
                                        [0, 0, 0], [np.nan, 0, 0]])]
    plot_exact_slice_from_ellipsoids(particles, 0, output_path=tmp_path / "speed.svg",
                                   velocity_colors=enabled)


@pytest.mark.parametrize("component", ["lambda2", "q"])
@pytest.mark.parametrize("amplitude", [0., 1e-20, 10.])
def test_vortex_symlog(tmp_path, monkeypatch, component, amplitude):
    from granular_analysis.plotting import slice_plotter
    from matplotlib.colors import SymLogNorm
    import matplotlib.pyplot as plt

    def inspect(fig, path):
        norm = fig.axes[0].collections[0].norm
        assert isinstance(norm, SymLogNorm)
        if component == "q":
            assert norm.vmax == pytest.approx(amplitude/.06 if amplitude else 1.)
            np.testing.assert_allclose(fig.axes[0].collections[0].get_array(),
                amplitude/.06 * np.tile([-1., 0., 1.], (3, 1)))
            assert fig.axes[1].get_xlabel() == slice_plotter.plotstyle.Q_LABEL
        assert norm(0) == pytest.approx(.5)
        assert norm.vmin == -norm.vmax
        assert norm.linthresh == pytest.approx(norm.vmax * 1e-3)
        assert np.isfinite(norm([-amplitude, 0, amplitude])).all()
        if amplitude:
            assert norm(-amplitude) < norm(0) < norm(amplitude)
        fig.canvas.draw()
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(slice_plotter, "save_figure", inspect)
    plot_exact_slice_from_ellipsoids(
        [{"x": np.array([1., 1., 0.]), "a": .5, "b": .5, "c": .5}],
        z_plane=0, output_path=tmp_path / "vortex.svg",
        fluid={"x": np.arange(3.), "y": np.arange(3.),
               "values": amplitude * np.tile([-1., 0., 1.], (3, 1)),
               "component": component, "time": 2., "counter": 3},
    )


def test_fluid_is_covered_inside_all_particles(tmp_path, monkeypatch):
    from granular_analysis.plotting import slice_plotter
    import matplotlib.pyplot as plt

    def inspect(fig, path):
        patches = fig.axes[0].patches
        assert len(patches) == 1
        assert patches[0].get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert patches[0].get_zorder() > fig.axes[0].collections[0].get_zorder()
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(slice_plotter, "save_figure", inspect)
    plot_exact_slice_from_ellipsoids(
        [{"id": 1, "x": np.array([1., 1., 0.]), "a": .5, "b": .5, "c": .5}],
        z_plane=0, skip_points=1000, output_path=tmp_path / "fluid.svg",
        fluid={"x": np.array([0., 1., 2.]), "y": np.array([0., 1., 2.]),
               "values": np.ones((3, 3)), "component": "magnitude",
               "time": 2., "counter": 3},
    )


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


def test_occupied_area_uses_union_for_overlapping_particles(tmp_path):
    particle = {"x": np.array([1.0, 2.0, 0.0]), "a": 1.5, "b": 1.0, "c": 1.0}
    single = plot_exact_slice_from_ellipsoids(
        [particle], z_plane=0.0, output_path=tmp_path / "single.png"
    )
    overlapping = plot_exact_slice_from_ellipsoids(
        [particle, particle], z_plane=0.0, output_path=tmp_path / "double.png"
    )
    assert overlapping["occupied_area"] == pytest.approx(single["occupied_area"])


def test_periodic_slice_includes_boundary_particle(tmp_path):
    particle = {"x": np.array([1.0, 2.0, 0.1]), "a": 0.5, "b": 0.5, "c": 0.5}
    result = plot_exact_slice_from_ellipsoids(
        [particle],
        z_plane=4.9,
        periodic_length=5.0,
        output_path=tmp_path / "periodic.png",
    )
    assert result["num_ellipses"] == 1


def test_dimensionless_fluid_scale_preserves_input(tmp_path, monkeypatch):
    from granular_analysis.plotting import slice_plotter
    import matplotlib.pyplot as plt
    raw = np.full((3, 3), np.sqrt(.06))

    def inspect(fig, path):
        np.testing.assert_allclose(fig.axes[0].collections[0].get_array(), 1.)
        assert fig.axes[1].get_xlabel() == slice_plotter.plotstyle.velocity_label('f')
        assert fig.axes[2].get_xlabel() == slice_plotter.plotstyle.velocity_label('p')
        for bar in fig.axes[1:]:
            assert bar.get_position().y0 > fig.axes[0].get_position().y1
            assert bar.get_position().x0 == pytest.approx(fig.axes[0].get_position().x0)
            assert bar.get_position().x1 == pytest.approx(fig.axes[0].get_position().x1)
        fig.canvas.draw()
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(slice_plotter, 'save_figure', inspect)
    plot_exact_slice_from_ellipsoids(
        [dict(x=np.array([1., 1., 0.]), a=.5, b=.5, c=.5, u=[np.sqrt(.06), 0, 0])],
        0, output_path=tmp_path/'normalized.svg',
        fluid=dict(x=np.arange(3.), y=np.arange(3.), values=raw,
                   component='magnitude', time=2., counter=3),
    )
    np.testing.assert_array_equal(raw, np.full((3, 3), np.sqrt(.06)))


def test_fluid_grid_uses_faces_and_is_behind_particles(tmp_path, monkeypatch):
    from granular_analysis.plotting import slice_plotter
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    def inspect(fig, path):
        ax = fig.axes[0]
        grid = [item for item in ax.collections if isinstance(item, LineCollection)]
        assert len(grid) == 1
        np.testing.assert_allclose(grid[0].get_colors()[0], [1., 1., 1., .65])
        segments = grid[0].get_segments()
        np.testing.assert_allclose([item[0, 0] for item in segments[:3]], [0., .8, 2.])
        np.testing.assert_allclose([item[0, 1] for item in segments[3:]], [0., 1.2, 2.])
        assert grid[0].get_zorder() < ax.patches[0].get_zorder()
        assert not any(line.get_visible() for line in ax.get_xgridlines())
        fig.canvas.draw()
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(slice_plotter, 'save_figure', inspect)
    plot_exact_slice_from_ellipsoids(
        [dict(x=np.array([1., 1., 0.]), a=.4, b=.3, c=.5)], 0,
        output_path=tmp_path/'grid.svg', xlim=(0, 2), ylim=(0, 2), fluid_grid=True,
        fluid=dict(x=np.array([.4, 1.4]), y=np.array([.6, 1.6]),
                   x_edges=np.array([0., .8, 2.]), y_edges=np.array([0., 1.2, 2.]),
                   values=np.ones((2, 2)), component='magnitude', time=0., counter=0),
    )


def test_direction_arrows_use_unit_in_plane_velocity(tmp_path, monkeypatch):
    from granular_analysis.plotting import slice_plotter
    from matplotlib.quiver import Quiver
    import matplotlib.pyplot as plt

    def inspect(fig, path):
        arrows = [item for item in fig.axes[0].collections if isinstance(item, Quiver)]
        assert len(arrows) == 1
        np.testing.assert_allclose(arrows[0].get_facecolors()[0], [0., 0., 0., 1.])
        np.testing.assert_allclose(arrows[0].get_edgecolors()[0], [0., 0., 0., 1.])
        # Same particle size: equal lengths despite different speeds; w is omitted.
        np.testing.assert_allclose(arrows[0].U, [.7, 0])
        np.testing.assert_allclose(arrows[0].V, [0, -.7])
        assert arrows[0].get_zorder() > fig.axes[0].patches[0].get_zorder()
        fig.canvas.draw()
        plt.close(fig)
        return str(path)

    monkeypatch.setattr(slice_plotter, 'save_figure', inspect)
    particles = [dict(x=np.array([i+1., 1., 0.]), a=.5, b=.5, c=.5, u=v)
                 for i, v in enumerate([[1, 0, 100], [0, -20, 0], [0, 0, 2], [0, 0, 0]])]
    plot_exact_slice_from_ellipsoids(particles, 0, output_path=tmp_path/'directions.svg',
                                   velocity_directions=True)
