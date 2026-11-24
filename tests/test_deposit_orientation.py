import numpy as np
import pytest
from granular_analysis.analysis.deposit_orientation import deposit_orientation


def particle(x=0, **updates):
    return dict(dict(x=[x, 1, 0], a=2, b=1, c=1, q=[1., 0, 0, 0], t=4), **updates)


def test_shape_axes_and_stationary_particles():
    particles = [particle(), particle(a=2, b=2, c=.5, u=[0, 0, 0]),
                 particle(a=2, b=2, c=.5), particle(a=3, b=1, c=1)]
    table, hist, _, _ = deposit_orientation(particles, n_bins=1)
    assert table['mode'].tolist() == ['prolate', 'oblate', 'oblate', 'prolate']
    assert table.axis.tolist() == ['a', 'c', 'c', 'a']
    np.testing.assert_allclose(table[['angle_x_deg', 'angle_y_deg', 'angle_z_deg']],
                               [[0, 90, 90], [90, 90, 0], [90, 90, 0], [0, 90, 90]])
    for _, group in hist.groupby(['mode', 'bin', 'axis']):
        assert group.fraction.sum() == pytest.approx(1)
        assert group['count'].sum() == 2


def test_rotations_sign_and_no_input_mutation():
    q = np.array([1., 0, 0, 1.])
    table, _, _, _ = deposit_orientation([particle(q=q), particle(q=[0, 0, 0, 1])])
    np.testing.assert_allclose(table.iloc[0][['angle_x_deg', 'angle_y_deg', 'angle_z_deg']].astype(float),
                               [90, 0, 90], atol=2e-6)
    assert table.iloc[1].angle_x_deg == 0
    np.testing.assert_array_equal(q, [1, 0, 0, 1])


def test_bin_boundaries_filtering_and_empty_bins():
    table, hist, edges, _ = deposit_orientation([particle(-1), particle(0), particle(1), particle(2), particle(3)],
                                               n_bins=4, xlim=(0, 2), ylim=(0, 2))
    assert table.bin.tolist() == [0, 2, 3]
    np.testing.assert_allclose(edges, [0, .5, 1, 1.5, 2])
    empty = hist[(hist['mode'] == 'prolate') & (hist.bin == 1)]
    assert empty.fraction.isna().all()
    assert (empty['count'] == 0).all()


def test_spheres_triaxial_missing_and_tied_axes():
    particles = [particle(a=1), particle(a=3, b=2), particle(q=None)]
    particles[2].pop('q')
    table, _, _, _ = deposit_orientation(particles)
    assert table.status.tolist() == ['undefined_shape_axis', 'undefined_shape_axis', 'missing_orientation']
    table, _, _, _ = deposit_orientation([particle(a=2, b=2, c=1)], mode='prolate')
    assert table.status.iloc[0] == 'ambiguous_axis'
    assert table.angle_x_deg.isna().all()


def test_skipping_and_partial_angle_bins():
    table, hist, _, angles = deposit_orientation([particle(0), particle(1)], skip_first=1, angle_step=17)
    assert table.particle_row.tolist() == [2]
    np.testing.assert_allclose(angles, [0, 17, 34, 51, 68, 85, 90])
    assert hist[(hist['mode'] == 'prolate') & (hist.axis == 'y')]['count'].sum() == 1


@pytest.mark.parametrize('kwargs', [dict(n_bins=0), dict(angle_step=0), dict(angle_step=np.nan),
                                    dict(xlim=(2, 1)), dict(skip_first=-1)])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        deposit_orientation([particle()], **kwargs)


def test_invalid_orientation():
    with pytest.raises(ValueError, match='quaternion'):
        deposit_orientation([particle(q=[0, 0, 0, 0])])
    with pytest.raises(ValueError, match='rotation'):
        deposit_orientation([particle(R=np.diag([1, 2, 1]))])


def test_routine_exports_and_last_frame(monkeypatch, tmp_path):
    import json
    from granular_analysis.scripts import run_deposit_orientation as routine
    from granular_analysis.paths import CasePaths
    case = CasePaths.from_input(tmp_path)
    case.trn.mkdir(parents=True)
    (case.trn / "ell_trn_000042.bin").touch()
    monkeypatch.setattr(routine, 'latest_trn_counter', lambda case: 42)
    monkeypatch.setattr(routine, 'load_frame', lambda *args: [particle(0), particle(1)])
    calls = []
    monkeypatch.setattr(routine, 'plot_orientation_histograms', lambda *args: calls.append(args))
    monkeypatch.setattr(routine, 'plot_slice_y_insets', lambda *args, **kwargs: None)
    routine.run_deposit_orientation(case=case, out_dir=tmp_path / 'out', skip_first=0)
    folder = tmp_path / 'out' / 'deposit_orientation'
    prefix = folder / 'deposit_orientation_auto_000042'
    assert prefix.with_suffix('.json').exists()
    metadata = json.loads(prefix.with_suffix('.json').read_text())
    assert metadata['counter'] == 42
    assert metadata['time'] == 4
    assert metadata['skip_first_rows'] == 0
    assert len(metadata['x_edges']) == 7
    assert len(calls) == 1
    assert (folder / (prefix.name + '_particles.csv')).exists()
    assert (folder / (prefix.name + '_histograms.csv')).exists()


def test_cli_dispatch(monkeypatch, tmp_path):
    from granular_analysis.scripts import run_all
    from granular_analysis.scripts import run_deposit_orientation as routine
    calls = []
    monkeypatch.setattr(routine, 'run_deposit_orientation', lambda **kwargs: calls.append(kwargs))
    assert run_all.main(['--only', 'deposit_orientation', '--path', str(tmp_path),
                         '--out', str(tmp_path / 'out'), '--counter', '42',
                         '--orientation-mode', 'oblate', '--orientation-bins', '3',
                         '--orientation-angle-step', '15', '--skip-points', '0', '--xlim', '0', '30']) == 0
    assert calls[0]['counter'] == 42
    assert calls[0]['mode'] == 'oblate'
    assert calls[0]['n_bins'] == 3
    assert calls[0]['angle_step'] == 15
    assert calls[0]['xlim'] == [0, 30]
    assert calls[0]['skip_first'] == 0


def test_named_axes_are_not_replaced_by_other_semiaxes():
    table, _, _, _ = deposit_orientation([particle(a=1, b=3, c=1), particle(a=2, b=.5, c=2)])
    assert table.axis.tolist() == ['a', 'c']
    assert table.status.tolist() == ['ambiguous_axis', 'ambiguous_axis']


@pytest.mark.parametrize('combined', [False, True])
def test_cut_insets_use_y_angle_and_full_bin_population(monkeypatch, tmp_path, combined):
    from granular_analysis.scripts import run_deposit_orientation as routine
    from granular_analysis.plotting import plotstyle
    particles = [dict(particle(0), x=[0, 1, 0]), dict(particle(1, q=[1., 0, 0, 1.]), x=[1, 1, 2.5])]
    table, hist, edges, angle_edges = deposit_orientation(particles, n_bins=1, xlim=(-3, 4))
    style = dict(plotstyle.PLOT_STYLE, **{'text.usetex': False})
    monkeypatch.setattr(plotstyle, 'PLOT_STYLE', style)
    calls = []
    def inspect(fig, path, **kwargs):
        ax = fig.axes[-1]
        if combined:
            assert len(fig.axes[0].images) == 1
            render_ax = fig.axes[0]
            np.testing.assert_allclose(render_ax.get_xlim(), ax.get_xlim())
            np.testing.assert_allclose(render_ax.get_position().width, ax.get_position().width)
            assert render_ax.get_position().y1 < ax.get_position().y0
            assert render_ax.get_ylabel() == r'$z / D_{\mathrm{Eq}}$'
            assert ax.get_xlabel() == ''
        # Only one ellipsoid intersects z=2.5; its selected axis is vertical.
        np.testing.assert_allclose(ax.collections[0].get_array(), [0], atol=2e-6)
        assert ax.get_aspect() == 1
        assert not any('z=' in text.get_text() or 't=' in text.get_text() for text in ax.texts)
        assert len(ax.child_axes) == 1
        inset = ax.child_axes[0]
        assert len(inset.lines) == 0
        np.testing.assert_array_equal(inset.get_xticks(), [0, 45, 90])
        # Both particles belong to the spatial bin, including the off-slice one.
        assert sum(patch.get_height() for patch in inset.patches) == pytest.approx(1)
        assert inset.patches[0].get_height() == pytest.approx(.5)
        assert inset.patches[-1].get_height() == pytest.approx(.5)
        heading = next(text for text in inset.texts if text.get_text() == r'$N_{\mathrm{Mob}}=2$')
        assert heading.get_position()[1] > 1
        assert heading.get_horizontalalignment() == 'center'
        assert not any('Bin ' in text.get_text() for text in inset.texts)
        suffix = '_renderview_combined' if combined else ''
        assert str(path).endswith(f'_prolate_cut_z2.50_y_insets{suffix}.png')
        calls.append(path)
    monkeypatch.setattr(plotstyle, 'align_slice_legends', lambda *args: None)
    monkeypatch.setattr(plotstyle, 'export_figure', inspect)
    render_path = None
    if combined:
        from PIL import Image
        render_path = tmp_path / 'top.png'
        Image.new('RGB', (100, 20), 'red').save(render_path)
        render_path.with_suffix('.projection.json').write_text('{"extent": [-3, 4, 5, 0]}')
    routine.plot_slice_y_insets(particles, table, hist, edges, angle_edges,
                               tmp_path / 'test', z_plane=2.5, lz=5, renderview_image=render_path)
    assert len(calls) == 1
