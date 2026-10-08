import pandas as pd
import pytest

from granular_analysis.analysis.porosity2d import surface_metrics
from granular_analysis.scripts.run_2d_analysis import write_surface_summary


def test_surface_extents_and_export(tmp_path, capsys):
    result = surface_metrics([[5, 0], [5, 8], [12, 3], [17, 0]])
    assert result == dict(x_min=5, x_max=17, max_runout_length=12, remaining_height=8)
    write_surface_summary(result, tmp_path, tmp_path / 'ell_trn_000042.bin', 2.5)
    row = pd.read_csv(tmp_path / 'surface_metrics.csv').iloc[0]
    assert row['frame'] == 'ell_trn_000042.bin'
    assert row['max_runout_length'] == 12
    assert row['remaining_height'] == 8
    assert 'Resthöhe' in capsys.readouterr().out


def test_empty_surface_has_no_fabricated_dimensions():
    assert all(value is None for value in surface_metrics(None).values())


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError, match='finite'):
        surface_metrics([[0, 0], [1, float('nan')], [2, 0]])


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('counter', [None, 0])
def test_analysis_exports_initial_and_selected(tmp_path, monkeypatch, cached, counter):
    from granular_analysis.scripts import run_2d_analysis as runner

    initial_frame = tmp_path / 'ell_trn_000000.bin'
    last_frame = tmp_path / 'ell_trn_000042.bin'
    initial_frame.touch()
    last_frame.touch()
    monkeypatch.setattr(runner, 'resolve_data_folder', lambda path: str(tmp_path))
    monkeypatch.setattr(runner, 'load_frame', lambda folder, name: name)
    monkeypatch.setattr(runner, 'load_state', lambda *args, **kwargs: {'porosity': .4} if cached else None)
    monkeypatch.setattr(runner, 'save_state', lambda *args, **kwargs: None)
    calls = []

    def compute(name, **kwargs):
        calls.append((name, kwargs))
        height = 8 if name == initial_frame.name else 3
        return dict(surface_metrics([[5, 0], [5, height], [17, 0]]),
                    porosity=.4, A_ell=1, A_alpha=2, n=3)

    monkeypatch.setattr(runner, 'compute_slice_porosity', compute)
    runner.main(path=str(tmp_path), out=str(tmp_path / 'output'), counter=counter)
    rows = pd.read_csv(tmp_path / 'output' / '2d' / 'surface_metrics.csv')
    assert list(rows.state) == ['initial', 'selected']
    assert list(rows.frame) == [initial_frame.name, initial_frame.name if counter == 0 else last_frame.name]
    assert list(rows.remaining_height) == [8, 8 if counter == 0 else 3]
    assert calls[0][1]['write_upper_edge'] is True
    if counter is None:
        assert len(calls) == 2
        assert calls[1][1]['write_upper_edge'] is False
        for key in ('z0', 'alpha', 'lz', 'skip_first', 'n_bins', 'densify_max_segment_length'):
            assert calls[0][1][key] == calls[1][1][key]
    else:
        assert len(calls) == 1
