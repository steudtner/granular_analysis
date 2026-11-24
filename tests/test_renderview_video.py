import math
import numpy as np
import pytest
from granular_analysis.scripts.run_renderview_video import physical_time_scale, playback_schedule, select_video_frames


def test_physical_conversion_and_real_time_schedule():
    scale = math.sqrt(.000615/9.81)
    assert physical_time_scale() == pytest.approx(scale)
    frames = [(0, 0), (100, 1), (400, 2)]
    indices, times, result = playback_schedule(frames, fps=30)
    assert result == pytest.approx(scale)
    assert indices[0] == 0 and indices[-1] == 2
    assert times[-1] == pytest.approx(400*scale)
    assert len(indices) == math.ceil(400*scale*30)+1
    fast, _, _ = playback_schedule(frames, fps=30, speed=2)
    assert len(fast) < len(indices)
    np.testing.assert_allclose(np.diff(times)[:-1], 1/30)


def test_matching_and_time_filter(monkeypatch, tmp_path):
    from granular_analysis.paths import CasePaths
    from granular_analysis.scripts import run_renderview_video as video
    case = CasePaths.from_input(tmp_path)
    case.trn.mkdir(parents=True)
    folder = case.results / 'fluid'
    folder.mkdir()
    for counter in range(4):
        (case.trn / f'ell_trn_{counter:06d}.bin').touch()
    for counter in (10, 12, 13):
        (folder / f'uvw_{counter:06d}.bin').touch()
    monkeypatch.setattr(video, '_read_bin_time', lambda path: int(path.stem.split('_')[-1]))
    monkeypatch.setattr(video, 'restart_records', lambda path: [dict(t=str(i), counter=str(i+10)) for i in range(4)])
    assert select_video_frames(case, component='magnitude') == [(0, 0), (2, 2), (3, 3)]
    assert select_video_frames(case, component='magnitude', time_start=2) == [(2, 2), (3, 3)]
    with pytest.raises(ValueError, match='at least two'):
        select_video_frames(case, component='magnitude', time_start=3)


def test_video_cli_dispatch(monkeypatch, tmp_path):
    from granular_analysis.scripts import run_all, run_renderview_video
    calls=[]
    monkeypatch.setattr(run_renderview_video, 'run_renderview_video', lambda **kwargs: calls.append(kwargs))
    assert run_all.main(['--only','renderview_video','--path',str(tmp_path),'--out',str(tmp_path/'out'),
                         '--fluid','pressure','--video-view','side','--time-start','0','--time-end','30',
                         '--video-resolution','640','448']) == 0
    assert calls[0]['view'] == 'side'
    assert calls[0]['component'] == 'pressure'
    assert calls[0]['lz'] == 5
    assert calls[0]['fluid_z'] == 0
    assert calls[0]['legend_scale'] == 1.0
    assert calls[0]['elevation'] == 15
    assert calls[0]['window_size'] == (640,448)
    assert calls[0]['time_end'] == 30


@pytest.mark.parametrize('elevation', [0, 15, 30])
def test_camera_elevation_is_measured_above_xz(elevation):
    pytest.importorskip('pyvista')
    from types import SimpleNamespace
    from granular_analysis.plotting.particle_3d_plotter import _set_camera
    plotter = SimpleNamespace(camera=SimpleNamespace(), window_size=(640, 448),
                              renderer=SimpleNamespace(GetViewport=lambda: (0, 0, 1, 1)),
                              reset_camera_clipping_range=lambda: None)
    center = np.array([30., 10., 2.5])
    _set_camera(plotter, 'isometric', center, 120, (0,60,0,20,0,5), elevation=elevation)
    direction = (plotter.camera_position[0] - center) / 120
    assert np.degrees(np.arcsin(direction[1])) == pytest.approx(elevation)
    assert direction[0] == pytest.approx(direction[2])
    assert np.linalg.norm(direction) == pytest.approx(1)


def test_video_volume_and_static_camera(monkeypatch, tmp_path):
    from granular_analysis.paths import CasePaths
    from granular_analysis.scripts import run_renderview_video as video
    from granular_analysis.plotting import particle_3d_plotter
    calls = []
    monkeypatch.setattr(video, 'select_video_frames', lambda *a, **k: [(0., 1), (1., 2)])
    monkeypatch.setattr(video, 'playback_schedule', lambda *a, **k: (np.array([0, 1]), np.array([0., 1.]), 1.))
    monkeypatch.setattr(video, 'find_ffmpeg', lambda: 'ffmpeg')
    monkeypatch.setattr(video, 'load_frame', lambda *a: [])
    monkeypatch.setattr(video, 'load_fluid_slice', lambda *a: pytest.fail('unexpected slice'))
    fluid_loads = []
    def load_volume(*args):
        fluid_loads.append(args)
        return dict(component='q', values=np.array([[[1., 100.]]]),
                    x_edges=np.array([0., 30., 60.]), y_edges=np.array([0., 20.]))
    monkeypatch.setattr(video, 'load_fluid_volume', load_volume)
    monkeypatch.setattr(particle_3d_plotter, 'ellipsoids_to_polydata', lambda *a, **k: None)
    def render(mesh, path, view, **kwargs):
        calls.append(kwargs)
        path.touch()
    monkeypatch.setattr(particle_3d_plotter, 'render_particle_view', render)
    def encode(command, **kwargs):
        from pathlib import Path
        Path(command[-1]).touch()
    monkeypatch.setattr(video.subprocess, 'run', encode)
    case = CasePaths.from_input(tmp_path)
    case.trn.mkdir(parents=True)
    (case.trn / 'ell_trn_000001.bin').touch()
    output = video.run_renderview_video(case=case, out_dir=tmp_path,
        component='q', vortex_threshold=.2, match_renderview=True)
    assert output.exists()
    assert len(calls) == 2
    assert calls[0]['camera_elevation'] is None
    assert calls[0]['legend_scale'] == 1
    assert calls[0]['fluid_z'] == 2.5
    assert calls[0]['fluid_clim'][0] > 0
    assert calls[0]['fluid_clim'] == calls[1]['fluid_clim']
    assert np.isnan(calls[0]['fluid']['values'][0, 0, 0])
    assert len(fluid_loads) == 2
    assert not list(output.parent.glob('fluid-cache-*'))
