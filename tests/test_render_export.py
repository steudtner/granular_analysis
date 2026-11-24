"""The SVG must retain the original render pixels."""
import base64
import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image


def test_mobile_colors_survive_merge_and_clipping():
    pytest.importorskip('pyvista')
    from granular_analysis.plotting.particle_3d_plotter import ellipsoids_to_polydata
    particles = [dict(x=[x, 0, 0], a=a, b=1, c=1, q=[1, 0, 0, 0])
                 for x, a in ((0, 1), (3, 0), (6, 1))]
    mesh = ellipsoids_to_polydata(particles, nu=8, nv=6, mobile_start=2)
    colors = mesh.cell_data['particle_rgb']
    centers = mesh.cell_centers().points[:, 0]
    assert (colors[centers < 2] == [128, 128, 128]).all()
    assert (colors[centers > 4] == [220, 30, 30]).all()
    clipped = mesh.clip_box(bounds=(5, 7, -2, 2, -2, 2), invert=False)
    assert (clipped.cell_data['particle_rgb'] == [220, 30, 30]).all()
    plain = ellipsoids_to_polydata(particles, nu=8, nv=6)
    assert 'particle_rgb' not in plain.cell_data


def test_top_includes_full_particle_surfaces_and_projection(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    pv = pytest.importorskip('pyvista')
    from granular_analysis.plotting import particle_3d_plotter as renderer
    captured = {}
    class Plotter:
        def __init__(self, **kwargs):
            self.camera = SimpleNamespace(parallel_scale=40)
        def set_background(self, *args):
            pass
        def add_mesh(self, *args, **kwargs):
            pass
        def screenshot(self, path):
            return np.zeros((100, 200, 3), dtype=np.uint8)
        def close(self):
            pass
    def camera(plotter, view, center, distance, bounds):
        captured['bounds'] = bounds
    monkeypatch.setattr(pv, 'Plotter', Plotter)
    monkeypatch.setattr(renderer, '_set_camera', camera)
    mesh = pv.Box(bounds=(-1, 101, -1, 21, -1, 6))
    target = tmp_path / 'top.png'
    renderer.render_particle_view(mesh, target, 'top', export_document=False)
    np.testing.assert_allclose(captured['bounds'], mesh.bounds)
    metadata = json.loads(target.with_suffix('.projection.json').read_text())
    np.testing.assert_allclose(metadata['extent'], [-30, 130, 42.5, -37.5])


def test_svg_preserves_render_resolution(monkeypatch, tmp_path):
    pv = pytest.importorskip('pyvista')
    from granular_analysis.plotting import particle_3d_plotter, plotstyle

    pixels = np.zeros((140, 200, 3), dtype=np.uint8)
    class Plotter:
        def __init__(self, **kwargs):
            pass
        def set_background(self, *args):
            pass
        def add_mesh(self, *args, **kwargs):
            pass
        def screenshot(self, path):
            Image.fromarray(pixels).save(path)
            return pixels
        def close(self):
            pass
    monkeypatch.setattr(pv, 'Plotter', Plotter)
    monkeypatch.setattr(particle_3d_plotter, '_set_camera', lambda *args: None)
    monkeypatch.setattr(plotstyle, 'check_tex', lambda: None)
    monkeypatch.setattr(plotstyle, 'prepare_figure', lambda fig: None)
    output = tmp_path / 'render.png'
    particle_3d_plotter.render_particle_view(pv.PolyData(), output, 'side')
    root = ET.parse(output.with_suffix('.svg')).getroot()
    embedded = root.find('.//{http://www.w3.org/2000/svg}image')
    href = embedded.attrib['{http://www.w3.org/1999/xlink}href']
    image = Image.open(io.BytesIO(base64.b64decode(href.split(',', 1)[1])))
    assert image.size == (200, 140)


@pytest.mark.parametrize("component,values", [("q", [-1, 0, 1, 100]), ("lambda2", [1, 0, -1, -100])])
def test_log_vortex_colors(component, values):
    from granular_analysis.plotting.particle_3d_plotter import _vortex_log_colors
    strength, limits = _vortex_log_colors(values, component)
    assert np.isnan(strength[:2]).all()
    np.testing.assert_array_equal(strength[2:], [1, 100])
    assert limits == (1, 100)


def test_log_vortex_empty_and_constant():
    from granular_analysis.plotting.particle_3d_plotter import _vortex_log_colors
    for values in ([0, -1], [2, 2]):
        _, (low, high) = _vortex_log_colors(values, "q")
        assert 0 < low < high
