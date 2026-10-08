import sys
from types import SimpleNamespace

import numpy as np
import pytest

from granular_analysis.scripts import render_particle_fluid as renderer


def test_front_light_and_projection_share_z_direction(monkeypatch):
    calls={}
    class Plotter:
        def __init__(self,**kwargs):
            calls['window']=kwargs['window_size'];self.camera=SimpleNamespace()
        def set_background(self,*a): pass
        def remove_all_lights(self): calls['cleared']=True
        def add_light(self,light): calls['light']=light
        def add_mesh(self,*a,**kw): pass
        def enable_parallel_projection(self): calls['parallel']=True
        def reset_camera_clipping_range(self): pass
        def screenshot(self,**kwargs):
            calls['camera']=self.camera_position
            calls['scale']=self.camera.parallel_scale
            return np.zeros((2,2,4),dtype=np.uint8)
        def close(self): calls['closed']=True
    monkeypatch.setitem(sys.modules,'pyvista',SimpleNamespace(Plotter=Plotter,Light=lambda **kw:kw))
    renderer.render_front_projection(SimpleNamespace(bounds=(0,45,0,15,0,5)),renderer.DEFAULT_BOUNDS)
    light=calls['light']
    direction=np.subtract(light['focal_point'],light['position'])
    np.testing.assert_allclose(direction/np.linalg.norm(direction),[0,0,-1])
    assert light['positional'] is False
    assert calls['cleared'] and calls['closed'] and calls['parallel']
    assert calls['window']==(2400,1440)
    assert calls['scale']==15
    assert calls['camera'][0]==light['position']


def test_invalid_bounds_rejected_before_loading_data(tmp_path):
    for bounds in [(0,0,0,30),(0,50,30,0),(0,np.nan,0,30),(0,50)]:
        with pytest.raises(ValueError,match='Bounds'):
            renderer.run_particle_fluid(tmp_path,tmp_path/'out',bounds=bounds)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('extra,expected_bounds,expected_component',[
    ([],(0.,50.,0.,30.),'magnitude'),
    (['--fluid-clim','0','0.55'],(0.,50.,0.,30.),'magnitude'),
    (['--xlim','1','40','--ylim','2','20','--fluid','v'],(1.,40.,2.,20.),'v'),
])
def test_pipeline_dispatch_uses_exact_pair_and_common_limits(tmp_path,monkeypatch,extra,expected_bounds,expected_component):
    from granular_analysis.scripts import run_all
    from granular_analysis.io import fluid_slice,loader
    calls={}
    (tmp_path/'results/fluid').mkdir(parents=True)
    (tmp_path/'results/fluid/uvw_000003.bin').touch()
    monkeypatch.setattr(fluid_slice,'latest_common_counter',lambda *a:12)
    monkeypatch.setattr(loader,'_read_bin_time',lambda *a:2.)
    monkeypatch.setattr(fluid_slice,'restart_records',lambda *a:[])
    monkeypatch.setattr(fluid_slice,'matching_record',lambda *a:{'counter':3})
    monkeypatch.setattr(renderer,'run_particle_fluid',lambda *a,**kw:calls.update(args=a,**kw))
    monkeypatch.setattr(run_all,'_run_exact_slice_plot',lambda *a:pytest.fail('Unrequested exact plot'))
    result=run_all.main(['--only','particle_fluid','--path',str(tmp_path),'--out',str(tmp_path/'out'),*extra])
    assert result==0
    assert calls['counter']==12
    assert calls['bounds']==expected_bounds
    assert calls['component']==expected_component
    assert calls['clim']==([0.,.55] if '--fluid-clim' in extra else None)
    assert calls['args'][1]==tmp_path/'out/particle_fluid'
    assert 'particle_fluid' not in run_all.DEFAULT_PIPELINE_NAMES


def test_particle_fluid_time_selects_common_pair(tmp_path,monkeypatch):
    from granular_analysis.scripts import run_all
    from granular_analysis.io import fluid_slice,loader
    calls={}
    (tmp_path/'results/fluid').mkdir(parents=True)
    (tmp_path/'results/fluid/uvw_000003.bin').touch()
    def nearest(case,time,component):
        assert time==90 and component=='magnitude'
        return 89.9,12
    monkeypatch.setattr(fluid_slice,'nearest_frame',nearest)
    monkeypatch.setattr(loader,'_read_bin_time',lambda *a:89.9)
    monkeypatch.setattr(fluid_slice,'restart_records',lambda *a:[])
    monkeypatch.setattr(fluid_slice,'matching_record',lambda *a:{'counter':3})
    monkeypatch.setattr(renderer,'run_particle_fluid',lambda *a,**kw:calls.update(**kw))
    assert run_all.main(['--only','particle_fluid','--time','90','--path',str(tmp_path),'--out',str(tmp_path/'out')])==0
    assert calls['counter']==12


def test_shared_colour_limits_reach_both_figure_fields(tmp_path,monkeypatch):
    from matplotlib.axes import Axes
    records=[]
    original=Axes.pcolormesh
    def record(self,*args,**kwargs):
        if 'vmin' in kwargs:
            records.append((kwargs['vmin'],kwargs['vmax']))
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Axes,'pcolormesh',record)
    monkeypatch.setattr(renderer,'load_fluid_slice',lambda *a:dict(
        x=np.array([0.,1.,2.]),y=np.array([0.,1.,2.]),values=np.full((3,3),.2),time=1.,counter=2))
    monkeypatch.setattr(renderer,'load_frame',lambda *a:[dict(x=np.array([1.,1.,1.]),a=.2,b=.2,c=.2)])
    monkeypatch.setitem(sys.modules,'granular_analysis.plotting.particle_3d_plotter',
                        SimpleNamespace(ellipsoids_to_polydata=lambda *a,**kw:SimpleNamespace(n_points=1)))
    monkeypatch.setattr(renderer,'render_front_projection',lambda *a:np.zeros((2,2,4),dtype=np.uint8))
    monkeypatch.setattr(renderer.plt.Figure,'savefig',lambda *a,**kw:None)
    meta=renderer.run_particle_fluid(tmp_path,tmp_path/'out',counter=1,skip_first=0,clim=(0,.55))
    assert records==[(0.,.55),(0.,.55)]
    assert meta['clim']==[0.,.55]
    assert meta['color_scale_source']=='explicit'


def test_colour_scale_validation(tmp_path):
    from granular_analysis.scripts.run_all import main
    with pytest.raises(ValueError,match='Colour'):
        renderer.run_particle_fluid(tmp_path,tmp_path/'out',clim=(.5,0))
    for args in [ ['--only','particle_fluid','--fluid-clim','1','0'],
                  ['--only','particle_fluid','--fluid-clim','0','nan'],
                  ['--only','porosity_time','--fluid-clim','0','1'] ]:
        with pytest.raises(SystemExit):
            main(args)
