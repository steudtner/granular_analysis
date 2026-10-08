import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
from scipy.io import FortranFile

from granular_analysis.analysis import porosity_time as analysis
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.scripts import run_porosity_time as runner


def particle(x,y,z,r=.35):
    return dict(x=np.array([x,y,z],dtype=float),q=np.array([1.,0,0,0]),a=r,b=r,c=r)


def write_frame(folder,counter,time,particles):
    folder.mkdir(parents=True,exist_ok=True)
    with FortranFile(folder/f'ell_trn_{counter:06d}.bin','w') as stream:
        stream.write_record(np.array([len(particles)],dtype='i4'))
        stream.write_record(np.array([time],dtype='f8'))
        for p in particles:
            stream.write_record(p['x'])
            stream.write_record(np.zeros(3))
            stream.write_record(p['q'])
            for name in ('a','b','c'):
                stream.write_record(np.array([p[name]],dtype='f8'))
            stream.write_record(np.array([0],dtype='i4'))


def test_sampling_uses_physical_time_and_handles_restart_duplicates():
    frames=[(1,0.),(2,.1),(3,.2),(4,5.),(5,10.),(6,5.)]
    assert analysis.select_time_samples(frames,3)==[(1,0.),(6,5.),(5,10.)]
    assert analysis.select_time_samples(frames,0,1.,9.)==[(6,5.)]
    assert analysis.select_time_samples(frames,1)==[(5,10.)]
    assert len(analysis.select_time_samples([(1,0),(2,1)],15))==2
    with pytest.raises(ValueError,match='No particle'):
        analysis.select_time_samples(frames,3,20,30)
    with pytest.raises(ValueError):
        analysis.select_time_samples(frames,3,5,1)
    with pytest.raises(ValueError):
        analysis.select_time_samples([(1,np.nan)])


def test_uniform_periodic_grid_has_no_double_endpoint():
    z=analysis.periodic_z_grid(1.,.3)
    np.testing.assert_allclose(z,[0,.25,.5,.75])
    assert z[-1] < 1.
    for lz,step in [(0,.1),(1,0),(1,np.nan),(np.inf,.1),(1,1)]:
        with pytest.raises(ValueError):
            analysis.periodic_z_grid(lz,step)


def test_mean_std_is_spatial_population_not_temporal():
    data=pd.DataFrame(dict(counter=[1]*3+[2]*3,time=[0.]*3+[10.]*3,
                           z=[0,1,2]*2,porosity=[.2,.4,.6,.7,.7,.7]))
    summary=analysis.summarize_porosity(data)
    np.testing.assert_allclose(summary.porosity_mean,[.4,.7])
    np.testing.assert_allclose(summary.porosity_std,[np.sqrt(2*.2**2/3),0],atol=1e-15)
    assert summary.n_valid.tolist()==[3,3]
    data.loc[0,'porosity']=np.nan
    with pytest.warns(RuntimeWarning,match='invalid slices'):
        summary=analysis.summarize_porosity(data)
    assert np.isnan(summary.porosity_mean.iloc[0])
    assert summary.n_valid.tolist()==[2,3]
    data.loc[0,'porosity']=-.1
    with pytest.warns(RuntimeWarning):
        assert np.isnan(analysis.summarize_porosity(data).porosity_std.iloc[0])


def test_prefilter_preserves_existing_periodic_porosity_and_fixed_exclusion():
    particles=[particle(100,100,0)]+[particle(x,y,z) for x in (0,1) for y in (0,1) for z in (.1,.9)]
    zs=[0.,.2,.8]
    profile=analysis.compute_porosity_profile(particles,zs,alpha=0,lz=1,skip_first=1)
    for i,z in enumerate(zs):
        expected=compute_slice_porosity(particles,z,0,1,skip_first=1)
        assert profile.A_alpha.iloc[i]==pytest.approx(expected['A_alpha'])
        assert profile.A_ell.iloc[i]==pytest.approx(expected['A_ell'])
        assert profile.porosity.iloc[i]==pytest.approx(expected['porosity'])
    empty=analysis.compute_porosity_profile([],zs,alpha=.6,lz=1)
    assert empty.porosity.isna().all()
    assert not empty.valid.any()


def test_real_binary_workflow_exports_and_reuses_only_matching_cache(tmp_path,monkeypatch):
    case=tmp_path/'case';folder=case/'results/ellipsoid/ellipsoid_trn'
    particles=[particle(x,y,z) for x in (0,1) for y in (0,1) for z in (.2,.7)]
    for counter,time in [(1,0.),(2,2.),(3,4.)]:
        write_frame(folder,counter,time,particles)
    (case/'input.dat').write_text('Nell_fixed=0,\n')
    assert analysis.discover_frame_times(folder)==[(1,0.),(2,2.),(3,4.)]
    output=tmp_path/'out'
    profiles,summary=runner.run_porosity_time(case,output,n_frames=3,lz=1,z_step=.5,alpha=0)
    assert len(profiles)==6 and summary.complete.all()
    for stem in ('porosity_profiles','porosity_mean_std'):
        for suffix in ('png','pdf','svg'):
            assert (output/f'{stem}.{suffix}').stat().st_size > 0
    meta=json.loads((output/'porosity_time.json').read_text())
    assert len(meta['plotted_profiles'])==3
    assert meta['z']==[0,.5]
    assert meta['skip_first']==0
    assert 'ddof=0' in meta['statistics']
    calls=[]
    original=runner.compute_porosity_profile
    def count_calls(*args,**kwargs):
        calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'compute_porosity_profile',count_calls)
    monkeypatch.setattr(runner,'plot_porosity_time',lambda *a,**kw: [])
    _,cached=runner.run_porosity_time(case,output,n_frames=3,lz=1,z_step=.5,alpha=0)
    assert calls==[]
    np.testing.assert_allclose(summary.porosity_mean,cached.porosity_mean)
    runner.run_porosity_time(case,output,n_frames=3,lz=1,z_step=.5,alpha=.1)
    assert len(calls)==3  # Changing alpha must not reuse an old profile.
    calls.clear()
    runner.run_porosity_time(case,output,n_frames=3,lz=1,z_step=.5,alpha=.1,force=True)
    assert len(calls)==3


def test_cli_dispatch_and_ambiguous_time_rejection(tmp_path,monkeypatch):
    from granular_analysis.scripts import run_all
    calls={}
    monkeypatch.setattr(runner,'run_porosity_time',lambda *a,**kw:calls.update(args=a,**kw))
    result=run_all.main(['--only','porosity_time','--path',str(tmp_path),'--out',str(tmp_path/'out'),
                         '--porosity-frames','30','--porosity-profiles','15',
                         '--porosity-z-step','.25','--lz','5','--time-start','10','--time-end','90'])
    assert result==0
    assert calls['n_frames']==30 and calls['n_profiles']==15
    assert calls['time_start']==10 and calls['time_end']==90
    assert calls['args'][1]==tmp_path/'out/porosity_time'
    assert 'porosity_time' not in run_all.DEFAULT_PIPELINE_NAMES
    with pytest.raises(SystemExit):
        run_all.main(['--only','porosity_time','--counter','5'])
    dry=tmp_path/'dry'
    assert run_all.main(['--only','porosity_time','--dry','--out',str(dry)])==0
    assert not dry.exists()
