"""Scientific definitions and refusal of incomplete or ambiguous measurements."""
import numpy as np
import pandas as pd
import pytest
from scipy.io import FortranFile
from granular_analysis.analysis.collapse import (
    CollapseOptions,read_frame,upper_edge,quasistatic_time,t95,regress_surface,
    orientation_statistics,contact_statistics,grid_comparison,dry_scaling,
    dissipation_from_complete_budget,pressure_reference_statistics,snapshot_monitor,
    _energy_particle,choose_frames,
)


def test_trn_vector_reader_and_record_corruption(tmp_path):
    path=tmp_path/'ell_trn_000001.bin'
    with FortranFile(path,'w') as f:
        f.write_record(np.array([1],dtype='i4'));f.write_record(np.array([2.5]))
        f.write_record(np.array([1.,2.,3.]));f.write_record(np.array([4.,5.,6.]))
        f.write_record(np.array([1.,0.,0.,0.]))
        for a in (2.,1.,.5):f.write_record(np.array([a]))
        f.write_record(np.array([42],dtype='i4'))
    time,data=read_frame(path)
    assert time==2.5
    np.testing.assert_array_equal(data['u'][0],[4,5,6]);assert data['a'][0]==2
    raw=bytearray(path.read_bytes());raw[28:32]=np.array([100],dtype='i4').tobytes();path.write_bytes(raw)
    with pytest.raises(ValueError,match='record markers'):read_frame(path)


def test_upper_edge_uses_euclidean_distance():
    coords=np.array([[0,0],[1.5,1.5],[2.,.1],[0,-1],[2,-1]])
    edge=upper_edge(coords,n_bins=3,max_distance=2.)
    assert not any(np.allclose(row,[1.5,1.5]) for row in edge)


def test_quasistatic_holds_through_tail_and_rejects_gaps():
    t=np.arange(20.);energy=np.r_[0,100,np.ones(18)*.1]
    assert quasistatic_time(t,energy)==2
    energy[-2]=10;assert np.isnan(quasistatic_time(t,energy))
    energy[-2]=np.nan;assert quasistatic_time(t,energy)==2
    energy[3]=np.nan;assert quasistatic_time(t,energy)==4
    with pytest.raises(ValueError):quasistatic_time([0,1,1,2],[0,1,0,0])


def test_t95_uses_initial_and_final_advance():
    assert t95([0,1,2,3],[30,40,49.1,50])==2
    assert np.isnan(t95([0,1],[30,30]))


def test_piecewise_scores_include_parameter_and_variance_counts():
    x=np.linspace(0,10,100);y=15-2*np.maximum(0,x-4)+.001*np.sin(x)
    best,scores,edge=regress_surface(np.column_stack((x,y)))
    assert set(scores.n_sections)=={1,2,3,4} and scores.selected.sum()==1
    k=scores.n_parameters.to_numpy();n=scores.n_points.to_numpy()
    np.testing.assert_allclose(scores.AIC_c,scores.AIC+2*k*(k+1)/(n-k-1))
    assert (scores.selection_criterion=='AIC_c').all()
    assert np.min(abs(best.breakpoints-4))<.1


def test_orientation_uses_oblate_normal_and_prolate_long_axis():
    dtype=np.dtype([('x',float,(3,)),('u',float,(3,)),('q',float,(4,)),('a',float),('b',float),('c',float)])
    data=np.zeros(1,dtype=dtype);data['q']=[1,0,0,0];data['a']=2;data['b']=1;data['c']=.5
    assert orientation_statistics(data,'Prolat').angle_to_flow.iloc[0]==0
    assert orientation_statistics(data,'Oblat').angle_to_flow.iloc[0]==90
    assert np.isnan(orientation_statistics(data,'Kugel').angle_to_flow.iloc[0])


def test_contact_pairs_are_unique_and_include_fixed_bed(tmp_path):
    path=tmp_path/'ell_coll_000001_3.bin'
    with FortranFile(path,'w') as f:
        f.write_record(np.array([3],dtype='i4'));f.write_record(np.array([1.],dtype='f4'));f.write_record(np.array([.1],dtype='f4'))
        for first,second,impulse in ((1,3,1),(3,1,1),(3,4,0)):
            for i in (first,second):f.write_record(np.array([i],dtype='i4'))
            for _ in range(6):f.write_record(np.zeros(3,dtype='f4'))
            for _ in range(3):f.write_record(np.array([impulse,0,0],dtype='f4'))
    _,counts,pairs=contact_statistics(path,2,2)
    assert pairs=={(1,3)} and counts.contact_count.tolist()==[1,0]


def test_two_grid_differences_no_order_and_no_divide_by_zero():
    table=grid_comparison({'l':1,'h':0},{'l':1.1,'h':.1},metrics=('l','h'))
    assert table.relative_difference.iloc[0]==pytest.approx(.1)
    assert np.isnan(table.relative_difference.iloc[1]) and 'order' not in table


def test_complete_budget_requires_work_and_preserves_balance():
    np.testing.assert_allclose(dissipation_from_complete_budget(10,[9,8],[0,1]),[1,3])
    with pytest.raises(ValueError):dissipation_from_complete_budget(10,[np.nan],[0])


def test_same_height_pressure_subtracts_gauge_and_hydrostatics():
    x=np.array([1.,20.,90.,95.]);y=np.array([1.,2.])
    field=np.array([[101.,101.,100.,100.],[201.,201.,200.,200.]])
    a=pressure_reference_statistics(x,y,field,30);b=pressure_reference_statistics(x,y,field+500,30)
    assert a==b and a['excess_pressure_mean']==1


def test_restart_monitor_prefix_hashes_exact_bytes(tmp_path):
    (tmp_path/'history').mkdir();file=tmp_path/'ell_0001_monitor.dat'
    rows=np.zeros((3,11));rows[:,0]=[1,2,3];rows[:,1]=[.1,.2,.3]
    np.savetxt(tmp_path/'history'/file.name,rows)
    current=rows[1:].copy();current[:,5]=2;np.savetxt(file,current)
    data,sources,discarded=snapshot_monitor(file)
    assert len(sources)==2 and discarded==2
    np.testing.assert_array_equal(data[:,5],[0,2,2])
    _,tr,rot,_,_=_energy_particle((1,file,2,np.ones(3),np.array([0,.1,.2,.4])))
    assert np.isnan(tr[[0,3]]).all() and tr[2]==4 and rot[1]==0


def test_sampling_preserves_endpoints_and_constraints():
    selected=choose_frames(np.geomspace(.001,500,100),10)
    assert selected[0]==0 and selected[-1]==99 and len(selected)<=10
    np.testing.assert_array_equal(choose_frames(np.arange(10),0),np.arange(10))
    with pytest.raises(ValueError):CollapseOptions(z_slices=5)
    with pytest.raises(ValueError):CollapseOptions(rotation_share_threshold=.4)


def test_dry_shapes_are_proportional_not_absolute_amplitudes():
    a=dry_scaling([1,3,6]);assert a.planar_delta_l_shape.iloc[2]==pytest.approx(3*2**(2/3))
    with pytest.raises(ValueError):dry_scaling([0])


def test_porosity_clips_solid_to_control_volume(monkeypatch):
    from granular_analysis.analysis import porosity2d
    outer=np.array([[0.,0.],[1,0],[1,1],[0,1],[0,0]])
    monkeypatch.setattr(porosity2d,'compute_alpha_shape_2d',lambda *a,**kw:(outer,1.))
    particles=[dict(x=np.array([x,.5,.5]),q=np.array([1.,0,0,0]),a=.4,b=.4,c=.4) for x in (.5,10.)]
    result=porosity2d.compute_slice_porosity(particles,.5,.6,5,clip_solid_to_shape=True)
    # Only the circle inside the square contributes; the distant one is clipped.
    # The 720-point contour approximates the analytic circle area.
    area=np.pi*.4**2
    assert result['porosity']==pytest.approx(1-area,abs=1e-5)


def test_grid_comparison_refuses_different_forms():
    with pytest.raises(ValueError,match='same particle form'):
        grid_comparison({'form':'Prolat'},{'form':'Oblat'})


def test_pressure_statistics_handles_missing_control_region():
    with pytest.raises(ValueError,match='outside exported field'):
        pressure_reference_statistics([0,1],[1,2],np.zeros((2,2)),30)


def test_completed_case_pipeline_keeps_inputs_and_marks_unavailable_terms(tmp_path):
    from granular_analysis.analysis.collapse import analyze_case
    case=tmp_path/'case';case.mkdir();out=tmp_path/'new_analysis'
    coords=[(x,y,z) for x in (.5,2.,3.5) for y in (.5,2.,3.5) for z in (0.,.5)]
    n=len(coords)
    config=f'grx=0.d0\ngry=-0.06d0\ngrz=0.d0\nrho_f=1000\nNell={n}\nNell_fixed=0\nlenz=1\n'
    (case/'input.dat').write_text(config);(case/'main.def').write_text('fixture\n')
    (case/'input_ellipsoid_data.dat').write_text(''.join(f'{i},2500,.4,.4,.4\n' for i in range(1,n+1)))
    folder=case/'results'/'ellipsoid';trn=folder/'ellipsoid_trn';trn.mkdir(parents=True)
    for counter,time in enumerate((0.,1.,2.,3.),1):
        with FortranFile(trn/f'ell_trn_{counter:06d}.bin','w') as f:
            f.write_record(np.array([n],dtype='i4'));f.write_record(np.array([time]))
            for x,y,z in coords:
                f.write_record(np.array([x+.02*time,y,z]));f.write_record(np.array([.02,0,0]))
                f.write_record(np.array([1.,0,0,0]))
                for _ in range(3):f.write_record(np.array([.4]))
                f.write_record(np.array([1],dtype='i4'))
    for number,(x,y,z) in enumerate(coords,1):
        rows=np.zeros((4,11));rows[:,0]=np.arange(1,5);rows[:,1]=np.arange(4)
        rows[:,2]=x+.02*np.arange(4);rows[:,3]=y;rows[:,4]=z;rows[:,5]=.02
        np.savetxt(folder/f'ell_{number:04d}_monitor.dat',rows)
    before=(case/'input.dat').read_bytes()
    options=CollapseOptions(frames=4,z_slices=4,workers=1,l0=4,h0=4)
    status=analyze_case('Prolat',case,out,options)
    assert (case/'input.dat').read_bytes()==before
    assert status['n_native_frames']==4 and not status['contact_available']
    geometry=pd.read_csv(out/'geometry_timeseries.csv')
    assert len(geometry)==4 and geometry.E_kin_tot.notna().all()
    assert geometry.t.iloc[-1]==pytest.approx(.9)
    assert not status['end']['quasistatic']
    assert (out/'herkunft.json').exists() and (out/'porosity_bins_end.csv').exists()
    budget=pd.read_csv(out/'energy_budget.csv');assert budget.Dissipation.isna().all()
    assert status['pressure_samples']==0


def test_initial_sinking_indicator_integrates_and_never_extrapolates():
    from granular_analysis.analysis.collapse import initial_settling_indicator
    value=initial_settling_indicator([0,2,6],[0,-.2,-.6],'Prolat',window=5)
    assert value['mean_initial_sink_over_us']==pytest.approx(.25)
    assert value['ratio_to_single_particle_estimate']==pytest.approx(.25/.729)
    missing=initial_settling_indicator([1,2],[-.1,-.2],'Oblat')
    assert np.isnan(missing['mean_initial_sink_over_us'])
