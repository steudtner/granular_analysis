import numpy as np
import pytest
from granular_analysis.analysis.particle_shape import compute_zingg_ratios, compute_corey_shape_factor, compute_wadell_sphericity, read_particle_axes
from granular_analysis.plotting.zingg import plot_zingg


def test_ratios_sort_axes_and_ignore_size():
    axes = [[1, 1, 1], [4, 1, 1], [1, 4, 4], [1, 2, 4], [10, 20, 40]]
    x, y = compute_zingg_ratios(axes)
    np.testing.assert_allclose(x, [1, .25, 1, .5, .5])
    np.testing.assert_allclose(y, [1, 1, .25, .5, .5])


@pytest.mark.parametrize('axes', [[], [[1, 2]], [[0, 1, 1]], [[-1, 1, 1]], [[np.nan, 1, 1]], [[np.inf, 1, 1]]])
def test_invalid_axes_rejected(axes):
    with pytest.raises(ValueError): compute_zingg_ratios(axes)
    with pytest.raises(ValueError): compute_corey_shape_factor(axes)
    with pytest.raises(ValueError): compute_wadell_sphericity(axes)


def test_input_keeps_every_row_and_id(tmp_path):
    path = tmp_path / 'input_ellipsoid_data.dat'
    path.write_text('id,rho,a,b,c\n7,2500,1,1,1\n8,2500,4,1,1\n9,2500,1,4,4\n10,2.5D3,1,2,4\n')
    table = read_particle_axes(path)
    assert table.particle_id.tolist() == [7,8,9,10]
    assert table.input_line.tolist() == [2,3,4,5]
    np.testing.assert_allclose(table.Sf, [1, .5, .25, 1/np.sqrt(8)])
    assert table.zingg_class.tolist() == ['equant','rod','disk','blade']
    result = plot_zingg(table, tmp_path/'zingg.svg')
    assert '<svg' in result.read_text()
    assert 'Corey Sf' in result.read_text()


@pytest.mark.parametrize('row', ['1,2500,1,0,1', '1,2500,1', '1,2500,bad,1,1', '1.5,2500,1,1,1'])
def test_bad_numeric_row_not_silently_skipped(tmp_path, row):
    path = tmp_path/'input.dat';path.write_text(row)
    with pytest.raises(ValueError, match='line 1'): read_particle_axes(path)


def test_input_only_cli_includes_rows_beyond_nell(tmp_path):
    from granular_analysis.scripts.run_all import main
    (tmp_path/'input.dat').write_text('&ellipsoids\n Nell=1,\n Nell_fixed=1,\n/\n')
    (tmp_path/'input_ellipsoid_data.dat').write_text('1,2500,1,1,1\n2,2500,4,1,1\n')
    assert main(['--only','particle_shape','--skip-particles','0','--path',str(tmp_path),'--out',str(tmp_path/'out')]) == 0
    import pandas as pd
    table = pd.read_csv(tmp_path/'out/particle_shape/particle_shape.csv')
    assert table.particle_id.tolist() == [1,2]
    np.testing.assert_allclose(table.Sf, [1, .5])
    assert table.wadell_sphericity.iloc[0] == pytest.approx(1)
    assert 0 < table.wadell_sphericity.iloc[1] < 1
    assert (tmp_path/'out/particle_shape/zingg_wadell.svg').is_file()
    assert (tmp_path/'out/particle_shape/zingg.svg').is_file()


def test_corey_sorted_axes_scale_and_extreme_lengths():
    axes = np.array([[1, 1, 1], [4, 1, 1], [4, 4, 1], [4, 2, 1], [1, 4, 2]])
    expected = [1, .5, .25, 1/np.sqrt(8), 1/np.sqrt(8)]
    for scale in [1, 1e200, 1e-200]:
        np.testing.assert_allclose(compute_corey_shape_factor(axes * scale), expected)


def test_wadell_matches_closed_form_spheroids():
    # Independent elementary surface formulas for prolate and oblate shapes.
    e = np.sqrt(1 - 1/16)
    prolate_area = 2*np.pi * (1 + 4*np.arcsin(e)/e)
    oblate_area = 2*np.pi*16 * (1 + np.arctanh(e)/(16*e))
    expected = [1, 4*np.pi*4**(2/3)/prolate_area, 4*np.pi*16**(2/3)/oblate_area]
    np.testing.assert_allclose(compute_wadell_sphericity([[1,1,1],[4,1,1],[4,4,1]]), expected, rtol=1e-13)


def test_wadell_triaxial_surface_integral_and_scale():
    # Integrate the cross-product surface element over an octant.
    from scipy.integrate import dblquad
    a,b,c = 4.,2.,1.
    octant, _ = dblquad(lambda phi,theta: np.sin(theta)*np.sqrt(
        (b*c*np.sin(theta)*np.cos(phi))**2 +
        (a*c*np.sin(theta)*np.sin(phi))**2 + (a*b*np.cos(theta))**2),
        0, np.pi/2, lambda _:0, lambda _:np.pi/2, epsabs=1e-10, epsrel=1e-10)
    expected = 4*np.pi*(a*b*c)**(2/3)/(8*octant)
    for scale in [1,1e200,1e-200]:
        np.testing.assert_allclose(compute_wadell_sphericity(np.array([[a,b,c],[c,a,b]])*scale), expected, rtol=1e-12)


def test_sieve_weighting_and_duplicate_diameters():
    import pandas as pd
    from granular_analysis.analysis.particle_shape import compute_sieve_curve
    table = pd.DataFrame(dict(a=[1,1,2],b=[1,1,2],c=[1,1,2],rho=[1,3,1]))
    curve = compute_sieve_curve(table)
    np.testing.assert_allclose(curve.d_eq, [2,4])
    assert curve['count'].tolist() == [2,1]
    np.testing.assert_allclose(curve.passing_volume_percent, [20,100])
    np.testing.assert_allclose(curve.passing_mass_percent, [100/3,100])
    np.testing.assert_allclose(curve.passing_number_percent, [200/3,100])


@pytest.mark.parametrize('option',['--skip-points','--skip-particles'])
def test_skip_applies_to_every_output(tmp_path,option):
    from granular_analysis.scripts.run_all import main
    import pandas as pd
    (tmp_path/'input_ellipsoid_data.dat').write_text('1,2500,10,10,10\n2,2500,1,1,1\n3,2500,2,1,1\n')
    out=tmp_path/'out'
    assert main(['--only','particle_shape','--path',str(tmp_path),'--out',str(out),option,'1'])==0
    table=pd.read_csv(out/'particle_shape/particle_shape.csv')
    assert table.particle_id.tolist()==[2,3]
    assert len(pd.read_csv(out/'particle_shape/sieve_curve.csv'))==2
    assert (out/'particle_shape/sieve_curve.svg').is_file()
    assert main(['--only','particle_shape','--path',str(tmp_path),'--out',str(out),option,'3'])==1


def test_din_boundaries_and_fractions():
    import pandas as pd
    from granular_analysis.analysis.particle_shape import din_sieve_tables
    curve=pd.DataFrame(dict(d_eq=[.002,.063,2.,63.,200.,630.,1000.],
        passing_number_percent=[10,20,40,60,80,90,100],
        passing_mass_percent=[10,20,40,60,80,90,100],
        passing_volume_percent=[10,20,40,60,80,90,100]))
    sampled,classes=din_sieve_tables(curve,length_unit_mm=1)
    assert sampled.passing_mass_percent.tolist()==[10,10,10,20,20,20,40,40,40,60,80,90]
    assert classes.fraction_mass_percent.sum()==pytest.approx(100)
    assert classes.iloc[0].grain_class=='Ton'
    assert classes.iloc[0].fraction_mass_percent==10
    assert classes.iloc[-1].fraction_mass_percent==10
    changed,_=din_sieve_tables(curve,length_unit_mm=1000)
    assert changed.passing_mass_percent.iloc[0]==0


@pytest.mark.parametrize('scale',[0,-1,np.nan,np.inf])
def test_invalid_din_scale(scale):
    import pandas as pd
    from granular_analysis.analysis.particle_shape import din_sieve_tables
    with pytest.raises(ValueError):din_sieve_tables(pd.DataFrame(),length_unit_mm=scale)


def test_din_cli_outputs_and_labels(tmp_path):
    from granular_analysis.scripts.run_all import main
    import pandas as pd
    (tmp_path/'input_ellipsoid_data.dat').write_text('1,2500,1,1,1\n2,2500,2,1,1\n')
    out=tmp_path/'out'
    assert main(['--only','particle_shape','--path',str(tmp_path),'--out',str(out),
                 '--skip-particles','0','--length-unit-mm','0.5'])==0
    base=out/'particle_shape'
    assert pd.read_csv(base/'particle_shape.csv').d_eq_mm.iloc[0]==pytest.approx(1)
    assert len(pd.read_csv(base/'grain_classes_din.csv'))==13
    assert (base/'sieve_curve_din.csv').is_file()
    assert (base/'sieve_curve.pdf').is_file()
    assert 'DIN EN ISO 14688-1' not in (base/'sieve_curve.svg').read_text()
    assert '<text' not in (base/'sieve_curve.svg').read_text()
    assert 'Schluff' in (base/'sieve_curve.svg').read_text()


def test_confirmed_default_scale_and_din_classification(tmp_path):
    from granular_analysis.scripts.run_all import main
    import pandas as pd
    # Nominal diameters 0.7, 1.0 and 1.3 in input units.
    (tmp_path/'input_ellipsoid_data.dat').write_text(
        '1,2500,0.35,0.35,0.35\n2,2500,0.5,0.5,0.5\n3,2500,0.65,0.65,0.65\n')
    out=tmp_path/'out'
    assert main(['--only','particle_shape','--path',str(tmp_path),'--out',str(out),'--skip-particles','0'])==0
    base=out/'particle_shape'
    table=pd.read_csv(base/'particle_shape.csv')
    np.testing.assert_allclose(table.d_eq_mm,[.4305,.615,.7995])
    classes=pd.read_csv(base/'grain_classes_din.csv').set_index('grain_class')
    assert classes.loc['Mittelsand','fraction_number_percent']==pytest.approx(200/3)
    assert classes.loc['Grobsand','fraction_number_percent']==pytest.approx(100/3)
    assert classes.fraction_mass_percent.sum()==pytest.approx(100)
