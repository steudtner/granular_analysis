"""Regression checks for information criteria and refusal of ambiguous input."""
import numpy as np
import pandas as pd
import pytest
from granular_analysis.thesis.aicc import augment
from granular_analysis.thesis.plots import read_table, grouped, kinetic_energy, SCALINGS


def test_direct_fit_export_preserves_likelihood_scores(tmp_path):
    p=tmp_path/'fits.csv'
    pd.DataFrame({'model':['Linear'],'n':[20],'k':[3],'aic':[10.],
        'bic':[10+3*(np.log(20)-2)]}).to_csv(p,index=False)
    data,source=augment(p,16)
    assert data.AIC.iloc[0]==10
    assert data.AICc.iloc[0]==pytest.approx(11.5)
    assert not source['legacy']


def test_legacy_requires_opt_in_and_recovers_integer_n(tmp_path):
    p=tmp_path/'old.csv'
    pd.DataFrame({'model':['3 Segmente'],'n_breakpoints':[3],'aic':[38.0735379328009],
        'bic':[53.7125644973512]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='n/k fehlen'): augment(p,16)
    data,source=augment(p,16,True)
    assert data.n.iloc[0]==42 and data.k.iloc[0]==9
    assert data.AICc.iloc[0]==pytest.approx(43.6985379328009)
    assert source['legacy']


def test_reject_inconsistent_criteria_and_small_sample(tmp_path):
    p=tmp_path/'bad.csv'
    for n,bic in [(4,10+3*(np.log(4)-2)),(20,99)]:
        pd.DataFrame({'model':['Linear'],'n':[n],'k':[3],'aic':[10.],'bic':[bic]}).to_csv(p,index=False)
        with pytest.raises(ValueError): augment(p,16)


def test_missing_input_and_restart_duplicates(tmp_path):
    with pytest.raises(FileNotFoundError,match='Eingabedatei fehlt'):
        read_table(tmp_path/'missing.csv',['t'])
    d=pd.DataFrame({'form':['Kugel']*2,'fall':['a']*2,'t':[1,1]})
    with pytest.raises(ValueError,match='Restart'):
        list(grouped(d,'t'))


def test_no_invented_quasistatic_marker(tmp_path):
    p=tmp_path/'energy.csv'
    pd.DataFrame({'form':['Kugel'],'fall':['a'],'t':[0.],'E_kin_star':[0.]}).to_csv(p,index=False)
    with pytest.raises(FileNotFoundError): kinetic_energy(p,tmp_path/'undecided.csv')


def test_height_exponents_use_consistent_normalization():
    assert SCALINGS['balmforth_wide_h'][1]==pytest.approx(1-.6)
    assert SCALINGS['lajeunesse_channel_h_high'][1]==pytest.approx(1/3)
    assert SCALINGS['lube_axisymmetric_h_mid'][1]==pytest.approx(1/6)
