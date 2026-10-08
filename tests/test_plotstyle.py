"""Print-size and TeX regression checks for the common CLI plot exporter."""
import shutil
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pytest

from granular_analysis.plotting import plotstyle


@pytest.mark.skipif(not shutil.which('latex') or not shutil.which('pdflatex'),
                    reason='Real TeX export requires MiKTeX or TeX Live')
def test_real_vector_pair_preserves_size_and_labels(tmp_path):
    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.figwidth_full, 3), layout='constrained')
    axes[0].plot([0, .5, 1], [1e-7, 2e-7, 3e-7], label='A & B: 0.5', linewidth=.8)
    axes[0].set(xlabel=r'$l^{*}$', ylabel=r'$\beta$ [rad]', title='Remove this title')
    axes[0].legend(frameon=True, fontsize=7)
    axes[1].bar(['Fall #1', 'Fall #2'], [.2, .4])
    axes[1].set_yscale('log')
    before = fig.get_size_inches().copy()
    paths = plotstyle.export_figure(fig, tmp_path/'style.svg')
    assert {path.suffix for path in paths} == {'.svg', '.pdf'}
    assert all(path.is_file() for path in paths)
    np.testing.assert_array_equal(fig.get_size_inches(), before)
    root = ET.parse(tmp_path/'style.svg').getroot()
    assert float(root.attrib['width'].removesuffix('pt')) == pytest.approx(160/25.4*72, abs=1e-5)
    assert not list(root.iter('{http://www.w3.org/2000/svg}text'))
    assert axes[0].get_title() == ''
    assert not axes[0].get_legend().get_frame_on()
    assert all(text.get_fontsize() == 10 and text.get_usetex()
               for text in fig.findobj(match=Text))
    assert r'\#' in axes[1].get_xticklabels()[0].get_text()
    assert '{,}' in plotstyle.CommaScalarFormatter.comma('$0.5$')
    assert axes[0].yaxis.get_major_formatter().get_offset()
    plt.close(fig)


def test_missing_tex_is_reported_without_font_fallback(monkeypatch):
    monkeypatch.setattr(plotstyle.shutil, 'which', lambda _name: None)
    with pytest.raises(RuntimeError, match='LaTeX fehlt im PATH'):
        plotstyle.check_tex()


def test_shape_styles_are_not_assigned_by_case_order():
    assert plotstyle.case_style('oblat_case')['color'] == '#cc6600'
    assert plotstyle.case_style('Prolat')['color'] == '#000099'
    assert plotstyle.case_style('sphere_case')['color'] == '#000000'
    prolat = plotstyle.MARKERS['Prolat'].vertices
    oblat = plotstyle.MARKERS['Oblat'].vertices
    assert np.ptp(prolat[:, 0]) > np.ptp(prolat[:, 1])
    assert np.ptp(oblat[:, 1]) > np.ptp(oblat[:, 0])
