"""Geometric equivalent-diameter cumulative distribution."""

from granular_analysis.plotting import plotstyle
from pathlib import Path
import numpy as np
from granular_analysis.analysis.particle_shape import DIN_CLASS_LIMITS_MM, din_sieve_tables
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def plot_sieve_curve(curve, output_path, *, length_unit_mm=None):
    figure = Figure(figsize=(plotstyle.figwidth_full, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    diameter = curve.d_eq.to_numpy()
    if length_unit_mm is not None:
        din_sieve_tables(curve, length_unit_mm=length_unit_mm)
        diameter = diameter * length_unit_mm
    for name in ('volume', 'mass', 'number'):
        ax.step(diameter, curve[f'passing_{name}_percent'], where='post', label=name.capitalize())
    ax.set_xscale('log')
    ax.set(xlabel='Volume-equivalent diameter (input length units)',
           ylabel='Cumulative fraction ≤ diameter (%)', ylim=(0, 101),
           title='Equivalent-diameter size distribution')
    if length_unit_mm is not None:
        # Full classic soil-classification chart; extend for outlying grains.
        lo = min(.001, diameter.min()/2)
        hi = max(1000., diameter.max()*2)
        ax.set_xlim(lo, hi)
        ax.set_xlabel('Volumenäquivalenter Korndurchmesser d (mm)')
        ax.set_ylabel('Kumulativer Anteil ≤ d (%)')
        ax.set_title('Sieblinie – Korngrößenklassen nach DIN EN ISO 14688-1', pad=58)
        ax.set_xticks(DIN_CLASS_LIMITS_MM,
                      ['0.002','0.0063','0.02','0.063','0.2','0.63','2','6.3','20','63','200','630'])
        ax.tick_params(axis='x', labelrotation=45)
        groups = [(lo,.002,'Ton'),(.002,.063,'Schluff'),(.063,2,'Sand'),
                  (2,63,'Kies'),(63,200,'Steine'),(200,630,'Blöcke'),(630,hi,'Große Blöcke')]
        for i,(left,right,label) in enumerate(groups):
            ax.axvspan(left,right,color='0.92' if i%2==0 else '0.98',zorder=0)
            ax.text(np.sqrt(left*right),1.10,label,transform=ax.get_xaxis_transform(),
                    ha='center',va='bottom',fontsize=10,clip_on=False)
        for boundary in DIN_CLASS_LIMITS_MM:
            ax.axvline(boundary,color='0.65',linewidth=.6,zorder=0)
        for left,right in [(.002,.063),(.063,2),(2,63)]:
            edges=[left,*DIN_CLASS_LIMITS_MM[(DIN_CLASS_LIMITS_MM>left)&(DIN_CLASS_LIMITS_MM<right)],right]
            for lower,upper,label in zip(edges[:-1],edges[1:],['fein','mittel','grob']):
                ax.text(np.sqrt(lower*upper),1.025,label,transform=ax.get_xaxis_transform(),
                        ha='center',va='bottom',fontsize=10,clip_on=False)
    ax.legend()
    ax.grid(True, which='both', alpha=.25)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plotstyle.export_figure(figure, destination)
    return destination
