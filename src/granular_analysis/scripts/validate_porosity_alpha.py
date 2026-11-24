"""Sweep the existing six-sphere benchmark over routine alpha=0,...,1."""

from granular_analysis.plotting import plotstyle
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.scripts.validate_porosity_six_spheres import six_spheres


def run_alpha_sweep(output):
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows=[]
    for alpha in np.round(np.arange(11)/10,1):
        for z in [1.25,1.5,2.,2.5,2.75]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', RuntimeWarning)
                warnings.simplefilter('ignore', DeprecationWarning)
                warnings.simplefilter('ignore', PendingDeprecationWarning)
                res=compute_slice_porosity(six_spheres(),z,float(alpha),6.)
            rows.append(dict(alpha=alpha,alpha_Deq=2*alpha,z_over_Deq=z/2,A_alpha_over_Deq2=res['A_alpha']/4,A_solid_over_Deq2=res['A_ell']/4,porosity=res['porosity'],physical=bool(0<=res['porosity']<=1),n_points=720,clip_solid_to_shape=False,warnings=' | '.join(str(w.message) for w in caught)))
            if z==2 and res['shape'] is not None:
                np.savetxt(output/f'alpha_shape_{alpha:.1f}.csv',res['shape']/2,delimiter=',',header='x_over_Deq,y_over_Deq',comments='')
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'alpha_sweep.csv',index=False)
    frame['porosity_percent']=100*frame.porosity
    fig, ax=plt.subplots(figsize=(plotstyle.figwidth_full,4))
    for z, group in frame.groupby('z_over_Deq'):
        ax.plot(group.alpha, 100*group.porosity, marker='o', label=f'z/Deq={z:g}')
    ax.set(xlabel=r'$\alpha$ (Routine, $D_\mathrm{eq}=2$)', ylabel=r'$\varepsilon$ [\%]')
    ax.legend(fontsize=10)
    plotstyle.prepare_figure(fig); fig.tight_layout(); plotstyle.export_figure(fig, output/'alpha_sweep.png', dpi=180); plt.close(fig)
    fig, axes=plt.subplots(3,4,figsize=(plotstyle.figwidth_full,10),sharex=True,sharey=True)
    for ax, alpha in zip(axes.flat, np.round(np.arange(11)/10,1)):
        for p in six_spheres():
            ax.add_patch(plt.Circle(p['x'][:2]/2,.5,color='steelblue',alpha=.35))
        points=np.loadtxt(output/f'alpha_shape_{alpha:.1f}.csv',delimiter=',',skiprows=1)
        ax.plot(*points.T,color='black',lw=1)
        ax.set(title=rf'$\alpha={alpha:.1f}$',xlim=(-.6,2.6),ylim=(-.6,2.35))
        ax.set_aspect('equal')
    axes.flat[-1].axis('off')
    fig.supxlabel(r'$x / D_{\mathrm{eq}}$'); fig.supylabel(r'$y / D_{\mathrm{eq}}$')
    plotstyle.prepare_figure(fig); fig.tight_layout(); plotstyle.export_figure(fig, output/'alpha_shapes.png',dpi=160); plt.close(fig)
    write_appendix(frame)
    print(frame[frame.z_over_Deq==1].drop(columns='warnings').to_string(index=False))
    return frame


def write_appendix(frame):
    """Export the sweep table and data for the accompanying PGFPlots appendix."""
    folder=Path('bilder/generiert/anhang_porositaetsvalidierung')
    folder.mkdir(parents=True,exist_ok=True)
    equator=frame[frame.z_over_Deq==1].copy()
    equator['porosity_percent']=100*equator.porosity
    equator.to_csv(folder/'alpha_equator.csv',index=False)
    lines=[r'\begin{tabular}{rrrrl}',r'\hline',
           r'$\alpha$ (Routine) & $\alpha D_{\mathrm{eq}}$ & $A_\alpha / D_{\mathrm{eq}}^{2}$ & $\varepsilon$ [\%] & Bewertung' + chr(92)*2,r'\hline']
    for row in equator.itertuples():
        values=[f'{row.alpha:.1f}',f'{row.alpha_Deq:.1f}',f'{row.A_alpha_over_Deq2:.6f}',f'{100*row.porosity:.6f}']
        lines.append(' & '.join(v.replace('.',',') for v in values)+' & '+('zulässig' if row.physical else 'ungültig')+chr(92)*2)
    lines += [r'\hline',r'\end{tabular}']
    (folder/'alpha_tabelle.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results_granular_analysis/validation_six_spheres'))
    run_alpha_sweep(parser.parse_args().out)
