"""Aktuelle Low-/Mid-Fits exportieren: direkte n/k, eindeutige Abschnittsnamen, neue Plots."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from granular_analysis.thesis.aicc import augment, latex_rows
from granular_analysis.plotting import plotstyle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'bilder/generiert'


def sha(path):
    """SHA256 einer Fitquelle zur Erkennung abweichender Datenstände."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Aktuelle slope_results_R.csv und Fitpunkte prüfen; CSV, booktabs und SVG/PDF exportieren."""
    tables, sources = [], []
    for slug, case, resolution in [('jainsu','primeJainSU',16),
                                   ('higparmidfluid','primeHigParMidFluid',20)]:
        run = OUT / f'kap6_2_slope_{slug}'
        source = ROOT / f'DiscretizationGranularCollapse/{case}/results_granular_analysis/alphashape_last_timestep.csv'
        data, provenance = augment(run / 'slope_results_R.csv', resolution)
        fits = pd.read_csv(run / 'slope_results_R.csv')
        points = pd.read_csv(run / 'fit_upper_edge.csv')
        current = pd.read_csv(source, header=None, names=['x','z_max'])
        current = current.apply(pd.to_numeric, errors='coerce')
        current = current[np.isfinite(current).all(axis=1)].sort_values('x')
        if len(fits)!=6 or not (data.n==len(points)).all():
            raise ValueError(f'{case}: unvollständige Fits/Punktzahlen')
        if current.shape!=points.shape or not np.allclose(current,points,rtol=0,atol=1e-12):
            raise ValueError(f'{case}: aktuelle Quelldatei stimmt nicht mit Fitpunkten überein')
        if not np.allclose(fits.aicc,data.AICc):
            raise ValueError(f'{case}: AICc-Abweichung zwischen R und Python')
        if slug=='higparmidfluid':
            previous=json.loads((run/'herkunft.json').read_text())
            if sha(source)!=previous['upper_edge_sha256']:
                raise ValueError('Mid-Quelldatei seit dem Neufit verändert')
        provenance.update(case=case, upper_edge_source=str(source), upper_edge_sha256=sha(source),
            script_sha256=sha(ROOT/'granular_analysis/src/granular_analysis/scripts/slope_analysis.R'),seed=20260924,
            model_labels='CSV/Tabellen: Anzahl Geradenabschnitte = n_breakpoints + 1; R-Originalnamen unverändert')
        (run/'herkunft_aktuell.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        data['modell'] = ['Linear (1 Abschnitt)' if b==0 else f'{int(b)+1} Abschnitte'
                           for b in fits.n_breakpoints]
        tables.append(data)
        sources.append(provenance)
    combined=pd.concat(tables,ignore_index=True)
    combined.to_csv(OUT/'kap7_aicc_aktuell.csv',index=False,float_format='%.12g')
    (OUT/'kap7_1_tabellenzeilen_aktuell.tex').write_text(latex_rows(tables[0]),encoding='utf-8')
    (OUT/'kap6_2_tabellenzeilen_aktuell.tex').write_text(latex_rows(combined),encoding='utf-8')
    (OUT/'kap7_aicc_aktuell_herkunft.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    for p in (ROOT/'granular_analysis/thesis/fonts').glob('*.ttf'):
        font_manager.fontManager.addfont(str(p))
    plotstyle.configure(font_file=str(ROOT/'granular_analysis/thesis/fonts/NotoSans-Regular.ttf'))
    for table in tables:
        resolution=int(table.aufloesung.iloc[0])
        fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.5),layout='constrained')
        for metric,marker in [('AIC','o'),('BIC','s'),('AICc','^')]:
            ax.plot(table.modell,table[metric],marker=marker,label=metric)
        ax.set(xlabel='Modell (Anzahl der Geradenabschnitte)',ylabel='Informationskriterium')
        ax.tick_params(axis='x',rotation=25)
        ax.legend(loc='upper left',bbox_to_anchor=(1.02,1))
        plotstyle.save(fig,f'kap7_2_modellvergleich_delta{resolution}_aktuell')
        plt.close(fig)
    run=OUT/'kap6_2_slope_jainsu'
    fits=pd.read_csv(run/'slope_results_R.csv')
    points=pd.read_csv(run/'fit_upper_edge.csv')
    fig,axes=plt.subplots(3,2,figsize=(plotstyle.figwidth_full,7.4),sharex=True,sharey=True,layout='constrained')
    for ax,row,label in zip(axes.flat,fits.itertuples(),tables[0].modell):
        ax.scatter(points.x,points.z_max,s=8,color='black',label='Oberkante')
        ax.plot(np.fromstring(row.x_pred,sep=';'),np.fromstring(row.y_pred,sep=';'),label=label)
        ax.set(xlabel=r'$x\,D_{\mathrm{eq}}^{-1}$',ylabel=r'Höhe $y\,D_{\mathrm{eq}}^{-1}$')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.24),ncol=2)
    plotstyle.save(fig,'kap6_2_slope_jainsu_modelle')
    plt.close(fig)
    print(combined.to_string(index=False))
    for table in tables:
        print('Delta',int(table.aufloesung.iloc[0]))
        for metric in ['AIC','BIC','AICc']:
            best=table.loc[table[metric].idxmin()]
            print(metric,best.modell,float(best[metric]))

if __name__=='__main__': main()
