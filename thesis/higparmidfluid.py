"""Aktuellen HigParMidFluid-R-Fit prüfen und als neue Tabellen/Plots exportieren."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from granular_analysis.thesis.aicc import augment, latex_rows
from granular_analysis.plotting import plotstyle

root = Path(__file__).resolve().parents[2]
out = root / 'bilder/generiert'
run = out / 'kap6_2_slope_higparmidfluid'
source = root / 'DiscretizationGranularCollapse/primeHigParMidFluid/results_granular_analysis/alphashape_last_timestep.csv'
data, provenance = augment(run / 'slope_results_R.csv', 20)
fits = pd.read_csv(run / 'slope_results_R.csv')
points = pd.read_csv(run / 'fit_upper_edge.csv')
assert len(fits) == 6 and (data.n == len(points)).all()
assert np.allclose(fits.aicc, data.AICc)
provenance['upper_edge_source'] = str(source)
provenance['upper_edge_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
provenance['script_sha256'] = hashlib.sha256((root / 'granular_analysis/src/granular_analysis/scripts/slope_analysis.R').read_bytes()).hexdigest()
provenance['seed'] = 20260924
(run / 'herkunft.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
data.to_csv(out / 'kap6_2_slope_higparmidfluid_kriterien.csv', index=False, float_format='%.12g')
(out / 'kap6_2_slope_higparmidfluid_tabellenzeilen.tex').write_text(latex_rows(data), encoding='utf-8')
for p in (root / 'granular_analysis/thesis/fonts').glob('*.ttf'):
    font_manager.fontManager.addfont(str(p))
plotstyle.configure(font_file=str(root / 'granular_analysis/thesis/fonts/NotoSans-Regular.ttf'))
fig, axes = plt.subplots(3, 2, figsize=(plotstyle.figwidth_full, 7.4), sharex=True, sharey=True, layout='constrained')
for ax, row in zip(axes.flat, fits.itertuples()):
    ax.scatter(points.x, points.z_max, s=8, color='black', label='Oberkante')
    ax.plot(np.fromstring(row.x_pred, sep=';'), np.fromstring(row.y_pred, sep=';'), label=row.model)
    ax.set(xlabel=r'$x / D_{\mathrm{eq}}$', ylabel=r'Höhe $y / D_{\mathrm{eq}}$')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.24), ncol=2)
plotstyle.save(fig, 'kap6_2_slope_higparmidfluid_modelle')
plt.close(fig)
fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 3.5), layout='constrained')
for metric, marker in [('AIC', 'o'), ('BIC', 's'), ('AICc', '^')]:
    ax.plot(data.modell, data[metric], marker=marker, label=metric)
ax.set(xlabel='Modell (Benennung nach Knickpunktzahl)', ylabel='Informationskriterium')
ax.tick_params(axis='x', rotation=25)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plotstyle.save(fig, 'kap6_2_slope_higparmidfluid_kriterien')
plt.close(fig)
print(data.to_string(index=False))
for metric in ['AIC', 'BIC', 'AICc']:
    best = data.loc[data[metric].idxmin()]
    print(f'{metric}: {best.modell}, {best[metric]:.6f}')
