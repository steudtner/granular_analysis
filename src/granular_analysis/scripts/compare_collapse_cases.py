"""Compare multiple runs of the same particle form using the P0–P2 products.

Run --products LABEL=DIR for existing products, or --case LABEL=CASE to analyze
a case first. Copies inputs to a new output directory; simulation cases are read-only.
"""

from granular_analysis.plotting import plotstyle
import argparse
import json
import logging
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from granular_analysis.analysis.collapse import (
    CollapseOptions, analyze_case, input_numbers, sha256,
)
from granular_analysis.scripts.run_collapse_analysis import export_plot, latex_table
from granular_analysis.utils.physics import load_ellipsoid_properties


def compare_products(products, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from granular_analysis.plotting.plotstyle import configure
    fonts = Path(__file__).resolve().parents[3] / 'thesis/fonts'
    for font in fonts.glob('NotoSans-*.ttf'):
        font_manager.fontManager.addfont(font)
    configure()
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['mathtext.fallback'] = None
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    statuses = {label: json.loads((folder/'status.json').read_text())
                for label, folder in products.items()}
    provenance = {label: json.loads((folder/'herkunft.json').read_text())
                  for label, folder in products.items()}
    if len({s['form'] for s in statuses.values()}) != 1:
        raise ValueError('This comparison requires the same particle form')
    options = next(iter(provenance.values()))['options']
    if any(p['options'] != options for p in provenance.values()):
        raise ValueError('Comparison products must use identical analysis options')
    end = pd.DataFrame([dict(label=label, **s['end']) for label, s in statuses.items()])
    end.to_csv(out/'endwerte_vergleich.csv', index=False)
    (out/'endwerte_vergleich.tex').write_text(latex_table(end, [
        ('label','Case'), ('L_end','L'), ('H_end','H'), ('phi_end','φ'),
        ('t95','t95'), ('t_end','Energie-Endzeit'), ('theta','θ')]))
    config_rows = []
    for label, status in statuses.items():
        root = Path(status['case'])
        cfg = input_numbers(root/'input.dat')
        main = (root/'main.def').read_text()
        _, masses, _, _, _ = load_ellipsoid_properties(root/'input_ellipsoid_data.dat')
        config_rows.append(dict(label=label, case=str(root),
            **{k: cfg.get(k, np.nan) for k in ('nell','nell_fixed','dt','adjdt',
                'lenx','leny','lenz','nx','ny','nz','vis','rho_f','gry','bc_north')},
            mobile_particles=int(cfg['nell']-cfg['nell_fixed']),
            mobile_mass=float(masses[int(cfg['nell_fixed']):int(cfg['nell'])].sum()),
            adaptive_dt=bool(re.search(r'^\s*adjdt\s*=\s*\.T\.',
                (root/'input.dat').read_text(), re.M | re.I)),
            collision_cut=bool(re.search(r'^\s*#define\s+_ellipsoid_cut_fp_collision_',main,re.M)),
            input_sha256=sha256(root/'input.dat'), main_sha256=sha256(root/'main.def')))
    pd.DataFrame(config_rows).to_csv(out/'case_parameter_vergleich.csv', index=False)
    tables = {}
    for name in ('geometry_timeseries','energy','top_layer_settling','excess_pressure',
                 'orientation_summary','contact_summary','alpha_sensitivity','slope_fit'):
        rows = []
        for label, folder in products.items():
            path = folder/(name+'.csv')
            if path.exists():
                table = pd.read_csv(path)
                table['label'] = label
                rows.append(table)
        tables[name] = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        tables[name].to_csv(out/(name+'_vergleich.csv'), index=False)
    geometry = tables['geometry_timeseries']
    energy = tables['energy']
    common_end = min(geometry.groupby('label').t.max())
    common_start = max(geometry.groupby('label').t.min())
    common = []
    for label, group in geometry.groupby('label',sort=False):
        # Report observed snapshots, never present interpolated geometry as a measurement.
        row = group[group.t <= common_end+1e-10].iloc[-1].to_dict()
        row['target_common_end'] = common_end
        row['snapshot_time_offset'] = common_end-row['t']
        common.append(row)
    pd.DataFrame(common).to_csv(out/'gemeinsamer_zeithorizont.csv',index=False)
    plots = [
        ('geometry_timeseries','L_t','Frontposition L / D_eq','front'),
        ('geometry_timeseries','H_t','Höhe H / D_eq','hoehe'),
        ('geometry_timeseries','phi_t','Feststoffanteil φ','porositaet'),
        ('geometry_timeseries','u_front','Frontgeschwindigkeit / u_s','frontgeschwindigkeit'),
        ('energy','E_kin_trans','Translationsenergie E*','translation'),
        ('energy','E_kin_rot','Rotationsenergie E*, gefiltert','rotation'),
        ('energy','E_kin_tot','Kinetische Energie E*, gefiltert','energie'),
        ('energy','E_kin_rot_raw','Rotationsenergie E*, roh','rotation_roh'),
        ('top_layer_settling','mean_vy_over_us','Obere Schicht: v_y / u_s','absinken'),
        ('excess_pressure','excess_pressure_mean','Druck gegenüber Fernfeld','druck'),
    ]
    for source, metric, ylabel, name in plots:
        data = tables[source]
        if metric not in data or data.empty:
            continue
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full,3.5))
        for label, group in data.groupby('label',sort=False):
            ax.plot(group.t,group[metric],label=label)
        ax.set(xlabel='t* = t u_s / D_eq',ylabel=ylabel)
        ax.axvline(common_end,color='gray',linestyle=':',linewidth=.8,
                   label='Ende gemeinsamer Zeitraum')
        ax.legend()
        export_plot(fig,out,name,data)
    fig, axes = plt.subplots(2,2,figsize=(plotstyle.figwidth_full,7))
    for ax, source, metric, title in zip(axes.flat,
            ['energy','geometry_timeseries','geometry_timeseries','energy'],
            ['E_kin_tot','L_t','phi_t','E_kin_rot'],
            ['Kinetische Energie E*','Frontposition L / D_eq','Feststoffanteil φ','Rotationsenergie E*']):
        for label, group in tables[source].groupby('label',sort=False):
            ax.plot(group.t,group[metric],label=label)
        ax.set(xlim=(common_start,common_end),xlabel='t*',ylabel=title)
        ax.legend()
    plotstyle.prepare_figure(fig)
    fig.tight_layout()
    plotstyle.export_figure(fig, out/'uebersicht.png',dpi=150)
    export_plot(fig,out,'uebersicht',geometry)
    peaks = []
    for label, group in energy.groupby('label',sort=False):
        index = group.E_kin_tot.idxmax()
        initial_potential = geometry[geometry.label==label].iloc[0].E_pot
        peaks.append(dict(label=label,E_kin_peak=group.loc[index,'E_kin_tot'],
            t_peak=group.loc[index,'t'],E_pot_initial=initial_potential,
            peak_over_initial_potential=group.loc[index,'E_kin_tot']/initial_potential,
            missing_energy_samples=int(group.E_kin_tot.isna().sum()),
            rotation_filter_samples=int((group.E_kin_rot_excluded>0).sum()),
            monitor_frame_match_audited=bool(group.monitor_frame_match_audited.all())))
    pd.DataFrame(peaks).to_csv(out/'energie_kennwerte.csv',index=False)
    issues = '\n'.join('- '+label+': '+issue for label,s in statuses.items() for issue in s['issues'])
    (out/'README.md').write_text(f'''# Vergleich: {" und ".join(products)}

Beide Cases wurden mit denselben P0–P2-Einstellungen ausgewertet. Original-Cases
sind unverändert. Fallprodukte und Herkunftsnachweise stehen in den Unterordnern.
Die Routine `granular_analysis.scripts.compare_collapse_cases` kann weitere
Läufe derselben Partikelform vergleichen (`--case LABEL=PATH` oder
`--products LABEL=PRODUCT_DIR`, jeweils wiederholbar; `--out NEW_DIR`).

`endwerte_vergleich.csv` enthält die jeweils letzte gespeicherte Geometrie;
diese Zeitpunkte unterscheiden sich. `gemeinsamer_zeithorizont.csv` enthält
die letzten tatsächlich ausgewerteten Snapshots bis zum gemeinsamen Ende
t*={common_end:.6f}, einschließlich Zeitabstand. Die Übersicht zeigt nur den
gemeinsamen Zeitraum, Einzelplots zeigen die volle Laufdauer.
t95 bezieht sich jeweils auf 95% der eigenen beobachteten Endausbreitung.
Quasistatische Endzeit: Energie <1% des eigenen Maximums für mindestens 10 t*.

`case_parameter_vergleich.csv` dokumentiert Randbedingungen, Partikelzahl,
Zeitschritt und Collision-Cut-Schalter. Unterschiedliche mobile Partikelzahlen,
Anfangspackungen, Zeitschritte und obere Randbedingungen verhindern eine
isolierte Zuschreibung der Unterschiede an den Collision-Cut-Schalter.
Beide Gitter sind gleich; dies ist keine Gitterkonvergenzstudie.
Energie E*=E/(ρ_p g D_eq⁴); `energie_kennwerte.csv` enthält zusätzlich das
Verhältnis zum anfänglichen potentiellen Energieinhalt zur Einordnung
unterschiedlicher Gesamtmassen. Der Rotationsfilter entfernt pro Sample nur
den einzelnen Beitrag >50% der rohen Rotationssumme; Rohwerte bleiben erhalten.
NaNs in Energieplots werden nicht überbrückt. Fehlt ein Monitor/TRN-Audit,
wird keine Übereinstimmung behauptet. Kontakte sind nur impulsaktive Kontakte,
keine vollständige persistente Koordinationszahl. Vollständige Dissipation
ist aus diesen Daten nicht bestimmbar.

Bekannte Grenzen:
{issues}
''')
    (out/'herkunft.json').write_text(json.dumps(dict(options=options,
        source_products={label:str(folder) for label,folder in products.items()},
        product_provenance={label:sha256(folder/'herkunft.json') for label,folder in products.items()},
        comparison_script_sha256=sha256(__file__),common_time_end=common_end),indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',default=[],metavar='LABEL=PATH')
    parser.add_argument('--products',action='append',default=[],metavar='LABEL=PATH')
    parser.add_argument('--form',choices=['Prolat','Oblat','Kugel'],default='Prolat')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True,exist_ok=True)
    products = {}
    for kind, supplied in [('case',v) for v in args.case]+[('products',v) for v in args.products]:
        label, source = supplied.split('=',1)
        if not re.fullmatch(r'[A-Za-z0-9_-]+',label) or label in products:
            raise ValueError('Labels must be unique safe directory names')
        folder = args.out/label
        if kind=='case':
            analyze_case(args.form,source,folder,CollapseOptions())
        elif Path(source).resolve()!=folder.resolve():
            if folder.exists():
                raise ValueError('Refusing to overwrite existing copied products')
            shutil.copytree(source,folder)
        products[label]=folder
    if len(products)<2:
        raise ValueError('At least two runs required')
    compare_products(products,args.out)
    (args.out/'delivery_manifest.json').write_text(json.dumps({str(p.relative_to(args.out)):sha256(p)
        for p in sorted(args.out.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'},indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
