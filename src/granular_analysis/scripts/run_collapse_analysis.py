"""Generate P0–P2 collapse products in a new directory from completed cases.

Example:
 python -m granular_analysis.scripts.run_collapse_analysis \
   --case Oblat=/path/to/oblat --case Prolat=/path/to/complete_results --out /new/output
"""

from granular_analysis.plotting import plotstyle
import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from granular_analysis.analysis.collapse import (
    CollapseOptions, analyze_case, dry_scaling, grid_comparison, sha256, initial_settling_indicator,
)


def latex_table(table,columns):
    def cell(value):
        if isinstance(value,(float,np.floating)):
            return f'{value:.4f}'.replace('.', '{,}') if np.isfinite(value) else '--'
        return str(value).replace('_',r'\_')
    rows=[r'\begin{tabular}{l'+'r'*(len(columns)-1)+'}',r'\toprule',
          ' & '.join(label for _,label in columns)+r' \\',r'\midrule']
    for _,row in table.iterrows(): rows.append(' & '.join(cell(row[key]) for key,_ in columns)+r' \\')
    return '\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])+'\n'


def export_plot(fig,out,name,table):
    from matplotlib.ticker import FuncFormatter
    for ax in fig.axes:
        if ax.get_xscale()=='log': ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
        if ax.get_yscale()=='log': ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    plotstyle.export_figure(fig, out/(name+'.pdf'))
    table.to_csv(out/(name+'_eingabe.csv'),index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)


def report(out,statuses,options,grid_pairs=None,literature=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from granular_analysis.plotting.plotstyle import configure,COLORS
    fonts=Path(__file__).resolve().parents[3]/'thesis/fonts'
    for font in fonts.glob('NotoSans-*.ttf'):
        font_manager.fontManager.addfont(font)
    configure()
    matplotlib.rcParams['axes.unicode_minus']=False
    matplotlib.rcParams['mathtext.fallback']=None
    end=pd.DataFrame([s['end'] for s in statuses])
    end.to_csv(out/'endwerte_formen.csv',index=False)
    columns=[('form','Form'),('l','$l$'),('h','$h$'),('phi_end','$\\phi$'),
             ('t_collapse','$t_{\\mathrm{collapse}}^*$'),('u_front_max','$u_{\\mathrm{front,max}}/u_s$'),
             ('Pi_S','$\\Pi_S$'),('theta_1','$\\theta_1$'),('theta_2','$\\theta_2$'),('theta_3','$\\theta_3$')]
    (out/'endwerte_formen_tabellenzeilen.tex').write_text(latex_table(end,columns))
    # CSVs retain selection/completeness flags alongside numerical values.
    if not end.quasistatic.all():
        (out/'ENDWERTE_HINWEIS.txt').write_text('Zeilen mit quasistatic=False sind beobachtete Abschluss-Geometrien, keine bestätigten quasistatischen Endwerte. t_collapse ist die Front-t95-Zeit; t_end ist die gesonderte energiebezogene Endzeit.\n')
    all_geometry=[];all_energy=[];all_alpha=[];all_spread=[];all_orientation=[];all_settling=[];all_pressure=[];all_budget=[]
    for status in statuses:
        folder=out/status['run_id'];form=status['form']
        for filename,target in [('geometry_timeseries.csv',all_geometry),('energy.csv',all_energy),
             ('alpha_sensitivity.csv',all_alpha),('spanwise_2d.csv',all_spread),
             ('orientation.csv',all_orientation),('top_layer_settling.csv',all_settling),
             ('excess_pressure.csv',all_pressure),('energy_budget.csv',all_budget)]:
            path=folder/filename
            if path.exists():
                data=pd.read_csv(path);data['form']=form;data['run_id']=status['run_id'];target.append(data)
    groups={'geometry':all_geometry,'energy':all_energy,'alpha':all_alpha,'spread':all_spread,
            'orientation':all_orientation,'settling':all_settling,'pressure':all_pressure,'budget':all_budget}
    tables={key:pd.concat(value,ignore_index=True) for key,value in groups.items() if value}
    for key in ('geometry','energy','settling','pressure','budget'):
        if key in tables:tables[key].to_csv(out/(key+'_vergleich.csv'),index=False)
    for metric,label,name in [('E_kin_trans','Translationsenergie','kap7_1_translation'),
        ('E_kin_rot','Rotationsenergie, gefiltert','kap7_1_rotation'),
        ('E_kin_tot','Kinetische Energie, gefiltert','kap7_1_kinenergie')]:
        fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4))
        for form,data in tables['energy'].groupby('form'):
            ax.plot(data.t,data[metric],color=COLORS[form],label=form)
            qs=end.loc[end.form==form,'t_end'].iloc[0]
            if np.isfinite(qs):ax.axvline(qs,color=COLORS[form],ls=':',linewidth=.8)
        ax.set(xlabel='t* = t u_s/D_eq',ylabel=label+' E*');ax.legend()
        export_plot(fig,out,name,tables['energy'])
    for metric,label,name in [('L_t','L/D_eq','kap7_2_runout'),('H_t','H/D_eq','kap7_2_resthoehe'),
                             ('phi_t','φ','kap7_3_porositaet_zeit'),('delta_phi','φ(t) - φ(0)','kap7_3_dilatanz')]:
        fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4))
        for form,data in tables['geometry'].groupby('form'):
            ax.plot(data.t,data[metric],color=COLORS[form],label=form)
        ax.set(xlabel='t*',ylabel=label);ax.legend();export_plot(fig,out,name,tables['geometry'])
    fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4))
    for form,data in tables['energy'].groupby('form'):
        ax.plot(data.t,data.E_kin_rot_raw,color=COLORS[form],ls=':',label=form+' roh')
        ax.plot(data.t,data.E_kin_rot,color=COLORS[form],label=form+' gefiltert')
    ax.set(xlabel='t*',ylabel='Rotationsenergie E*');ax.legend()
    export_plot(fig,out,'kap7_1_rotation_filtervergleich',tables['energy'])
    for metric,label,name in [('L_end','Auslauflänge','kap6_2_alpha_runout'),('A_end','Fläche','kap6_2_alpha_flaeche')]:
        fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4))
        for form,data in tables['alpha'].groupby('form'):
            ax.plot(data.alpha,100*data[metric+'_relative_difference'],'o-',color=COLORS[form],label=form)
        ax.set(xlabel='α',ylabel=label+': Abweichung zu alpha=0.6 [%]');ax.legend()
        export_plot(fig,out,name,tables['alpha'])
    spread=tables['spread'];spread.to_csv(out/'spanwise_2d_formen.csv',index=False)
    fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4));width=.22
    for j,key in enumerate(('L','H','A','phi')):
        ax.bar(np.arange(len(spread))+(j-1.5)*width,100*spread[key+'_normalized_std'],width,label=key)
    ax.set_xticks(np.arange(len(spread)));ax.set_xticklabels(spread.form)
    ax.set_ylabel('Normierte Streuung über z [%]');ax.legend(ncol=4)
    export_plot(fig,out,'kap5_2_z_abhaengigkeit',spread)
    # Π_S plots are routines for arbitrary shape sets. Missing Kugel is explicit.
    metrics=[('l_minus_1','l - 1','auslauf'),('h','h','resthoehe'),('theta','θ [Grad]','winkel'),
             ('t_collapse','t_collapse u_s/D_eq','dauer'),('u_front_max','u_front,max/u_s','frontgeschwindigkeit')]
    end['l_minus_1']=end.l-1 # Literal Auftrag P1.1; l is already ΔL/L0.
    for metric,label,slug in metrics:
        fig,ax=plt.subplots(figsize=(5.5,3.4));order=end.sort_values('Pi_S')
        valid=order[np.isfinite(order[metric])]
        for _,row in valid.iterrows():
            marker='o' if row.quasistatic else 'x'
            state=' (Snapshot)' if not row.quasistatic else ''
            ax.scatter(row.Pi_S,row[metric],color=COLORS[row.form],marker=marker,label=row.form+state)
        if len(valid)>1:
            first,last=valid.iloc[0],valid.iloc[-1]
            ax.annotate('',xy=(last.Pi_S,last[metric]),xytext=(first.Pi_S,first[metric]),
                        arrowprops={'arrowstyle':'->','color':'gray','lw':.8})
        ax.set(xlabel='Π_S (Wu-Wang)',ylabel=label)
        if len(valid):ax.legend()
        if not np.isfinite(order[metric]).all():ax.text(.02,.98,'Endzeitkriterium nicht für alle Fälle erfüllt',transform=ax.transAxes,va='top',fontsize=10)
        ax.text(.02,.02,'Kugel wird nach Abschluss ergänzt; keine Fits',transform=ax.transAxes,fontsize=10)
        ax.plot([],[],color='gray',ls=':',label='Lu et al. 2026: weniger E, kürzerer Auslauf')
        ax.legend(fontsize=10)
        export_plot(fig,out,'kap7_4_pis_'+slug,order)
    # More intuitive ΔL/L0 companion avoids ambiguity in the specification's l−1.
    fig,ax=plt.subplots(figsize=(5.5,3.4))
    for _,r in end.iterrows():ax.scatter(r.Pi_S,r.l,color=COLORS[r.form],label=r.form)
    ax.set(xlabel='Π_S',ylabel='l = ΔL/L0');ax.legend()
    export_plot(fig,out,'kap7_4_pis_delta_l',end)
    fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4));edges=np.linspace(0,90,19)
    histogram=[]
    for form,data in tables['orientation'].groupby('form'):
        counts,_=np.histogram(data.angle_to_flow.dropna(),bins=edges)
        ax.stairs(counts/counts.sum() if counts.sum() else counts,edges,color=COLORS[form],label=form)
        histogram.extend(dict(form=form,angle_lower=a,angle_upper=b,fraction=float(n/counts.sum()) if counts.sum() else np.nan)
                         for a,b,n in zip(edges[:-1],edges[1:],counts))
    ax.set(xlabel='Winkel zur Fließrichtung [Grad]',ylabel='Partikelanteil pro Winkelbin');ax.legend()
    export_plot(fig,out,'kap7_5_orientierung',pd.DataFrame(histogram))
    fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4))
    for form,data in tables['settling'].groupby('form'):
        early=data[data.t<=10]
        ax.plot(early.t,-early.mean_vy_over_us,color=COLORS[form],label=form)
    ax.set(xlabel='t*',ylabel='Oberste Anfangsschicht: mittleres -v_y/u_s');ax.legend()
    export_plot(fig,out,'kap7_5_oberschicht_sinken',tables['settling'])
    sinking=pd.DataFrame([initial_settling_indicator(data.t,data.mean_vy_over_us,form)
                          for form,data in tables['settling'].groupby('form')])
    sinking.to_csv(out/'hindered_settling_indiz.csv',index=False)
    fig,ax=plt.subplots(figsize=(5.5,3.4))
    for _,row in sinking.iterrows():
        ax.scatter(row.Pi_S,row.mean_initial_sink_over_us,color=COLORS[row.form],label=row.form)
    ax.set(xlabel='Π_S (Wu-Wang)',ylabel='Anfangsschicht: mittleres -v_y/u_s, t*=0..5')
    ax.legend();export_plot(fig,out,'kap7_5_oberschicht_pis',sinking)
    if 'pressure' in tables:
        fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4))
        for form,data in tables['pressure'].groupby('form'):
            ax.plot(data.t,data.excess_pressure_mean,'o-',color=COLORS[form],label=form)
        ax.axhline(0,color='gray',lw=.8)
        ax.set(xlabel='t*',ylabel='Druckdifferenz zum entfernten Fluid');ax.legend()
        export_plot(fig,out,'kap7_5_druckdifferenz',tables['pressure'])
    fig,axs=plt.subplots(1,len(statuses),figsize=(plotstyle.figwidth_full,3.4),squeeze=False)
    for ax,status in zip(axs[0],statuses):
        data=pd.read_csv(out/status['run_id']/'energy_budget.csv')
        for metric,label in [('E_pot','Potentiell'),('E_kin_trans','Translation'),('E_kin_rot','Rotation, gefiltert')]:
            ax.plot(data.t,data[metric],label=label)
        ax.set(xlabel='t*',ylabel=status['form']+': Energie E*');ax.legend(fontsize=10)
    export_plot(fig,out,'kap7_5_partikelenergien',tables['budget'])
    contact=pd.concat([pd.read_csv(out/s['run_id']/'contact_summary.csv').assign(form=s['form']) for s in statuses],ignore_index=True)
    contact.to_csv(out/'kontaktstatistik_formen.csv',index=False)
    (out/'kontaktstatistik_tabellenzeilen.tex').write_text(latex_table(contact,[('form','Form'),('mean_contact_count','Mittlere Kontaktzahl')]))
    grid=[]
    for form,coarse_path,fine_path in grid_pairs or []:
        coarse=pd.read_csv(coarse_path).iloc[0];fine=pd.read_csv(fine_path).iloc[0]
        g=grid_comparison(coarse,fine);g['form']=form;grid.append(g)
    grid_table=pd.concat(grid,ignore_index=True) if grid else pd.DataFrame([dict(form=s['form'],status='No matched coarse/fine pair supplied') for s in statuses])
    grid_table.to_csv(out/'gitterstudie_endwerte.csv',index=False)
    if grid:
        (out/'gitterstudie_tabellenzeilen.tex').write_text(latex_table(grid_table,[('form','Form'),('metric','Kennwert'),('coarse','Grob'),('fine','Fein'),('relative_difference','Relative Differenz')]))
    # Unit-amplitude exponent illustrations are never presented as absolute laws.
    shapes=dry_scaling(np.geomspace(.1,10,120));shapes.to_csv(out/'literatur_proportionalformen.csv',index=False)
    literature_status='Absolute amplitudes not supplied; exponent routines ready; no invented fitted reference lines'
    if literature:
        refs=pd.read_csv(literature)
        required={'label','coefficient','exponent','chi_min','chi_max','metric','source_url'}
        if not required<=set(refs) or refs[list(required)].isna().any().any():raise ValueError('Incomplete sourced literature reference table')
        numeric=refs[['coefficient','exponent','chi_min','chi_max']].apply(pd.to_numeric,errors='raise')
        if (not np.isfinite(numeric).all().all() or (numeric.coefficient<=0).any()
                or (numeric.chi_min<=0).any() or (numeric.chi_max<=numeric.chi_min).any()
                or not set(refs.metric)<={'l','h'}):
            raise ValueError('Invalid literature coefficients, domains or metric')
        for metric,ylabel in [('l','ΔL/L0'),('h','H_end/H0')]:
            fig,ax=plt.subplots(figsize=(plotstyle.figwidth_full,3.4));inputs=[]
            for _,r in refs[refs.metric==metric].iterrows():
                chi=np.geomspace(r.chi_min,r.chi_max,100);value=r.coefficient*chi**r.exponent
                ax.plot(chi,value,label=r.label);inputs.append(pd.DataFrame(dict(chi=chi,value=value,label=r.label,source_url=r.source_url)))
            for _,r in end.iterrows():
                state='quasistatisch' if r.quasistatic else 'Abschluss-Snapshot'
                ax.scatter(r.chi,r[metric],color=COLORS[r.form],marker='o' if r.quasistatic else 'x',
                           label=r.form+' (eingetaucht; '+state+')')
                inputs.append(pd.DataFrame([dict(chi=r.chi,value=r[metric],label=r.form,source_url=str(r['case']))]))
            ax.set(xlabel='χ = H0/L0',ylabel=ylabel,xscale='log');ax.legend(fontsize=10)
            export_plot(fig,out,'kap7_4_literatur_'+metric,pd.concat(inputs,ignore_index=True))
        literature_status='Sourced absolute reference curves exported; dry and immersed conditions labeled separately'
    return dict(P0='Completed for supplied cases; quasistatic validity recorded per row',
        P1_1='Two-shape plots; collapse duration uses front t95; Kugel pending; no fitted trends',
        P1_2='Completed alpha sweep and reference stability check',P1_3='Completed' if grid else 'Routine ready; no matched grid pair supplied',
        P1_4=literature_status,P1_5='Completed normalized spanwise scatter',
        P2_1='Particle energies completed; true dissipation requires complete fluid/work budget',
        P2_2='Completed orientation distribution',P2_3='Measured if RK3 exports exist; otherwise explicitly unavailable',
        P2_4='Completed porosity evolution and ambient-referenced pressure',P2_5='Completed initial top-layer sinking indicator')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',required=True,metavar='FORM=PATH')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=81,help='Geometry samples; 0 analyzes every stored particle frame')
    parser.add_argument('--z-slices',type=int,default=8)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--alpha',type=float,default=.6)
    parser.add_argument('--rotation-share-threshold',type=float,default=.5)
    parser.add_argument('--energy-fraction',type=float,default=.01)
    parser.add_argument('--hold-time',type=float,default=10.)
    parser.add_argument('--reuse-energy',action='store_true',help='Reuse this delivery\'s verified energy tables, matching options/frame times')
    parser.add_argument('--grid-pair',nargs=3,action='append',metavar=('FORM','COARSE_END_CSV','FINE_END_CSV'))
    parser.add_argument('--literature',type=Path,help='CSV with label, coefficient, exponent, chi_min, chi_max, metric, source_url')
    parser.add_argument('--plots-only',action='store_true',help='Regenerate report from existing per-case status.json products')
    args=parser.parse_args(argv)
    options=CollapseOptions(alpha=args.alpha,frames=args.frames,z_slices=args.z_slices,workers=args.workers,
        rotation_share_threshold=args.rotation_share_threshold,energy_fraction=args.energy_fraction,hold_time=args.hold_time)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.literature is None and (out/'literatur_referenzen.csv').is_file():
        args.literature=out/'literatur_referenzen.csv'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log')])
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    statuses=[]
    for supplied in args.case:
        form,source=supplied.split('=',1)
        run_id='prolat_uncut' if form=='Prolat' else form.lower()
        if any(s['run_id']==run_id for s in statuses):raise ValueError('One reference case per form required')
        if args.plots_only:status=json.loads((out/run_id/'status.json').read_text())
        else:status=analyze_case(form,source,out/run_id,options,reuse_energy=args.reuse_energy)
        status['run_id']=run_id;statuses.append(status)
        (out/'status.json').write_text(json.dumps(statuses,indent=2,default=str)+'\n')
    report_status=report(out,statuses,options,args.grid_pair,args.literature)
    (out/'auftragsstatus.json').write_text(json.dumps(report_status,indent=2)+'\n')
    readme='''# P0–P2: Oblat und Prolat Uncut

Neue Lieferung; Original-Cases unverändert. Kugel ist ausgenommen.
Details zu Einheiten, Endzeitkriterium, Alpha-Auswahl, Kontaktdefinition und
bekannten Grenzen stehen in den Fall-READMEs. Alle Tabellen enthalten gültige
Messwerte oder explizit fehlende Werte; Endzustandsflags beachten.

Die Auftragstabelle fordert l−1 gegen Π_S, definiert aber l bereits als
ΔL/L0. Deshalb gibt es beide Darstellungen, klar beschriftet.
Gittervergleich benötigt zwei passende Auswertungs-Endtabellen, nicht zwei
verschiedene Partikelformen. Keine Konvergenzordnung aus zwei Gittern.
Literaturkoeffizienten müssen aus belegten Quellen kommen und werden mit
Gültigkeitsbereichen explizit an die Routine übergeben.

Aufruf (aus dem Workspace, Paket per PYTHONPATH verfügbar):

```bash
PYTHONPATH=granular_analysis/src python -m granular_analysis.scripts.run_collapse_analysis \\
 --case Oblat=oblaten/new-prime-gran-coll-2000_NewInput \\
 --case Prolat=prolaten/new-prime-gran-coll-2000Uncut_fp_collision/results_complete_20261002 \\
        --out bilder/generiert/p0_p2_oblat_prolat_uncut_20261002 --reuse-energy
```

`--rotation-share-threshold 1` deaktiviert den Filter. `--reuse-energy` ist nur
für exakt passende Options- und Frame-Tabellen vorgesehen. Ohne diese Option
werden Monitorquellen erneut eingelesen und gehasht. `--plots-only` regeneriert
Darstellungen aus vorhandenen Fall-Produkten.
`--grid-pair FORM grob/endzustand.csv fein/endzustand.csv` ergänzt den
Zwei-Gitter-Vergleich; `--literature references.csv` ergänzt belegte Trocken-
Referenzkurven. Die Routinen sind in `analysis/collapse.py` wiederverwendbar.
'''
    (out/'README.md').write_text(readme+'\nStatus:\n'+ '\n'.join('- '+k+': '+v for k,v in report_status.items())+'\n')
    provenance=dict(options=asdict(options),case_products={s['run_id']:sha256(out/s['run_id']/'herkunft.json') for s in statuses},
        scripts={str(Path(__file__).resolve()):sha256(__file__),
        str(Path(__file__).resolve().parents[1]/'analysis/collapse.py'):sha256(Path(__file__).resolve().parents[1]/'analysis/collapse.py')},
        command=sys.argv)
    fonts=Path(__file__).resolve().parents[3]/'thesis/fonts'
    provenance['font_sha256']={str(p.resolve()):sha256(p) for p in fonts.glob('NotoSans-*.ttf')}
    assignment=Path(__file__).resolve().parents[4]/'codex_auftrag_paketerweiterung_011026.md'
    if assignment.exists():provenance['assignment_sha256']=sha256(assignment)
    if args.literature:provenance['literature']={str(args.literature.resolve()):sha256(args.literature)}
    if args.grid_pair:provenance['grid_sources']={str(Path(p).resolve()):sha256(p) for _,a,b in args.grid_pair for p in (a,b)}
    (out/'herkunft.json').write_text(json.dumps(provenance,indent=2)+'\n')
    logging.info('P0–P2 report complete: %s',out)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
