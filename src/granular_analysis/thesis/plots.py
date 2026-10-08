"""Sieben Plot-Routinen für dokumentierte CSV-Eingaben; keine Ersatzdaten."""

from granular_analysis.plotting import plotstyle
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from granular_analysis.plotting import plotstyle as style

D = r'D_{\mathrm{eq}}'


def read_table(path, columns):
    """CSV aus path lesen; fehlende Dateien, Spalten und ungültige Zahlen ablehnen."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Eingabedatei fehlt: {path}; erforderliche Spalten: {", ".join(columns)}')
    df = pd.read_csv(path)
    if df.empty or not set(columns) <= set(df):
        raise ValueError(f'{path}: leere Tabelle oder fehlende Spalten {set(columns)-set(df)}')
    numeric = [c for c in columns if c not in ('form', 'fall', 'modell', 'quelle', 'methode')]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all():
            raise ValueError(f'{path}: nicht-endliche Werte in {col}')
    if 'form' in columns and not set(df.form) <= set(style.COLORS):
        raise ValueError('form muss Kugel, Prolat oder Oblat sein')
    return df


def grouped(df, x):
    """Fälle getrennt und sortiert halten; keine Mittelung verschiedener Prolat-Fälle."""
    for (form, case), group in df.groupby(['form', 'fall'], sort=True):
        group = group.sort_values(x)
        if group[x].duplicated().any():
            raise ValueError(f'{case}: doppelte {x}-Werte; Restart-Historie zuerst bereinigen')
        yield form, case, group


def finish(fig, axes, name):
    """Legenden außerhalb der Datenfläche setzen und beide Vektorformate speichern."""
    for ax in np.atleast_1d(axes):
        if ax.get_legend_handles_labels()[0]:
            labels=ax.get_legend_handles_labels()[1]
            if max(map(len, labels)) > 25:
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), borderaxespad=0)
            else:
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    paths = style.save(fig, name)
    plt.close(fig)
    return paths


def kinetic_energy(path, endpoints):
    """energy.csv: E_kin/(rho_ref*g*D_eq^4); Endzeiten aus quasistatisch.csv markieren.

    Eingabe: form,fall,t,E_kin_star; t in Sekunden. Nur konsistente Energiedefinitionen
    zusammenführen. Endzeiten werden nicht aus dem letzten gespeicherten Frame geraten.
    """
    df = read_table(path, ['form','fall','t','E_kin_star'])
    end = read_table(endpoints, ['fall','t_qs','methode'])
    if end.fall.duplicated().any() or not set(df.fall) <= set(end.fall):
        raise ValueError('Genau eine dokumentierte quasistatische Endzeit pro Fall erforderlich')
    if (df.E_kin_star < 0).any() or end.methode.isna().any():
        raise ValueError('Energie muss nichtnegativ, Endzeitmethode dokumentiert sein')
    fig, ax = plt.subplots(figsize=(style.figwidth_full,3.2), layout='constrained')
    for form,case,g in grouped(df,'t'):
        tq = float(end.set_index('fall').loc[case,'t_qs'])
        if not g.t.min() <= tq <= g.t.max():
            raise ValueError(f'{case}: t_qs liegt außerhalb der Messreihe')
        ax.plot(g.t,g.E_kin_star,color=style.COLORS[form],label=f'{form}: {case}')
        ax.axvline(tq,ls=':',color=style.COLORS[form])
        ax.plot(tq,np.interp(tq,g.t,g.E_kin_star),'o',color=style.COLORS[form])
    ax.set(xlabel='Simulationszeit t (s)',ylabel=r'Kinetische Energie $E_{\mathrm{kin}}^*$')
    return finish(fig,ax,'kap7_1_kinenergie')


def translational_preview(path):
    """Vorbereitete Alt-/Neuexporte: nur Translationsenergie, ohne offene Endzeitmarkierung."""
    df=read_table(path,['form','fall','t','E_trans_star'])
    # One figure per prolate case: exactly one curve for each shape when available.
    paths=[]
    sphere=df[df.form=='Kugel']; oblate=df[df.form=='Oblat']
    if sphere.empty or oblate.empty:
        raise ValueError('Kugel- und Oblat-Zeitreihe erforderlich')
    for case in sorted(df.loc[df.form=='Prolat','fall'].unique()):
        selected=pd.concat([sphere,oblate,df[(df.form=='Prolat') & (df.fall==case)]])
        fig,ax=plt.subplots(figsize=(style.figwidth_full,3.2),layout='constrained')
        for form,_,g in grouped(selected,'t'):
            ax.plot(g.t,g.E_trans_star,color=style.COLORS[form],label=form)
        ax.set(xlabel='Simulationszeit t (s)',ylabel=r'Translationsenergie $E_{\mathrm{trans}}^*$')
        import re
        slug=re.sub('[^a-z0-9]+','_',case.lower()).strip('_')
        paths+=finish(fig,ax,'kap7_1_translation_ohne_endzeit_'+slug)
    return paths


def runout(path):
    """runout.csv: L/D_eq und H/D_eq aus zeitaufgelösten Alpha-Shape-Grenzen darstellen.

    Benötigt form,fall,t,L_star,H_star; gleiche Ursprünge und Alpha-Definition je Fall.
    """
    df = read_table(path,['form','fall','t','L_star','H_star'])
    paths=[]
    for col,label,name in [('L_star',rf'Ablauflänge $L/{D}$','kap7_1_runout'),
                           ('H_star',rf'Resthöhe $H/{D}$','kap7_2_resthoehe')]:
        if (df[col]<0).any(): raise ValueError('Negative Länge/Höhe')
        fig,ax=plt.subplots(figsize=(style.figwidth_full,3.2),layout='constrained')
        for form,case,g in grouped(df,'t'):
            ax.plot(g.t,g[col],color=style.COLORS[form],label=f'{form}: {case}')
        ax.set(xlabel='Simulationszeit t (s)',ylabel=label)
        paths += finish(fig,ax,name)
    return paths


def alpha_sensitivity(path):
    """alpha.csv: L_end/D_eq und A_alpha/D_eq² für identische Endframes bei drei Alpha-Werten."""
    df=read_table(path,['form','fall','alpha','L_end_star','A_alpha_star'])
    for _,case,g in grouped(df,'alpha'):
        if len(g)!=3 or not np.allclose(g.alpha,[.4,.6,.8]):
            raise ValueError(f'{case}: genau alpha=0.4/0.6/0.8 erforderlich')
    paths=[]
    for col,label,name in [('L_end_star',rf'Ablauflänge $L_{{\mathrm{{end}}}}/{D}$','kap6_2_alpha_runout'),
                           ('A_alpha_star',rf'Alpha-Shape-Fläche $A_\alpha/{D}^2$','kap6_2_alpha_flaeche')]:
        fig,ax=plt.subplots(figsize=(style.figwidth_full,3.2),layout='constrained')
        for form,case,g in grouped(df,'alpha'):
            ax.plot(g.alpha,g[col],'o-',color=style.COLORS[form],label=f'{form}: {case}')
        ax.set(xlabel=r'Alpha-Parameter $\alpha$',ylabel=label,xticks=[.4,.6,.8])
        paths+=finish(fig,ax,name)
    return paths


def alpha_area(path):
    """alpha_sensitivity.csv: vorhandene Flächen isoliert plotten, ohne fehlendes L_end zu ersetzen."""
    df=read_table(path,['form','fall','alpha','A_alpha_star'])
    fig,ax=plt.subplots(figsize=(style.figwidth_full,3.2),layout='constrained')
    for form,case,g in grouped(df,'alpha'):
        if len(g)!=3 or not np.allclose(g.alpha,[.4,.6,.8]):
            raise ValueError(f'{case}: genau alpha=0.4/0.6/0.8 erforderlich')
        ax.plot(g.alpha,g.A_alpha_star,'o-',color=style.COLORS[form],label=f'{form}: {case}')
    ax.set(xlabel=r'Alpha-Parameter $\alpha$',ylabel=rf'Alpha-Shape-Fläche $A_\alpha/{D}^2$',xticks=[.4,.6,.8])
    return finish(fig,ax,'kap6_2_alpha_flaeche')


def grid_convergence(path):
    """gitter.csv: Endwerte für Δ/D_eq=1/16,1/20; relative Differenz, keine Konvergenzordnung."""
    df=read_table(path,['aufloesung','L_end_star','H_end_star','phi_z'])
    df=df.sort_values('aufloesung')
    if df.aufloesung.tolist()!=[16,20]:
        raise ValueError('Genau zwei vergleichbare Endzustände Δ16/Δ20 erforderlich')
    if not df.phi_z.between(0,1).all(): raise ValueError('Porosität außerhalb [0,1]')
    paths=[]
    changes=[]
    for col,label in [('L_end_star',rf'$L_{{\mathrm{{end}}}}/{D}$'),
                       ('H_end_star',rf'$H_{{\mathrm{{end}}}}/{D}$'),('phi_z',r'$\phi_z$')]:
        fig,ax=plt.subplots(figsize=(style.figwidth_half,2.6),layout='constrained')
        ax.plot(1/df.aufloesung,df[col],'o-')
        ax.set(xlabel=rf'Gitterweite $\Delta\,{D}^{{-1}}$',ylabel=label,
               xticks=[1/20,1/16],xticklabels=['1/20','1/16'])
        paths+=finish(fig,ax,'kap6_2_gitter_'+col.lower())
        coarse,fine=df[col].to_numpy()
        changes.append({'groesse':col,'delta16':coarse,'delta20':fine,
                        'relative_differenz':abs(coarse-fine)/abs(fine) if fine else np.nan})
    pd.DataFrame(changes).to_csv(style.OUTPUT_DIR/'kap6_2_gitter_differenzen.csv',index=False)
    return paths+[style.OUTPUT_DIR/'kap6_2_gitter_differenzen.csv']


def model_selection(path):
    """kap7_aicc.csv: unveränderte AIC/BIC und zusätzlich AICc je Auflösung vergleichen."""
    df=read_table(path,['modell','aufloesung','n','k','AIC','BIC','AICc'])
    paths=[]
    for resolution,g in df.groupby('aufloesung',sort=True):
        fig,ax=plt.subplots(figsize=(style.figwidth_full,3.4),layout='constrained')
        g=g.sort_values('k')
        for metric,marker in [('AIC','o'),('BIC','s'),('AICc','^')]:
            ax.plot(g.modell,g[metric],marker=marker,label=metric)
        ax.set(xlabel='Modell (Benennung nach Knickpunktzahl)',ylabel='Informationskriterium')
        ax.tick_params(axis='x',rotation=25)
        paths+=finish(fig,ax,f'kap7_2_modellvergleich_delta{int(resolution)}')
    return paths


def porosity(path):
    """porositaet.csv: phi(z) aus geometrisch bezeichneten Endzustands-Schnitten ohne Glättung."""
    df=read_table(path,['form','fall','z_star','phi'])
    if not df.phi.between(0,1).all(): raise ValueError('Porosität außerhalb [0,1]')
    fig,ax=plt.subplots(figsize=(style.figwidth_full,3.2),layout='constrained')
    for index,(form,case,g) in enumerate(grouped(df,'z_star')):
        ax.plot(g.z_star,g.phi,color=style.COLORS[form],
                ls=['-','--',':','-.'][index % 4],label=f'{form}: {case}')
    ax.set(xlabel=rf'Schnittposition $z\,{D}^{{-1}}$',ylabel=r'Porosität $\phi$')
    return finish(fig,ax,'kap7_3_porositaet')


# Verified primary-paper exponents, transformed to l=(L_end-L0)/L0, h=H_end/L0.
# No fabricated prefactors: proportional laws require a documented reference anchor.
SCALINGS = {
 'balmforth_wide_l': ('l',.9,2.,np.inf,'Balmforth & Kerswell 2005, Gl. 4.2'),
 'balmforth_wide_h': ('h',.4,2.,np.inf,'Balmforth & Kerswell 2005, Gl. 4.1; h=chi/(H0/Hend)'),
 'balmforth_narrow_l': ('l',.65,3.,30.,'Balmforth & Kerswell 2005, Gl. 5.2 / Abb. 19'),
 'balmforth_narrow_h': ('h',.5,2.,np.inf,'Balmforth & Kerswell 2005, Gl. 5.1'),
 'lajeunesse_channel_l_low': ('l',1.,0.,3.,'Lajeunesse 2005, S. 103302-14'),
 'lajeunesse_channel_l_high': ('l',2/3,3.,np.inf,'Lajeunesse 2005, S. 103302-14'),
 'lajeunesse_channel_h_low': ('h',1.,0.,.7,'Lajeunesse 2005, S. 103302-14'),
 'lajeunesse_channel_h_high': ('h',1/3,.7,np.inf,'Lajeunesse 2005, S. 103302-14'),
 'lube_axisymmetric_l_low': ('l',1.,0.,1.7,'Lube 2004, Gl. 3'),
 'lube_axisymmetric_l_high': ('l',.5,1.7,np.inf,'Lube 2004, Gl. 4'),
 'lube_axisymmetric_h_low': ('h',1.,0.,1.7,'Lube 2004, Gl. 8'),
 'lube_axisymmetric_h_mid': ('h',1/6,1.7,10.,'Lube 2004, Gl. 9'),
}


def scaling(path, references):
    """skalierung.csv: chi=H0/L0, l=(Lend-L0)/L0, h=Hend/L0 gegen belegte Potenzgesetze.

    references JSON: Liste {law,chi_ref,value_ref,source}; Referenzamplituden explizit,
    da Balmforth materialabhängige Vorfaktoren hat. Keine universelle Amplitude annehmen.
    """
    df=read_table(path,['form','fall','chi','l','h'])
    if not Path(references).is_file():
        raise FileNotFoundError(f'Belegte Skalierungsamplituden fehlen: {references}')
    refs=json.loads(Path(references).read_text())
    if not refs: raise ValueError('Mindestens eine Literaturreferenz erforderlich')
    if (df[['chi','l','h']]<=0).any().any(): raise ValueError('Logarithmische Achsen benötigen positive Werte')
    paths=[]
    for metric,label in [('l',r'Ablauflänge $l=(L_{\mathrm{end}}-L_0)/L_0$'),
                         ('h',r'Resthöhe $h=H_{\mathrm{end}}/L_0$')]:
        fig,ax=plt.subplots(figsize=(style.figwidth_full,3.2),layout='constrained')
        for form,case,g in grouped(df,'chi'):
            ax.loglog(g.chi,g[metric],'o',color=style.COLORS[form],label=f'{form}: {case}')
        count=0
        for ref in refs:
            m,exponent,lower,upper,source=SCALINGS[ref['law']]
            if m!=metric: continue
            x0,y0=float(ref['chi_ref']),float(ref['value_ref'])
            if not np.isfinite([x0,y0]).all() or x0<=0 or y0<=0 or not ref['source'] or not lower<=x0<=upper:
                raise ValueError('Gültiger Referenzpunkt samt Quelle innerhalb des Regimes erforderlich')
            lo,hi=max(lower,df.chi.min()),min(upper,df.chi.max())
            if hi<=lo: continue
            x=np.geomspace(lo,hi,100)
            ax.loglog(x,y0*(x/x0)**exponent,'--',label=source)
            count+=1
        if not count: raise ValueError(f'Keine passende Literaturkurve für {metric} im Datenbereich')
        ax.set(xlabel=r'Anfangsseitenverhältnis $\chi=H_0/L_0$',ylabel=label)
        paths+=finish(fig,ax,'kap7_2_skalierung_'+metric)
    return paths


ROUTINES={'translation':translational_preview,'energie':kinetic_energy,'runout':runout,'alpha':alpha_sensitivity,'alpha_flaeche':alpha_area,
          'gitter':grid_convergence,'kriterien':model_selection,'porositaet':porosity,'skalierung':scaling}


def main():
    """Eine CSV-Routine ausführen; fehlende Eingaben erzeugen FileNotFoundError."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('routine',choices=ROUTINES)
    p.add_argument('input',type=Path)
    p.add_argument('--zusatz',type=Path,help='Quasistatische Endzeiten oder Skalierungsreferenzen')
    p.add_argument('--svg-fonttype',choices=['none','path'],default='none')
    p.add_argument('--font',type=Path)
    a=p.parse_args()
    font=a.font or Path(__file__).resolve().parents[3]/'thesis/fonts/NotoSans-Regular.ttf'
    if font.is_file():
        from matplotlib import font_manager
        for file in font.parent.glob('NotoSans-*.ttf'): font_manager.fontManager.addfont(str(file))
    style.configure(a.svg_fonttype, str(font))
    extra=[]
    if a.routine in ('energie','skalierung'):
        extra=[a.zusatz or Path('quasistatisch.csv' if a.routine=='energie' else 'skalierungsreferenzen.json')]
    for path in ROUTINES[a.routine](a.input,*extra): print(path)

if __name__=='__main__': main()
