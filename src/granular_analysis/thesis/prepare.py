"""Vorhandene Exporte inventarisieren und konservativ in Plot-Tabellen übertragen."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'bilder/generiert'


def cases():
    """Benutzerzuordnung: primeJainSU, genau der Oblat-Fall, alle direkten Prolat-Fälle ohne old/."""
    yield 'Kugel', ROOT/'DiscretizationGranularCollapse/primeJainSU'
    yield 'Oblat', ROOT/'oblaten/new-prime-gran-coll-2000_NewInput'
    for p in sorted((ROOT/'prolaten').iterdir()):
        if p.is_dir() and p.name!='old' and (p/'input.dat').is_file():
            yield 'Prolat',p


def main():
    """energy.csv und slice_summary.csv übernehmen; Alpha-Formgrenzen aus gespeicherten JSONs.

    Keine Endzeitentscheidung: alle Geometrien sind letzte gespeicherte Snapshots.
    Die aktuelle PRIME-Konfiguration verwendet D_eq=1 als nominelle Referenzlänge.
    """
    OUT.mkdir(parents=True,exist_ok=True)
    energy=[]; profiles=[]; alpha=[]; areas=[]; inventory=[]
    for form,case in cases():
        results=case/'results_granular_analysis'
        record={'form':form,'fall':case.name,'path':str(case.relative_to(ROOT)),
                'status':{},'sources':[]}
        for rel in ['energy.csv','slices/slice_summary.csv','alpha_sensitivity/alpha_sensitivity.csv']:
            p=results/rel
            record['status'][rel]='vorhanden' if p.is_file() else 'fehlt'
            if p.is_file():
                record['sources'].append({'path':str(p.relative_to(ROOT)),
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        p=results/'energy.csv'
        if p.is_file():
            d=pd.read_csv(p)
            col='translational_energy_norm' if 'translational_energy_norm' in d else 'kinetic_energy_norm'
            # Legacy sphere export is translational only; don't mix it with newer totals.
            e=d[['time',col]].rename(columns={'time':'t',col:'E_trans_star'})
            e['form']=form;e['fall']=case.name
            if e.t.duplicated().any() or not np.isfinite(e[['t','E_trans_star']]).all().all():
                record['status']['energy.csv']='ungültige/mehrdeutige Zeitreihe'
            else: energy.append(e)
            record['energy_column']=col
        p=results/'slices/slice_summary.csv'
        if p.is_file():
            d=pd.read_csv(p).rename(columns={'z':'z_star','porosity':'phi'})
            d['form']=form;d['fall']=case.name
            profiles.append(d[['form','fall','z_star','phi']])
        sensitivity=results/'alpha_sensitivity/alpha_sensitivity.csv'
        if sensitivity.is_file():
            d=pd.read_csv(sensitivity)
            selected=d[np.isclose(d.alpha,.4)|np.isclose(d.alpha,.6)|np.isclose(d.alpha,.8)].copy()
            if len(selected)==3:
                selected['form']=form;selected['fall']=case.name
                selected=selected.rename(columns={'alphashape_area':'A_alpha_star'})
                areas.append(selected[['form','fall','alpha','A_alpha_star']])
            rows=[]
            for a in [.4,.6,.8]:
                row=d[np.isclose(d.alpha,a)]
                p=results/f'slope_alpha_sensitivity/alpha_{a:.1f}/analysis_state.json'
                if len(row)!=1 or not p.is_file():
                    rows=[];break
                state=json.loads(p.read_text()).get('2d')
                if not state or 'shape' not in state:
                    rows=[];break
                if not np.isclose(state['A_alpha'],row.iloc[0].alphashape_area,rtol=1e-10):
                    rows=[];break
                shape=np.asarray(state['shape'])
                # x=0 is the upstream wall; z_max in upper-edge CSVs is vertical y.
                rows.append({'form':form,'fall':case.name,'alpha':a,
                     'L_end_star':shape[:,0].max(),'A_alpha_star':state['A_alpha']})
                record['sources'].append({'path':str(p.relative_to(ROOT)),
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            alpha.extend(rows)
            record['status']['alpha_geometry']='vollständig, Flächen konsistent' if rows else 'fehlt/inkonsistent'
        inventory.append(record)
    for name,frames in [('kap7_1_translation_eingabe',energy),('kap7_3_porositaet_eingabe',profiles)]:
        if frames: pd.concat(frames,ignore_index=True).to_csv(OUT/f'{name}.csv',index=False)
    if areas: pd.concat(areas,ignore_index=True).to_csv(OUT/'kap6_2_alpha_flaeche_eingabe.csv',index=False)
    if alpha: pd.DataFrame(alpha).to_csv(OUT/'kap6_2_alpha_eingabe.csv',index=False)
    (OUT/'anhang_dateninventar.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__': main()
