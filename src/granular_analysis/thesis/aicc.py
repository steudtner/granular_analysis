"""AICc-Ergänzung aus R-Exporten; archivierte AIC/BIC bleiben unverändert."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'bilder/generiert'


def augment(path, resolution, allow_legacy=False):
    """slope_results_R.csv lesen; n/k direkt oder explizit aus Legacy-AIC/BIC rekonstruieren."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'R-Fitexport fehlt: {path}')
    df = pd.read_csv(path)
    required = {'model', 'aic', 'bic'}
    if not required <= set(df):
        raise ValueError(f'{path}: Spalten fehlen: {required-set(df)}')
    legacy = not {'n', 'k'} <= set(df)
    if legacy:
        if not allow_legacy:
            raise ValueError(f'{path}: n/k fehlen; Fit erneut exportieren oder --legacy verwenden')
        # Existing univariate segmented model: intercept, slope, slope changes,
        # estimated breakpoints; plus one variance parameter.
        b = pd.to_numeric(df['n_breakpoints']).to_numpy()
        if not np.all((b >= 0) & (b == np.floor(b))):
            raise ValueError('Ungültige Knickpunktzahl')
        df['k'] = (3 + 2*b).astype(int)
        n_float = np.exp(2 + (df.bic-df.aic)/df.k)
        if not np.allclose(n_float, np.rint(n_float), rtol=0, atol=1e-7):
            raise ValueError('Legacy-BIC impliziert keine ganzzahlige Punktzahl')
        df['n'] = np.rint(n_float).astype(int)
    numeric = df[['n', 'k', 'aic', 'bic']].to_numpy(float)
    if not np.isfinite(numeric).all() or (df.n <= df.k+1).any():
        raise ValueError('AICc benötigt endliche Werte und n > k+1')
    if not ((df.n == np.floor(df.n)) & (df.k == np.floor(df.k)) & (df.k > 0)).all():
        raise ValueError('n/k müssen positive ganze Zahlen sein')
    if df.n.nunique() != 1 or df.model.duplicated().any():
        raise ValueError('Modelle müssen dieselbe Stichprobe und eindeutige Namen haben')
    if not np.allclose(df.bic, df.aic + df.k*(np.log(df.n)-2), atol=1e-8, rtol=1e-10):
        raise ValueError('AIC/BIC passen nicht zu n/k; Definition nicht automatisch ändern')
    result = pd.DataFrame({'modell': df.model, 'aufloesung': resolution,
        'n': df.n.astype(int), 'k': df.k.astype(int), 'AIC': df.aic, 'BIC': df.bic,
        'AICc': df.aic + 2*df.k*(df.k+1)/(df.n-df.k-1)})
    provenance = {'source': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'n_k_method': 'legacy: k=3+2*n_breakpoints; n=exp(2+(BIC-AIC)/k)' if legacy
                      else 'R: nobs(fit), length(coef(fit))+1',
        'legacy': legacy, 'resolution': resolution}
    return result, provenance


def latex_rows(df):
    """CSV-Kriterien als booktabs-Zeilen mit Auflösung, n und k ausgeben."""
    rows = [r'\toprule', r'Modell & $D_{\mathrm{eq}}/\Delta$ & $n$ & $k$ & AIC & BIC & AICc \\', r'\midrule']
    for r in df.itertuples(index=False):
        rows.append(f'{r.modell} & {r.aufloesung} & {r.n} & {r.k} & {r.AIC:.3f} & {r.BIC:.3f} & {r.AICc:.3f} '+r'\\')
    return '\n'.join(rows + [r'\bottomrule', ''])


def main():
    """Zwei aufgelöste Fitexporte zu CSV, Tabellenzeilen und Herkunftsprotokoll verbinden."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delta16', type=Path, required=True)
    p.add_argument('--delta20', type=Path, required=True)
    p.add_argument('--legacy', action='store_true')
    args = p.parse_args()
    data, provenance = zip(*(augment(path, res, args.legacy)
         for path,res in [(args.delta16,16),(args.delta20,20)]))
    for d in data:
        if len(d) != 6:
            raise ValueError('Erwartet: Linear und fünf bestehende segmentierte Modelle')
    df = pd.concat(data, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT/'kap7_aicc.csv', index=False, float_format='%.12g')
    (OUT/'kap7_1_tabellenzeilen.tex').write_text(latex_rows(df[df.aufloesung==16]), encoding='utf-8')
    (OUT/'kap6_2_tabellenzeilen.tex').write_text(latex_rows(df), encoding='utf-8')
    (OUT/'kap7_aicc_herkunft.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
