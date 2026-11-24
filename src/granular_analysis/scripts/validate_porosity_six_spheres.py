"""Analytic benchmark of six touching spheres in a triangular lattice layer."""

from granular_analysis.plotting import plotstyle
import argparse
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from granular_analysis.analysis import porosity2d
from granular_analysis.analysis.porosity_time import compute_porosity_profile
from granular_analysis.geometry.ellipsoid import ellipse_points


def six_spheres():
    """Unit spheres, nine touching pairs, center plane z=2, periodic Lz=6."""
    return [dict(id=i, x=np.array([x, y, 2.]), q=np.array([1., 0., 0., 0.]),
                 a=1., b=1., c=1.)
            for i, (x, y) in enumerate(
                [(0., 0.), (2., 0.), (4., 0.), (1., np.sqrt(3)), (3., np.sqrt(3)), (2., 2*np.sqrt(3))], 1)]


def analytic_slice(z):
    """Steiner formula for the convex hull of six equal, disjoint disks."""
    rho = np.sqrt(1 - (z - 2)**2)
    solid = 6 * np.pi * rho**2
    outer = 4 * np.sqrt(3) + 12 * rho + np.pi * rho**2
    return outer, solid, 1 - solid / outer


def run_benchmark(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    particles = six_spheres()
    pd.DataFrame([dict(id=p['id'], x=p['x'][0], y=p['x'][1], z=p['x'][2], radius=1., D_eq=2.,
                      x_over_Deq=p['x'][0]/2, y_over_Deq=p['x'][1]/2, z_over_Deq=p['x'][2]/2)
                  for p in particles]).to_csv(output / 'particles.csv', index=False)
    rows = []
    for resolution in (72, 720, 1080):
        # Exercise the production slice/profile routines; vary only contour sampling.
        with patch.object(porosity2d, 'ellipse_points', lambda *a, **kw: ellipse_points(*a, **dict(kw, n=resolution))):
            profile = compute_porosity_profile(particles, [1.25, 1.5, 2., 2.5, 2.75], alpha=0., lz=6.)
        for row in profile.to_dict('records'):
            outer, solid, phi = analytic_slice(row['z'])
            rows.append(dict(n_points=resolution, z_over_Deq=row['z']/2, **row, A_alpha_exact=outer,
                             A_ell_exact=solid, porosity_exact=phi,
                             porosity_error=row['porosity'] - phi))
    results = pd.DataFrame(rows)
    results.to_csv(output / 'comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 5))
    for p in particles:
        ax.add_patch(plt.Circle(p['x'][:2]/2, .5, color='steelblue', alpha=.45))
        ax.text(*(p['x'][:2]/2), str(p['id']), ha='center', va='center')
    result = porosity2d.compute_slice_porosity(particles, 2., 0., 6.)
    ax.plot(*(result['shape'].T/2), color='black', label='Routine: Hülle mit 720 Punkten/Kreis')
    ax.set(xlim=(-.6, 2.6), ylim=(-.6, 2.35), xlabel=r'$x / D_{\mathrm{eq}}$', ylabel=r'$y / D_{\mathrm{eq}}$',
           title='Sechs berührende Kugeln: Äquatorschnitt z/Deq = 1')
    ax.set_aspect('equal'); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), fontsize=10)
    plotstyle.prepare_figure(fig); fig.tight_layout(); plotstyle.export_figure(fig, output / 'arrangement.png', dpi=180); plt.close(fig)
    (output / 'README.md').write_text(
        '# Analytischer Test: sechs Kugeln\n\n'
        'D_eq=2R=2; normierte Mittelpunkte siehe particles.csv. Drei Kugeln unten, zwei darüber und eine oben '
        'im Dreiecksgitter (Abstand benachbarter Mittelpunkte D_eq). '
        'Dies ist eine dicht gepackte Lage, kein dreidimensionaler FCC/HCP-Verband.\n\n'
        'Die Routine misst Schnittflächenporosität relativ zur Alpha-Hülle. '
        'Mit alpha=0 ist diese die konvexe Hülle. Lz/D_eq=3 verhindert zusätzliche '
        'periodische Schnitte in den untersuchten Ebenen. Keine festen Partikel.\n\n'
        'Mit z_star=z/D_eq, z_c_star=1 und rho_star=sqrt(1/4-(z_star-1)^2): '
        'Die Hülle der Mittelpunkte ist ein gleichseitiges Dreieck mit Seitenlänge '
        '2 D_eq, Fläche sqrt(3) D_eq² und Umfang 6 D_eq. Steiner-Formel:\n\n'
        'A_outer/D_eq² = sqrt(3) + 6 rho_star + pi rho_star²; '
        'A_solid/D_eq² = 6 pi rho_star²; epsilon = 1 - A_solid/A_outer.\n\n'
        'comparison.csv enthält Sollwerte, Routinewerte und absolute Differenzen '
        'für 72, 720 (Standard der Porositätsroutine) und 1080 Konturpunkte. '
        'Die Vergleichsauflösungen werden nur im Benchmark temporär eingestellt.\n\n'
        'Anhang mit TikZ-Abbildung und Einbauanweisung: '
        '`bilder/generiert/anhang_porositaetsvalidierung/` im Workspace.\n\n'
        'Reproduktion vom Repository granular_analysis aus:\n\n'
        '```sh\nPYTHONPATH=src python -m '
        'granular_analysis.scripts.validate_porosity_six_spheres '
        '--out ../results_granular_analysis/validation_six_spheres\n```\n', encoding='utf-8')
    print(results.to_string(index=False))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('validation_six_spheres'))
    run_benchmark(parser.parse_args().out)
