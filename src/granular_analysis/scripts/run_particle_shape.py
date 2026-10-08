"""Export all input-particle axis ratios and a Zingg scatter plot."""
from pathlib import Path
from granular_analysis.analysis.particle_shape import read_particle_axes, compute_sieve_curve, din_sieve_tables
from granular_analysis.plotting.zingg import plot_zingg
from granular_analysis.plotting.sieve import plot_sieve_curve
from granular_analysis.config import SKIP_FIRST_PARTICLES, USE_PARTICLE_SKIP, PARTICLE_LENGTH_UNIT_MM


def run_particle_shape(*, case, out_dir, skip_first=None, length_unit_mm=None):
    """Analyze input rows after skipping the first selected particles."""
    if length_unit_mm is None:
        length_unit_mm = PARTICLE_LENGTH_UNIT_MM
    source = case.case_root / 'input_ellipsoid_data.dat'
    skip = (SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0) if skip_first is None else skip_first
    if not isinstance(skip, int) or skip < 0:
        raise ValueError("skip_first must be a nonnegative integer")
    table = read_particle_axes(source).iloc[skip:].copy()
    if table.empty:
        raise ValueError(f"No particles remain after skipping {skip} input particles")
    curve = compute_sieve_curve(table)
    din_tables = din_sieve_tables(curve, length_unit_mm=length_unit_mm) if length_unit_mm is not None else None
    import numpy as np
    table["d_eq"] = 2*np.exp(np.log(table[["a", "b", "c"]]).sum(axis=1)/3)
    if length_unit_mm is not None:
        table["d_eq_mm"] = table["d_eq"] * length_unit_mm
    destination = Path(out_dir) / 'particle_shape'
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / 'particle_shape.csv', index=False, encoding='utf-8')
    plot_zingg(table, destination / 'zingg.svg')
    plot_zingg(table, destination / 'zingg_wadell.svg', color_by='wadell_sphericity')
    curve.to_csv(destination / 'sieve_curve.csv', index=False, encoding='utf-8')
    plot_sieve_curve(curve, destination / 'sieve_curve.svg', length_unit_mm=length_unit_mm)
    if din_tables is not None:
        din_tables[0].to_csv(destination / 'sieve_curve_din.csv', index=False, encoding='utf-8')
        din_tables[1].to_csv(destination / 'grain_classes_din.csv', index=False, encoding='utf-8')
    else:
        print('DIN-Einteilung benötigt --length-unit-mm; Sieblinie in Input-Einheiten ausgegeben.')
    print(f'Physical scale: {length_unit_mm} mm per input length unit')
    print(f'Skipped {skip} input particles')
    print(f'Particle shape: {len(table)} input rows -> {destination}')
    return table
