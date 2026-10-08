"""Input-particle axis ratios for Zingg shape analysis."""
from pathlib import Path
import numpy as np
import pandas as pd


def compute_zingg_ratios(axes):
    """Return I/L and S/I after sorting positive principal semi-axes.

    Full axis lengths give the same ratios. The supplied a/b/c order is
    irrelevant; L >= I >= S is determined separately for each particle.
    """
    axes = np.asarray(axes, dtype=float)
    if axes.ndim != 2 or axes.shape[1] != 3 or len(axes) == 0:
        raise ValueError("Expected a nonempty (N, 3) array of semi-axes")
    if not np.isfinite(axes).all() or np.any(axes <= 0):
        raise ValueError("Particle semi-axes must be finite and positive")
    ordered = np.sort(axes, axis=1)
    short, intermediate, long = ordered.T
    return intermediate / long, short / intermediate


def compute_corey_shape_factor(axes):
    """Return Corey Sf = S / sqrt(L*I), with L >= I >= S.

    Equivalently c/sqrt(a*b) for sorted axes a >= b >= c. Computing
    from dimensionless ratios avoids overflow in the axis product.
    """
    intermediate_over_long, short_over_intermediate = compute_zingg_ratios(axes)
    return short_over_intermediate * np.sqrt(intermediate_over_long)


def compute_wadell_sphericity(axes):
    """Return equal-volume sphere area divided by ellipsoid surface area.

    Use Carlson's symmetric elliptic integral RG (DLMF 19.33.1),
    evaluated numerically by scipy.special.elliprg. Scaling the RG
    arguments by the shortest axis avoids squared dimensional axes.
    This describes the smooth ellipsoid, not its discretized mesh.
    """
    from scipy.special import elliprg

    compute_zingg_ratios(axes)  # shared shape/positive/finite validation
    short, intermediate, long = np.sort(np.asarray(axes, dtype=float), axis=1).T
    area_factor = elliprg((short / long)**2, (short / intermediate)**2, 1.0)
    log_numerator = (2*np.log(short) - np.log(long) - np.log(intermediate)) / 3
    return np.minimum(np.exp(log_numerator) / area_factor, 1.0)


def read_particle_axes(input_path):
    """Read every numeric input row, preserving physical row number and ID.

    Blank lines and nonnumeric headers are allowed. A row beginning with a
    numeric ID must supply valid density and three strictly positive axes.
    No Nell limit or fixed-particle exclusion is applied.
    """
    rows = []
    for line_number, line in enumerate(Path(input_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith(("#", "!")):
            continue
        fields = [part.strip() for part in line.split(",")]
        try:
            particle_id = float(fields[0].replace("D", "e").replace("d", "e"))
        except ValueError:
            continue
        if not np.isfinite(particle_id) or particle_id != int(particle_id):
            raise ValueError(f"Invalid particle ID on line {line_number}")
        try:
            values = [float(value.replace("D", "e").replace("d", "e")) for value in fields[1:5]]
            if len(values) != 4 or not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid density or semi-axes on line {line_number}") from None
        rows.append((line_number, int(particle_id), *values))
    if not rows:
        raise ValueError(f"No particle rows in {input_path}")
    table = pd.DataFrame(rows, columns=["input_line", "particle_id", "rho", "a", "b", "c"])
    ordered = np.sort(table[["a", "b", "c"]].to_numpy(), axis=1)
    table[["S", "I", "L"]] = ordered
    table["I_over_L"], table["S_over_I"] = compute_zingg_ratios(table[["a", "b", "c"]])
    table["Sf"] = compute_corey_shape_factor(table[["a", "b", "c"]])
    table["wadell_sphericity"] = compute_wadell_sphericity(table[["a", "b", "c"]])
    high_i = table["I_over_L"] >= 2 / 3
    high_s = table["S_over_I"] >= 2 / 3
    table["zingg_class"] = np.select(
        [high_i & high_s, high_i & ~high_s, ~high_i & high_s],
        ["equant", "disk", "rod"], default="blade",
    )
    return table


def compute_sieve_curve(table):
    """Cumulative fractions <= volume-equivalent diameter (input units).

    This is a geometric size distribution, not simulated sieve passage
    for nonspherical grains. Equal diameters are grouped into one step.
    """
    if table.empty:
        raise ValueError('No particles remain for the sieve curve')
    axes = table[['a', 'b', 'c']].to_numpy(dtype=float)
    compute_zingg_ratios(axes)
    density = table['rho'].to_numpy(dtype=float)
    if not np.isfinite(density).all() or np.any(density <= 0):
        raise ValueError('Particle density must be finite and positive')
    log_volume = np.log(axes).sum(axis=1)
    diameter = 2 * np.exp(log_volume / 3)
    # Normalize weights before summing to avoid dimensional overflow.
    volume = np.exp(log_volume - log_volume.max())
    log_mass = log_volume + np.log(density)
    mass = np.exp(log_mass - log_mass.max())
    bins = pd.DataFrame(dict(d_eq=diameter, count=np.ones(len(table), dtype=int),
                             volume_weight=volume, mass_weight=mass)).groupby('d_eq', sort=True).sum()
    result = bins[['count']].reset_index()
    for name, weights in [('number', bins['count']), ('volume', bins['volume_weight']), ('mass', bins['mass_weight'])]:
        result[f'passing_{name}_percent'] = weights.cumsum().to_numpy() / weights.sum() * 100
    return result


# Grain-size class boundaries in mm (DIN EN ISO 14688-1).
DIN_CLASS_LIMITS_MM = np.array([.002, .0063, .02, .063, .2, .63, 2, 6.3, 20, 63, 200, 630])
DIN_CLASS_NAMES = ('Ton', 'Feinschluff', 'Mittelschluff', 'Grobschluff',
                   'Feinsand', 'Mittelsand', 'Grobsand', 'Feinkies',
                   'Mittelkies', 'Grobkies', 'Steine', 'Blöcke', 'Große Blöcke')


def din_sieve_tables(curve, *, length_unit_mm):
    """Sample empirical passing percentages at DIN class boundaries.

    Class intervals are lower-exclusive, upper-inclusive; Ton <= .002 mm.
    The scale must be explicitly supplied: no dimensional assumption.
    """
    if not np.isfinite(length_unit_mm) or length_unit_mm <= 0:
        raise ValueError('length_unit_mm must be finite and positive')
    diameter = curve.d_eq.to_numpy() * length_unit_mm
    if not np.isfinite(diameter).all() or np.any(diameter <= 0):
        raise ValueError('Scaled diameters must be finite and positive')
    positions = np.searchsorted(diameter, DIN_CLASS_LIMITS_MM, side='right') - 1
    sampled = pd.DataFrame({'diameter_mm': DIN_CLASS_LIMITS_MM})
    fractions = pd.DataFrame({'grain_class': DIN_CLASS_NAMES,
                             'lower_mm': np.r_[0., DIN_CLASS_LIMITS_MM],
                             'upper_mm': np.r_[DIN_CLASS_LIMITS_MM, np.inf]})
    for name in ('number', 'volume', 'mass'):
        cumulative = curve[f'passing_{name}_percent'].to_numpy()
        passing = np.where(positions >= 0, cumulative[np.maximum(positions, 0)], 0.)
        sampled[f'passing_{name}_percent'] = passing
        fractions[f'fraction_{name}_percent'] = np.diff(np.r_[0., passing, 100.])
    return sampled, fractions
