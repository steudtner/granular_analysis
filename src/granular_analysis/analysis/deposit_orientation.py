"""Undirected particle axes relative to the laboratory frame, binned in x."""
import numpy as np
import pandas as pd
from granular_analysis.geometry.ellipsoid import _rotation_matrix


def deposit_orientation(particles, *, n_bins=6, angle_step=10.0, mode='auto',
                        xlim=None, ylim=None, skip_first=0):
    if mode not in ('auto', 'prolate', 'oblate'):
        raise ValueError('mode must be auto, prolate or oblate')
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    if not np.isfinite(angle_step) or not 0 < angle_step <= 90:
        raise ValueError('angle_step must be finite and in (0, 90]')
    if skip_first < 0:
        raise ValueError('skip_first must be nonnegative')
    for limits in (xlim, ylim):
        if limits is not None and (len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]):
            raise ValueError('Spatial limits must be finite and increasing')
    rows = []
    for number, p in enumerate(particles, 1):
        if number <= skip_first:
            continue
        position = np.asarray(p['x'], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f'Invalid position in particle row {number}')
        if ylim is not None and not ylim[0] <= position[1] <= ylim[1]:
            continue
        if xlim is not None and not xlim[0] <= position[0] <= xlim[1]:
            continue
        axes = np.array([p[k] for k in 'abc'], dtype=float)
        if not np.isfinite(axes).all() or np.any(axes <= 0):
            raise ValueError(f'Invalid semiaxes in particle row {number}')
        order = np.argsort(axes)
        short, middle, long = axes[order]
        lower_equal = np.isclose(short, middle, rtol=1e-6, atol=0)
        upper_equal = np.isclose(middle, long, rtol=1e-6, atol=0)
        shape = ('sphere' if lower_equal and upper_equal else
                 'prolate' if lower_equal else 'oblate' if upper_equal else 'triaxial')
        selected = shape if mode == 'auto' else mode
        row = dict(particle_row=number, x=position[0], y=position[1], z=position[2],
                   shape=shape, mode=selected, status='valid', axis='',
                   axis_x=np.nan, axis_y=np.nan, axis_z=np.nan,
                   angle_x_deg=np.nan, angle_y_deg=np.nan, angle_z_deg=np.nan)
        if selected not in ('prolate', 'oblate'):
            row['status'] = 'undefined_shape_axis'
        else:
            index = 0 if selected == 'prolate' else 2
            row['axis'] = 'abc'[index]
            if np.count_nonzero(np.isclose(axes, axes[index], rtol=1e-6, atol=0)) != 1:
                row['status'] = 'ambiguous_axis'
            elif 'R' not in p and 'q' not in p:
                row['status'] = 'missing_orientation'
            else:
                oriented = dict(p)
                if 'R' not in oriented:
                    q = np.array(p['q'], dtype=float, copy=True)
                    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
                        raise ValueError(f'Invalid quaternion in particle row {number}')
                    oriented['q'] = q
                rotation = _rotation_matrix(oriented)
                if (not np.isfinite(rotation).all() or
                    not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or
                    not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)):
                    raise ValueError(f'Invalid rotation in particle row {number}')
                direction = rotation[:, index]
                angles = np.degrees(np.arccos(np.clip(np.abs(direction), 0, 1)))
                row['axis'] = 'abc'[index]
                for i, name in enumerate('xyz'):
                    row[f'axis_{name}'] = direction[i]
                    row[f'angle_{name}_deg'] = angles[i]
        rows.append(row)
    if not rows:
        raise ValueError('No particles in the selected deposit region; check --skip-points and spatial limits')
    table = pd.DataFrame(rows)
    low, high = xlim if xlim is not None else (table.x.min(), table.x.max())
    if low == high:
        low, high = low - .5, high + .5
    edges = np.linspace(low, high, n_bins + 1)
    table['bin'] = np.clip(np.searchsorted(edges, table.x, side='right') - 1, 0, n_bins - 1)
    angle_edges = np.r_[np.arange(0, 90, angle_step), 90.0]
    records = []
    for selected in ('prolate', 'oblate'):
        for bin_index in range(n_bins):
            group = table[(table.bin == bin_index) & (table['mode'] == selected)]
            valid = group[group.status == 'valid']
            for axis in 'xyz':
                counts, _ = np.histogram(valid[f'angle_{axis}_deg'], angle_edges)
                for j, count in enumerate(counts):
                    records.append(dict(mode=selected, bin=bin_index,
                        x_min=edges[bin_index], x_max=edges[bin_index+1], axis=axis,
                        angle_min_deg=angle_edges[j], angle_max_deg=angle_edges[j+1],
                        count=int(count), valid_particles=len(valid), total_particles=len(group),
                        fraction=float(count / len(valid)) if len(valid) else np.nan))
    return table, pd.DataFrame(records), edges, angle_edges
