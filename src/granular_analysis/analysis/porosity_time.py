"""Time-resolved periodic slice porosity and spatial (not temporal) spread."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import pandas as pd

from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.io.loader import _read_bin_time


def periodic_z_grid(lz, z_step):
    """Uniform sampling of [0, Lz), with actual spacing no larger than z_step."""
    if not np.isfinite(lz) or not np.isfinite(z_step) or lz <= 0 or not 0 < z_step < lz:
        raise ValueError('Require finite Lz > z_step > 0')
    return np.linspace(0.0, lz, int(np.ceil(lz / z_step)), endpoint=False)


def select_time_samples(frames, n_frames=15, time_start=None, time_end=None):
    """Select distinct nearest frames to uniform physical-time targets.

    ``frames`` contains (counter, time) pairs. Equal timestamps keep the larger
    counter. n_frames=0 keeps all times; sparse data may give fewer samples.
    """
    if not isinstance(n_frames, (int, np.integer)) or n_frames < 0:
        raise ValueError('n_frames must be a nonnegative integer (0 = all)')
    if any(t is not None and not np.isfinite(t) for t in (time_start, time_end)):
        raise ValueError('Time limits must be finite')
    if time_start is not None and time_end is not None and time_start > time_end:
        raise ValueError('time_start must not exceed time_end')
    by_time = {}
    for counter, time in frames:
        if not np.isfinite(time):
            raise ValueError(f'Nonfinite time in frame {counter}')
        if (time_start is None or time >= time_start) and (time_end is None or time <= time_end):
            by_time[float(time)] = max(int(counter), by_time.get(float(time), -1))
    times = np.array(sorted(by_time))
    if len(times) == 0:
        raise ValueError('No particle frames in the requested time range')
    if n_frames == 0 or n_frames >= len(times):
        indices = np.arange(len(times))
    elif n_frames == 1:
        indices = np.array([len(times) - 1])
    else:
        targets = np.linspace(times[0], times[-1], n_frames)
        indices = np.unique([np.argmin(np.abs(times - target)) for target in targets])
    return [(by_time[float(times[i])], float(times[i])) for i in indices]


def discover_frame_times(folder):
    """Use binary headers as the source of truth, including after restarts."""
    files = sorted(Path(folder).glob('ell_trn_*.bin'))
    if not files:
        raise FileNotFoundError(f'No particle frames in {folder}')
    files = [path for path in files if path.stem.removeprefix('ell_trn_').isdigit()]
    def read_time(path):
        return int(path.stem.removeprefix('ell_trn_')), _read_bin_time(path)
    # Independent tiny header reads otherwise spend minutes waiting on network
    # storage. Bounded threads retain source-of-truth checks and sorted order.
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(read_time, files))


def compute_porosity_profile(particles, z_values, *, alpha, lz, skip_first=0):
    """Use the existing alpha-shape definition, excluding fixed particles.

    Conservative periodic bounding spheres omit particles that cannot
    intersect a plane. The retained particles use the exact existing cuts.
    """
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(lz) or lz <= 0:
        raise ValueError('Require finite alpha >= 0 and Lz > 0')
    if not isinstance(skip_first, (int, np.integer)) or skip_first < 0:
        raise ValueError('skip_first must be a nonnegative integer')
    z_values = np.asarray(z_values, dtype=float)
    if z_values.ndim != 1 or not len(z_values) or not np.isfinite(z_values).all():
        raise ValueError('z_values must be a nonempty finite vector')
    mobile = []
    for index, particle in enumerate(particles, 1):
        if int(particle.get('id', particle.get('idx', index))) <= skip_first:
            continue
        axes = np.array([particle['a'], particle['b'], particle['c']], dtype=float)
        if not np.isfinite(axes).all() or np.any(axes <= 0):
            continue  # Match the existing slice routine's invalid-axis rule.
        center = np.asarray(particle['x'], dtype=float).copy()
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f'Invalid particle position at index {index}')
        center[2] %= lz
        mobile.append(dict(particle, x=center, id=index))
    centers = np.array([p['x'][2] for p in mobile])
    radii = np.array([max(p['a'], p['b'], p['c']) for p in mobile])
    rows = []
    for z in z_values:
        distance = np.abs((centers - z + lz/2) % lz - lz/2)
        candidates = [mobile[i] for i in np.flatnonzero(distance <= radii)]
        result = compute_slice_porosity(candidates, float(z), alpha, lz, skip_first=0)
        area = float(result['A_alpha']) if result is not None else np.nan
        occupied = float(result['A_ell']) if result is not None else np.nan
        phi = float(result['porosity']) if result is not None else np.nan
        valid = bool(np.isfinite(phi) and np.isfinite(area) and np.isfinite(occupied)
                     and area > 0 and occupied >= 0 and 0 <= phi <= 1)
        rows.append(dict(z=float(z), porosity=phi if valid else np.nan,
                         A_alpha=area, A_ell=occupied, valid=valid))
    return pd.DataFrame(rows)


def summarize_porosity(profiles):
    """Equal-z population mean/std (ddof=0), requiring full grid coverage.

    An undefined or nonphysical slice stays missing, never zero or clipped.
    Incomplete profiles produce gaps in the mean/std time series.
    """
    records = []
    for (counter, time), group in profiles.groupby(['counter', 'time'], sort=True):
        values = group['porosity'].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values >= 0) & (values <= 1)
        complete = bool(valid.all())
        records.append(dict(counter=int(counter), time=float(time),
                            porosity_mean=float(np.mean(values)) if complete else np.nan,
                            porosity_std=float(np.std(values, ddof=0)) if complete else np.nan,
                            n_z=len(values), n_valid=int(valid.sum()), complete=complete))
    summary = pd.DataFrame(records).sort_values('time').reset_index(drop=True)
    if not summary['complete'].all():
        warnings.warn('Some profiles have invalid slices: mean/std remain NaN for those times; see n_valid in CSV.', RuntimeWarning)
    return summary
