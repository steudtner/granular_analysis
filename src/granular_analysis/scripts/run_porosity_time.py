"""Plot porosity profiles coloured by time, and mean +/- spatial standard deviation."""

from granular_analysis.plotting import plotstyle
from pathlib import Path
import argparse
import json
import re

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

from granular_analysis.analysis.porosity_time import (
    periodic_z_grid, select_time_samples, discover_frame_times,
    compute_porosity_profile, summarize_porosity,
)
from granular_analysis.config import ALPHA_2D, LZ, D_EQ, SKIP_FIRST_PARTICLES, USE_PARTICLE_SKIP
from granular_analysis.io.loader import load_frame, _read_bin_time
from granular_analysis.paths import CasePaths
from granular_analysis.plotting.helpers import PLOT_STYLE
from granular_analysis.utils.state_manager import build_fingerprint


def plot_porosity_time(profiles, summary, output, *, n_profiles=15, lz=LZ):
    if not isinstance(n_profiles, (int, np.integer)) or n_profiles < 1:
        raise ValueError('n_profiles must be a positive integer')
    chosen = select_time_samples(summary[['counter', 'time']].itertuples(index=False, name=None), n_profiles)
    times = summary['time'].to_numpy()
    lo, hi = float(times.min()), float(times.max())
    norm = colors.Normalize(lo, hi if hi > lo else lo + max(1., abs(lo)*.01))
    cmap = colors.LinearSegmentedColormap.from_list('time_viridis', plt.get_cmap('viridis')(np.linspace(0, .88, 256)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full, 4.6), constrained_layout=True)
        try:
            for counter, time in chosen:
                data = profiles.loc[profiles.counter == counter].sort_values('z')
                # Close the periodic profile visually, without counting Lz again in statistics.
                z = np.r_[data.z.to_numpy(), lz]
                phi = np.r_[data.porosity.to_numpy(), data.porosity.iloc[0]]
                ax.plot(z / D_EQ, phi, color=cmap(norm(time)), linewidth=1.2)
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=.025)
            bar.set_label(r'Zeit $t$ (Simulationseinheiten)')
            if lo == hi:
                bar.set_ticks([lo])
            ax.set(xlabel=r'$z / D_{\mathrm{eq}}$', ylabel=r'Porosität $\phi(z,t)$', xlim=(0,lz/D_EQ))
            ax.grid(alpha=.2)
            plotstyle.export_figure(fig, output / 'porosity_profiles.png', dpi=250)
        finally:
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(plotstyle.figwidth_full,4.6), constrained_layout=True)
        try:
            mean = summary.porosity_mean.to_numpy()
            std = summary.porosity_std.to_numpy()
            ax.fill_between(times, mean-std, mean+std, color=cmap(.45), alpha=.22,
                            label=r'$\overline{\phi}_z \pm \sigma_z$')
            ax.plot(times,mean,'o-',color=cmap(.35),markersize=3,linewidth=1.2,
                    label=r'Mittelwert $\overline{\phi}_z$')
            if len(times) == 1:
                ax.errorbar(times,mean,yerr=std,fmt='none',color=cmap(.35),capsize=4)
            ax.set(xlabel=r'Zeit $t$ (Simulationseinheiten)',ylabel=r'Porosität $\phi$')
            ax.legend();ax.grid(alpha=.2)
            plotstyle.export_figure(fig, output / 'porosity_mean_std.png', dpi=250)
        finally:
            plt.close(fig)
    return chosen


def run_porosity_time(case, output, *, n_frames=15, n_profiles=15, time_start=None,
                      time_end=None, z_step=.1, lz=LZ, alpha=ALPHA_2D,
                      skip_first=None, force=False):
    """Return (long-form profiles, spatial statistics); export plots and CSVs.

    Per-frame caches are keyed by source stat, path and geometry parameters.
    The existing slice model assumes a periodic z domain. No fluid files needed.
    """
    case = case if isinstance(case, CasePaths) else CasePaths.from_input(case)
    z_values = periodic_z_grid(lz, z_step)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    if not isinstance(n_profiles, (int, np.integer)) or n_profiles < 1:
        raise ValueError('n_profiles must be a positive integer')
    if skip_first is None:
        input_path = case.case_root / 'input.dat'
        text = input_path.read_text() if input_path.is_file() else ''
        match = re.search(r'(?im)^\s*Nell_fixed\s*=\s*(\d+)', text)
        skip_first = int(match.group(1)) if match else (SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0)
    if not isinstance(skip_first, (int, np.integer)) or skip_first < 0:
        raise ValueError('skip_first must be a nonnegative integer')
    # Validate range/sample arguments before scanning the real directory.
    select_time_samples([(0, 0.)], n_frames, time_start=None, time_end=None)
    if any(t is not None and not np.isfinite(t) for t in (time_start,time_end)) or (time_start is not None and time_end is not None and time_start > time_end):
        raise ValueError('Invalid time range')
    trn = case.require_trn()
    print(f'Reading particle timestamps: {trn}',flush=True)
    frames = select_time_samples(discover_frame_times(trn), n_frames, time_start, time_end)
    print(f'{len(frames)} times, {len(z_values)} periodic z planes, excluding {skip_first} fixed particles',flush=True)
    output = Path(output)
    cache_dir = output / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts, sources = [], []
    for i,(counter,time) in enumerate(frames,1):
        path = trn/f'ell_trn_{counter:06d}.bin'
        if _read_bin_time(path) != time:
            raise RuntimeError(f'Frame timestamp changed during selection: {path}; rerun')
        stat = path.stat()
        source = dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,counter=counter,time=time)
        fingerprint = build_fingerprint(version=1,source=source,z=z_values.tolist(),lz=lz,alpha=alpha,skip_first=skip_first)
        cached = cache_dir/f'{counter:06d}_{fingerprint}.csv'
        if cached.is_file() and not force:
            data = pd.read_csv(cached)
            if len(data) != len(z_values) or not np.allclose(data.z, z_values, rtol=0, atol=1e-12):
                raise ValueError(f'Invalid cached grid: {cached}; use --force')
            print(f'[{i}/{len(frames)}] t={time:.8g}, frame={counter} (cache)',flush=True)
        else:
            print(f'[{i}/{len(frames)}] t={time:.8g}, frame={counter}',flush=True)
            particles = load_frame(trn,path.name)
            data = compute_porosity_profile(particles,z_values,alpha=alpha,lz=lz,skip_first=skip_first)
            data = data.assign(counter=counter,time=time)
            after = path.stat()
            if (after.st_size,after.st_mtime_ns) != (stat.st_size,stat.st_mtime_ns):
                raise RuntimeError(f'Frame changed during analysis: {path}; rerun after writing completes')
            temporary=cached.with_suffix('.tmp')
            data.to_csv(temporary,index=False);temporary.replace(cached)
        parts.append(data);sources.append(source)
    profiles=pd.concat(parts,ignore_index=True)
    summary=summarize_porosity(profiles)
    profiles.to_csv(output/'porosity_profiles.csv',index=False)
    summary.to_csv(output/'porosity_time_summary.csv',index=False)
    chosen=plot_porosity_time(profiles,summary,output,n_profiles=n_profiles,lz=lz)
    manifest=dict(case=str(case.case_root),alpha=alpha,lz=lz,z_step_requested=z_step,
                  z_step_actual=float(lz/len(z_values)),z=z_values.tolist(),skip_first=skip_first,
                  requested_frames=n_frames,requested_profiles=n_profiles,time_start=time_start,time_end=time_end,
                  frames=sources,plotted_profiles=[dict(counter=c,time=t) for c,t in chosen],
                  definition='phi=1-A_ell_union/A_alpha, using existing periodic slice alpha-shape routine',
                  statistics='Equal-z population mean and std (ddof=0); only complete finite profiles within [0,1]. Lz endpoint excluded.',
                  time_units='Stored simulation time, no normalization',
                  cache_version=1)
    (output/'porosity_time.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved porosity evolution: {output}',flush=True)
    return profiles,summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--frames',type=int,default=15,help='Sampled times; 0 means all frames')
    parser.add_argument('--profiles',type=int,default=15,help='Maximum overlaid profiles')
    parser.add_argument('--time-start',type=float)
    parser.add_argument('--time-end',type=float)
    parser.add_argument('--z-step',type=float,default=.1)
    parser.add_argument('--lz',type=float,default=LZ)
    parser.add_argument('--alpha',type=float,default=ALPHA_2D)
    parser.add_argument('--skip-points',type=int)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    run_porosity_time(args.path,args.out,n_frames=args.frames,n_profiles=args.profiles,
                      time_start=args.time_start,time_end=args.time_end,z_step=args.z_step,
                      lz=args.lz,alpha=args.alpha,skip_first=args.skip_points,force=args.force)


if __name__=='__main__':
    main()
