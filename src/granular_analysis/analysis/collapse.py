"""Reusable P0–P2 analysis of completed, particle-resolved collapse histories.

Simulation inputs are read-only. Time is t*u_s/D, energy E/(rho_p*g*D**4).
Missing work/fluid terms never become purported dissipation measurements.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.io import FortranFile

from granular_analysis.paths import CasePaths
from granular_analysis.io.loader import _read_bin_time
from granular_analysis.io.monitor_loader import monitor_source_paths, _stitch_restart_segments
from granular_analysis.utils.physics import load_ellipsoid_properties, read_gravity, read_fluid_density
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.analysis.slope import fit_slope_model
from granular_analysis.geometry.ellipsoid import _rotation_matrix

# Auftrag 01.10.2026, §1: Wu-Wang estimates, never fitted to these cases.
PI_S = {"Kugel": 1.0, "Prolat": 0.842, "Oblat": 0.634}
PI_SET = {"Kugel": 0.866, "Prolat": 0.729, "Oblat": 0.549}
PARTICLE_DTYPE = np.dtype(dict(
    names=["x", "u", "q", "a", "b", "c"],
    formats=[("<f8", 3), ("<f8", 3), ("<f8", 4), "<f8", "<f8", "<f8"],
    offsets=[4, 36, 68, 108, 124, 140], itemsize=164))


@dataclass(frozen=True)
class CollapseOptions:
    d_eq: float = 1.0
    u_s: float = 0.3
    rho_p: float = 2500.0
    gravity: float = 0.06
    l0: float = 30.0
    h0: float = 15.0
    x_origin: float = 0.0
    y_origin: float = 0.0
    alpha: float = 0.6
    frames: int = 81
    z_slices: int = 8
    rotation_share_threshold: float = 0.5
    energy_fraction: float = 0.01
    hold_time: float = 10.0
    displacement_threshold: float = 0.1
    workers: int = 8

    def __post_init__(self):
        if not 0.5 <= self.rotation_share_threshold <= 1:
            raise ValueError("rotation_share_threshold must be between 0.5 and 1")
        if not 0 < self.energy_fraction < 1 or self.hold_time <= 0:
            raise ValueError("Invalid quasistatic energy fraction or hold time")
        if min(self.d_eq, self.u_s, self.rho_p, self.gravity, self.l0, self.h0, self.alpha) <= 0:
            raise ValueError("Physical scales and reference geometry must be positive")
        if (self.frames != 0 and self.frames < 4) or self.z_slices < 4 or self.z_slices % 2 or self.workers < 1:
            raise ValueError("Need >=4 frames (or 0 for all), an even number of >=4 z slices and >=1 worker")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_numbers(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"\s*(\w+)\s*=\s*([^,!]+)", line)
        if match:
            try:
                values[match[1].lower()] = float(match[2].lower().replace("d", "e"))
            except ValueError:
                pass
    return values


def read_frame(path):
    """Vectorized native PRIME TRN reader, checking every Fortran record marker."""
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw, '<i4', count=1, offset=4)[0])
    t = float(np.frombuffer(raw, '<f8', count=1, offset=16)[0])
    if n < 1 or len(raw) != 28 + n * 164:
        raise ValueError(f"Invalid TRN size: {path}")
    if tuple(np.frombuffer(raw, '<i4', count=3)) != (4, n, 4):
        raise ValueError(f"Invalid TRN particle-count record: {path}")
    if any(int(np.frombuffer(raw, '<i4', count=1, offset=j)[0]) != 8 for j in (12, 24)):
        raise ValueError(f"Invalid TRN time record: {path}")
    for start, end, size in ((0,28,24),(32,60,24),(64,100,32),(104,116,8),
                             (120,132,8),(136,148,8),(152,160,4)):
        for offset in (start, end):
            markers = np.ndarray((n,), '<i4', buffer=raw, offset=28+offset, strides=(164,))
            if np.any(markers != size):
                raise ValueError(f"Invalid particle record markers: {path}")
    data = np.frombuffer(raw, PARTICLE_DTYPE, count=n, offset=28)
    if not np.isfinite(t) or any(not np.isfinite(data[k]).all() for k in PARTICLE_DTYPE.names):
        raise ValueError(f"Nonfinite TRN data: {path}")
    return t, data


def as_particles(data, time):
    return [dict(x=r['x'].copy(), u=r['u'].copy(), q=r['q'].copy(),
                 a=float(r['a']), b=float(r['b']), c=float(r['c']), t=time)
            for r in data]


def snapshot_monitor(path):
    """Read and hash exactly the bytes analyzed, including immutable prefixes."""
    segments, sources = [], {}
    for source in monitor_source_paths(path):
        raw = source.read_bytes()
        sources[str(source.resolve())] = hashlib.sha256(raw).hexdigest()
        a = np.loadtxt(io.BytesIO(raw), ndmin=2)
        if a.shape[1] != 11 or not np.isfinite(a).all():
            raise ValueError(f"Invalid monitor: {source}")
        segments.append(a)
    joined = np.concatenate(segments)
    cleaned = _stitch_restart_segments(joined)
    if np.any(np.diff(cleaned[:, 0]) <= 0):
        raise ValueError(f"Interleaved or repeated monitor steps need explicit reconstruction: {path}")
    return cleaned, sources, len(joined) - len(cleaned)


def _energy_particle(args):
    number, path, mass, inertia, times = args
    data, sources, discarded = snapshot_monitor(path)
    translation = .5 * mass * np.sum(data[:, 5:8] ** 2, axis=1)
    rotation = .5 * np.sum(inertia * data[:, 8:11] ** 2, axis=1)
    tr = np.interp(times, data[:, 1], translation, left=np.nan, right=np.nan)
    rot = np.interp(times, data[:, 1], rotation, left=np.nan, right=np.nan)
    return number, tr, rot, sources, discarded


def energy_series(case, times, skip, options):
    """Sum measured energies on requested native times; keep excluded contributions."""
    _, masses, a, b, c = load_ellipsoid_properties(case.case_root / 'input_ellipsoid_data.dat')
    inertia = masses[:, None] * np.column_stack((b*b+c*c, a*a+c*c, a*a+b*b)) / 5
    files = sorted(case.ellipsoid.glob('ell_*_monitor.dat'))
    if (case.ellipsoid / 'ellipsoid_monitor').is_dir():
        files = sorted((case.ellipsoid / 'ellipsoid_monitor').glob('ell_*_monitor.dat'))
    numbers = [int(f.name.split('_')[1]) for f in files]
    expected = set(range(skip+1, int(input_numbers(case.case_root/'input.dat')['nell'])+1))
    chosen = [(number, path) for number, path in zip(numbers, files) if number > skip]
    if set(n for n, _ in chosen) != expected:
        raise ValueError('Missing or unexpected mobile monitor files')
    args = [(n, p, masses[n-1], inertia[n-1], times) for n, p in chosen]
    trans = np.zeros(len(times)); raw = np.zeros(len(times)); largest = np.zeros(len(times))
    owner = np.zeros(len(times), dtype=int); sources = {}; removed_rows = 0
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        for index, (n, tr, rot, provenance, discarded) in enumerate(pool.map(_energy_particle, args, chunksize=4)):
            trans += tr; raw += rot
            bigger = rot > largest
            largest[bigger] = rot[bigger]; owner[bigger] = n
            sources.update(provenance); removed_rows += discarded
            if (index + 1) % 300 == 0:
                logging.info('Energy: %d/%d mobile monitors', index+1, len(chosen))
    exclude = largest > options.rotation_share_threshold * raw
    removed = np.where(exclude, largest, 0.0)
    rot = raw - removed
    norm = options.rho_p * options.gravity * options.d_eq**4
    frame = pd.DataFrame(dict(time=times, t=times*options.u_s/options.d_eq,
        E_kin_trans=trans/norm, E_kin_rot=rot/norm, E_kin_tot=(trans+rot)/norm,
        E_kin_rot_raw=raw/norm, E_kin_rot_excluded=removed/norm,
        rotation_excluded_monitor=np.where(exclude, owner, 0),
        rotation_share_threshold=options.rotation_share_threshold))
    return frame, sources, removed_rows


def quasistatic_time(t, energy, fraction=.01, hold=10.):
    """First continuous hold below threshold after the last observed exceedance.

    Missing samples break a candidate hold. Later unmeasured samples are flagged
    separately, and are not asserted to prove ongoing quiescence.
    """
    t, energy = np.asarray(t), np.asarray(energy)
    finite = np.isfinite(energy)
    if len(t) < 4 or np.any(np.diff(t) <= 0) or not np.isfinite(t).all():
        raise ValueError('Need at least four strictly increasing finite times')
    if not finite.any() or np.any(energy[finite] < 0) or np.nanmax(energy) <= 0:
        return np.nan
    peak = int(np.nanargmax(energy))
    above = np.flatnonzero(energy[peak:] > fraction*energy[peak])
    start = peak+above[-1]+1 if len(above) else peak
    candidate=None
    for index in range(start,len(t)):
        if not finite[index]:
            candidate=None
            continue
        if candidate is None:candidate=index
        if index-candidate>=2 and t[index]-t[candidate]>=hold:
            return float(t[candidate])
    return np.nan


def t95(t, front):
    """First reaching 95% of final observed advance, with an explicit positive advance."""
    t, front = np.asarray(t), np.asarray(front)
    if len(t) < 2 or np.any(np.diff(t) <= 0) or not np.isfinite(front).all():
        return np.nan
    advance = front[-1]-front[0]
    if advance <= 0:
        return np.nan
    hits = np.flatnonzero(front >= front[0]+.95*advance)
    return float(t[hits[0]])


def upper_edge(coords, n_bins=100, max_distance=2.):
    coords = np.asarray(coords)
    edges = np.linspace(coords[:, 0].min(), coords[:, 0].max(), n_bins+1)
    bins = np.clip(np.searchsorted(edges, coords[:, 0], side='right')-1, 0, n_bins-1)
    raw = [coords[np.flatnonzero(bins == i)[np.argmax(coords[bins == i, 1])]]
           for i in range(n_bins) if np.any(bins == i)]
    if not raw:
        raise ValueError('No upper-edge points')
    accepted = [raw[0]]
    for point in raw[1:]:
        if np.linalg.norm(point-accepted[-1]) <= max_distance:
            accepted.append(point)
    return np.asarray(accepted)


def regress_surface(coords):
    edge = upper_edge(coords)
    models, rows = [], []
    for sections in range(1, 5):
        k = 3 + 2*(sections-1)
        if len(edge) <= k+1:
            continue
        model = fit_slope_model(edge[:, 0], edge[:, 1], sections-1)
        aicc = model.aic + 2*k*(k+1)/(len(edge)-k-1)
        rows.append(dict(n_sections=sections, n_parameters=k, n_points=len(edge),
            AIC=model.aic, BIC=model.bic, AIC_c=aicc, RSS=model.rss,
            breakpoints=json.dumps(model.breakpoints.tolist()),
            angles=json.dumps(model.angles_deg.tolist())))
        models.append(model)
    if not rows:
        raise ValueError('Insufficient upper-edge points for regression')
    scores = pd.DataFrame(rows)
    criterion = 'AIC_c' if any(len(edge)/r['n_parameters'] < 40 for r in rows) else 'BIC'
    index = int(scores[criterion].idxmin())
    scores['selected'] = scores.index == index
    scores['selection_criterion'] = criterion
    return models[index], scores, pd.DataFrame(edge, columns=['x', 'y'])


def _geometry_job(args):
    path, skip, lz, options = args
    time, all_data = read_frame(path)
    data = all_data[skip:]
    particles = as_particles(data, time)
    rows, center = [], None
    for z in np.arange(options.z_slices)*lz/options.z_slices:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = compute_slice_porosity(particles, float(z), options.alpha, lz,
                                            densify_max_segment_length=.1, clip_solid_to_shape=True)
        fallback = any('convex hull' in str(w.message) for w in caught)
        if result is None or result['shape'] is None:
            rows.append(dict(z=z, x_front=np.nan, L=np.nan, H=np.nan, A=np.nan, phi=np.nan, fallback=True))
            continue
        coords = result['shape']
        front = float(coords[:,0].max())
        rows.append(dict(z=z, x_front=front, L=front-options.x_origin,
            span=float(np.ptp(coords[:,0])), H=float(coords[:,1].max()-options.y_origin),
            A=result['A_alpha'], A_solid=result['A_ell'], phi=result['porosity'], fallback=fallback))
        if np.isclose(z, lz/2): center=coords
    profiles = pd.DataFrame(rows)
    profiles.insert(0, 't', time*options.u_s/options.d_eq)
    profiles.insert(1, 'counter', int(Path(path).stem.split('_')[-1]))
    return str(path), profiles, center, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_frames(times, n_frames):
    """Denser early sampling plus a late tail; always retain first and final frames."""
    if n_frames == 0 or len(times) <= n_frames:
        return np.arange(len(times))
    split = min(times[0]+(times[-1]-times[0])/3, times[-1])
    early = np.linspace(times[0], split, n_frames//2, endpoint=False)
    late = np.linspace(split, times[-1], n_frames-len(early))
    targets = np.r_[early, late]
    return np.unique([int(np.argmin(abs(times-t))) for t in targets])


def orientation_statistics(data, form):
    rows=[]
    for index, p in enumerate(as_particles(data, 0)):
        axes=np.array([p['a'],p['b'],p['c']])
        if form == 'Kugel':
            vector=np.full(3,np.nan)
        else:
            axis=int(np.argmax(axes) if form == 'Prolat' else np.argmin(axes))
            vector=_rotation_matrix(p)[:,axis]
        angle=lambda j: float(np.degrees(np.arccos(np.clip(abs(vector[j]),0,1))))
        rows.append(dict(mobile_index=index, dx=vector[0], dy=vector[1], dz=vector[2],
            angle_to_flow=angle(0), angle_to_vertical=angle(1),
            axis_kind='major_axis' if form == 'Prolat' else 'minor_axis_normal' if form == 'Oblat' else 'undefined'))
    return pd.DataFrame(rows)


def contact_statistics(path, skip, n_mobile):
    """Unique RK3 nonzero collision-impulse pairs; retain contacts with fixed bed."""
    active=set()
    with FortranFile(path) as f:
        n=int(f.read_ints('i4')[0]); time=float(f.read_reals('f4')[0]); f.read_reals('f4')
        for _ in range(n):
            first=int(f.read_ints('i4')[0]); second=int(f.read_ints('i4')[0])
            for _ in range(6): f.read_reals('f4')
            impulses=[f.read_reals('f4') for _ in range(3)]
            if first != second and any(np.any(v != 0) for v in impulses):
                active.add(tuple(sorted((first,second))))
    degrees=np.zeros(n_mobile,dtype=int)
    for pair in active:
        for number in pair:
            if skip < number <= skip+n_mobile: degrees[number-skip-1]+=1
    return time, pd.DataFrame(dict(particle_row=np.arange(skip+1,skip+n_mobile+1),contact_count=degrees)), active


def grid_comparison(coarse, fine, metrics=('l','h','theta','phi_end','t_collapse')):
    """Two-grid differences only; no inferred convergence order."""
    if 'form' in coarse and 'form' in fine and coarse['form'] != fine['form']:
        raise ValueError('Grid comparison requires the same particle form')
    if 'chi' in coarse and 'chi' in fine and not np.isclose(coarse['chi'],fine['chi']):
        raise ValueError('Grid comparison requires the same initial aspect ratio')
    rows=[]
    for metric in metrics:
        a,b=float(coarse[metric]),float(fine[metric])
        rows.append(dict(metric=metric, coarse=a, fine=b,
            relative_difference=(b-a)/abs(a) if np.isfinite(a) and a != 0 else np.nan))
    return pd.DataFrame(rows)


def dissipation_from_complete_budget(initial_energy, mechanical_energy, boundary_work):
    """Cumulative dissipation for a caller-supplied complete system energy/work."""
    e,w=np.broadcast_arrays(np.asarray(mechanical_energy,float),np.asarray(boundary_work,float))
    if not np.isfinite(initial_energy) or not np.isfinite(e).all() or not np.isfinite(w).all():
        raise ValueError('Complete finite system energy and boundary work are required')
    return initial_energy+w-e


def dry_scaling(chi):
    """Dimensionless proportional shapes from Auftrag P1.4, amplitudes unspecified."""
    chi=np.asarray(chi,float)
    if np.any(chi <= 0): raise ValueError('chi must be positive')
    # Continuous unit-amplitude illustrations, not asserted literature amplitudes.
    return pd.DataFrame(dict(chi=chi,
        planar_delta_l_shape=np.where(chi<=3,chi,3*(chi/3)**(2/3)),
        wide_delta_l_shape=chi**.9, narrow_delta_l_shape=chi**.65,
        planar_h_shape=chi**(-2/3), wide_h_shape=chi**(-.6)))


def initial_settling_indicator(time, normalized_vy, form, window=5.0):
    """Time-mean initial top-layer sinking versus the supplied single-grain estimate.

    This includes startup acceleration and is an indicator, not a measured
    hindered-settling coefficient. No extrapolation outside initial coverage.
    """
    t,v=np.asarray(time,float),np.asarray(normalized_vy,float)
    if len(t)<2 or t.shape!=v.shape or np.any(np.diff(t)<=0) or window<=0:
        raise ValueError('Need increasing time, matching velocity samples and positive window')
    mean=np.nan
    if t[0]<=0 and t[-1]>=window and np.isfinite(v).all():
        sample=np.r_[0.,t[(t>0)&(t<window)],window]
        values=-np.interp(sample,t,v)
        mean=float(np.sum(.5*(values[:-1]+values[1:])*np.diff(sample))/window)
    return dict(form=form,t_lower=0.,t_upper=window,mean_initial_sink_over_us=mean,
        Pi_S=PI_S[form],Pi_set_Wu_Wang=PI_SET[form],
        ratio_to_single_particle_estimate=mean/PI_SET[form],
        definition='Initial-window time mean; includes acceleration, not a pure hindrance coefficient')


def _frame_header_record(path):
    with Path(path).open('rb') as stream:
        header=stream.read(28)
    if len(header)!=28:raise ValueError(f'Incomplete TRN header: {path}')
    time=float(np.frombuffer(header,'<f8',count=1,offset=16)[0])
    stat=Path(path).stat()
    return dict(counter=int(Path(path).stem.split('_')[-1]),time=time,path=str(path),
        file_size=stat.st_size,mtime_ns=stat.st_mtime_ns,header_sha256=hashlib.sha256(header).hexdigest())


def inventory_frames(case, workers=8, manifest_path=None):
    paths=sorted(case.require_trn().glob('ell_trn_*.bin'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records=list(pool.map(_frame_header_record,paths))
    times=np.asarray([r['time'] for r in records])
    if manifest_path is not None:
        pd.DataFrame(records).to_csv(manifest_path,index=False)
    counters=np.array([int(p.stem.split('_')[-1]) for p in paths])
    if np.any(np.diff(times) <= 0):
        raise ValueError('TRN time resets/duplicates require reconstruction before collapse analysis')
    if np.any(np.diff(counters) != 1):
        logging.warning('TRN sequence has missing counters; native sample times are preserved')
    return paths,times


def prepare_energy(case, folder, options):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    paths,times=inventory_frames(case,options.workers,folder/'frame_inventory.csv')
    p=input_numbers(case.case_root/'input.dat')
    skip=int(p['nell_fixed'])
    if not np.isclose(read_gravity(case.case_root/'input.dat'),options.gravity):
        raise ValueError('Configured gravity differs from actual case input')
    rho_f=read_fluid_density(case.case_root/'input.dat')
    if not np.isclose(np.sqrt((options.rho_p/rho_f-1)*options.gravity*options.d_eq),options.u_s):
        raise ValueError('Velocity scale inconsistent with actual densities/gravity')
    energy,sources,discarded=energy_series(case,times,skip,options)
    # Explicit initial condition only: the solver initializes om=0 and u=0.
    if times[0] == 0:
        _,first=read_frame(paths[0])
        initialization=case.case_root/'routines/ellipsoid/ellipsoid_initialize.F90'
        code=initialization.read_text() if initialization.exists() else ''
        if np.all(first['u'][skip:] == 0) and re.search(r'%om\s*=\s*0',code,re.I):
            for name in ('E_kin_trans','E_kin_rot','E_kin_tot','E_kin_rot_raw','E_kin_rot_excluded'):
                energy.loc[0,name]=0.
            sources[str(initialization.resolve())]=sha256(initialization)
    energy.to_csv(folder/'energy.csv',index=False)
    provenance=dict(sha256=sources,discarded_restart_monitor_rows=discarded,
        options=options.__dict__,monitored_particles=int(p['nell'])-skip,
        sources=[str(case.results)],initial_energy_policy='zero only if TRN u=0 at t=0 and solver initializes om=0')
    (folder/'energy_herkunft.json').write_text(json.dumps(provenance,indent=2)+'\n')
    logging.info('Energy written: %s, %d native frames, %d restart monitor rows replaced',folder,len(times),discarded)
    return energy,paths,times


def _porosity_bins_job(args):
    data,z,lz,alpha=args
    from shapely.geometry import Polygon,box
    from shapely.ops import unary_union
    from granular_analysis.geometry.ellipsoid import ellipsoid_slice,ellipse_points
    particles=as_particles(data,0)
    result=compute_slice_porosity(particles,z,alpha,lz,clip_solid_to_shape=True)
    if result is None or result['shape'] is None: raise ValueError('Missing end slice')
    outer=Polygon(result['shape']); solids=[]
    for shift in (-lz,0,lz):
        for p in particles:
            cut=ellipsoid_slice(p,z,shift)
            if cut is not None: solids.append(Polygon(ellipse_points(*cut[:4],theta=cut[4])))
    solid=unary_union(solids).intersection(outer)
    x0,y0,x1,y1=outer.bounds
    xs=np.linspace(x0,x1,51); ys=np.arange(np.floor(y0),np.ceil(y1)+1,1.)
    if len(ys)<2: ys=np.array([y0,y1])
    rows=[]
    for xa,xb in zip(xs[:-1],xs[1:]):
        for ya,yb in zip(ys[:-1],ys[1:]):
            cell=box(xa,ya,xb,yb); area=outer.intersection(cell).area
            if area<=0:continue
            occupied=solid.intersection(cell).area
            rows.append(dict(z=z,x_lower=xa,x_upper=xb,y_lower=ya,y_upper=yb,
                             A_control=area,A_solid=occupied,phi=1-occupied/area))
    return pd.DataFrame(rows)


def pressure_reference_statistics(x,y,values,l0,reference=(90.,95.),vertical=(.5,3.)):
    """Gauge-invariant same-height pressure difference relative to remote fluid."""
    x,y,values=np.asarray(x),np.asarray(y),np.asarray(values)
    ambient=(x>=reference[0])&(x<=reference[1]); column=(x>=0)&(x<=l0)
    height=(y>=vertical[0])&(y<=vertical[1])
    if not ambient.any() or not column.any() or not height.any():
        raise ValueError('Pressure reference/control regions outside exported field')
    reference_profile=np.mean(values[:,ambient],axis=1)
    excess=values-reference_profile[:,None]
    selected=excess[np.ix_(height,column)]
    return dict(excess_pressure_mean=float(np.mean(selected)),
                excess_pressure_min=float(np.min(selected)),excess_pressure_max=float(np.max(selected)))


def pressure_series(case,paths,times,lz,options):
    from granular_analysis.io.fluid_slice import restart_records,load_fluid_slice
    if not (case.results/'rst_info.dat').is_file():
        return pd.DataFrame(),{}
    available=[]
    for record in restart_records(case.results):
        t=float(record['t']); counter=int(record['counter'])
        index=int(np.argmin(abs(times-t)))
        field=case.results/'fluid'/f'pre_{counter:06d}.bin'
        if abs(times[index]-t)<=1e-9 and field.exists():
            available.append((index,counter,field))
    if not available: return pd.DataFrame(),{}
    chosen=np.unique(np.linspace(0,len(available)-1,min(9,len(available)),dtype=int))
    rows=[];sources={}
    for j in chosen:
        index,counter,field=available[j]
        particle_counter=int(paths[index].stem.split('_')[-1])
        sample=load_fluid_slice(case,particle_counter,lz/2,'pressure')
        stats=pressure_reference_statistics(sample['x'],sample['y'],sample['values'],options.l0)
        rows.append(dict(t=times[index]*options.u_s/options.d_eq,counter=counter,
            z=lz/2,reference_x_lower=90.,reference_x_upper=95.,y_lower=.5,y_upper=3.,**stats))
        sources[str(field.resolve())]=sha256(field)
        logging.info('Pressure checkpoint %d, t*=%.3f',counter,rows[-1]['t'])
    for path in [case.results/'rst_info.dat',case.results/'history/rst_info.dat',
                 *(case.results/'fluid'/f'{axis}.dat' for axis in 'xyz')]:
        if path.exists():sources[str(path.resolve())]=sha256(path)
    return pd.DataFrame(rows),sources


def analyze_case(form,source,folder,options=CollapseOptions(),reuse_energy=False):
    if form not in PI_S: raise ValueError('Form must be Kugel, Prolat or Oblat')
    case=CasePaths.from_input(source);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    config=input_numbers(case.case_root/'input.dat')
    skip=int(config['nell_fixed']);lz=config['lenz'];issues=[]
    paths,times=inventory_frames(case,options.workers,folder/'frame_inventory.csv')
    if reuse_energy and (folder/'energy.csv').exists():
        metadata=json.loads((folder/'energy_herkunft.json').read_text())
        if metadata['options'] != options.__dict__:raise ValueError('Energy cache options differ')
        energy=pd.read_csv(folder/'energy.csv',float_precision='round_trip')
        if len(energy)!=len(times) or not np.allclose(energy.time,times,rtol=0,atol=1e-10):
            raise ValueError('Energy cache no longer matches TRN times')
    else:
        energy,paths,times=prepare_energy(case,folder,options)
        metadata=json.loads((folder/'energy_herkunft.json').read_text())
    sources=metadata['sha256'].copy()
    for name in ('input.dat','input_ellipsoid_data.dat','main.def'):
        sources[str((case.case_root/name).resolve())]=sha256(case.case_root/name)
    quality=case.results/'frame_quality_summary.csv'
    if quality.exists():
        q=pd.read_csv(quality).set_index('counter')
        counters=[int(p.stem.split('_')[-1]) for p in paths]
        strict=q.reindex(counters).strict_monitor_energy_available.fillna(False).to_numpy(bool)
        energy['strict_monitor_frame_match']=strict
        if not strict.all():
            issues.append(f'{int((~strict).sum())} TRN frames disagree with mobile monitors; energy at these frames is unavailable')
            energy.loc[~strict,['E_kin_trans','E_kin_rot','E_kin_tot','E_kin_rot_raw','E_kin_rot_excluded']]=np.nan
        sources[str(quality.resolve())]=sha256(quality)
    else: energy['strict_monitor_frame_match']=np.nan
    energy['monitor_frame_match_audited']=quality.exists()
    energy.to_csv(folder/'energy.csv',index=False)
    qs=quasistatic_time(energy.t,energy.E_kin_tot,options.energy_fraction,options.hold_time)
    raw_qs=quasistatic_time(energy.t,energy.E_kin_trans+energy.E_kin_rot_raw,options.energy_fraction,options.hold_time)
    indices=choose_frames(times,options.frames)
    if np.isfinite(qs):indices=np.unique(np.r_[indices,np.searchsorted(times,qs*options.d_eq/options.u_s)])
    initial_time,initial_all=read_frame(paths[0]);initial=initial_all[skip:]
    _,masses,aa,bb,cc=load_ellipsoid_properties(case.case_root/'input_ellipsoid_data.dat')
    mass=masses[skip:skip+len(initial)]
    if not np.isclose(times[0],0):issues.append('Initial TRN history absent; initial/top-layer metrics incomplete')
    initial_y=initial['x'][:,1]
    top=initial_y>=np.quantile(initial_y,.9)
    records=[];profiles=[];centers={};top_rows=[]
    args=[(paths[i],skip,lz,options) for i in indices]
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        for filename,profile,center,digest in pool.map(_geometry_job,args,chunksize=1):
            path=Path(filename);counter=int(path.stem.split('_')[-1]);tt,data=read_frame(path);mobile=data[skip:]
            central=profile.iloc[np.argmin(abs(profile.z-lz/2))]
            complete=np.isfinite(profile[['L','H','A','phi']]).all().all() and not profile.fallback.any()
            phi=1-profile.A_solid.sum()/profile.A.sum() if complete else np.nan
            record=dict(t=tt*options.u_s/options.d_eq,time=tt,counter=counter,
                x_front=central.x_front/options.d_eq,L_t=central.L/options.d_eq,
                H_t=central.H/options.d_eq,A_t=central.A/options.d_eq**2,phi_t=phi,
                slice_phi_mean=float(profile.phi.mean()),alpha=options.alpha,
                complete=bool(complete),n_z=len(profile),
                E_pot=float(np.sum(mass*(mobile['x'][:,1]-options.y_origin))*options.gravity/
                            (options.rho_p*options.gravity*options.d_eq**4)))
            records.append(record);profiles.append(profile);centers[counter]=center
            top_rows.append(dict(t=record['t'],mean_vy_over_us=float(mobile['u'][top,1].mean()/options.u_s),
                                 n_top=int(top.sum()),Pi_S=PI_S[form]))
            sources[str(path.resolve())]=digest
            logging.info('%s geometry counter=%d t*=%.3f',form,counter,record['t'])
    geometry=pd.DataFrame(records).sort_values('t').reset_index(drop=True)
    # Join by native counter, avoiding equality joins on CSV-rounded times.
    positions={int(path.stem.split('_')[-1]):i for i,path in enumerate(paths)}
    selected_positions=[positions[int(counter)] for counter in geometry.counter]
    for name in ('E_kin_trans','E_kin_rot','E_kin_tot','E_kin_rot_raw','E_kin_rot_excluded'):
        geometry[name]=energy.iloc[selected_positions][name].to_numpy()
    geometry['u_front']=np.gradient(geometry.x_front,geometry.t)
    geometry['delta_phi']=geometry.phi_t-geometry.phi_t.iloc[0]
    geometry.to_csv(folder/'geometry_timeseries.csv',index=False)
    slices=pd.concat(profiles,ignore_index=True);slices.to_csv(folder/'slice_geometry_timeseries.csv',index=False)
    pd.DataFrame(top_rows).to_csv(folder/'top_layer_settling.csv',index=False)
    last=geometry.iloc[-1];_,last_all=read_frame(paths[-1]);last_mobile=last_all[skip:]
    coords=centers[int(last.counter)]
    best,scores,edge=regress_surface(coords)
    scores.to_csv(folder/'slope_models.csv',index=False);edge.to_csv(folder/'upper_edge.csv',index=False)
    fitted=edge.copy();fitted['y_fit']=best.predict(edge.x);fitted.to_csv(folder/'slope_fit.csv',index=False)
    boundaries=np.r_[edge.x.min(),best.breakpoints,edge.x.max()]
    lengths=np.diff(boundaries);descending=best.slopes<0
    theta_index=int(np.argmax(np.where(descending,lengths,-1))) if descending.any() else 0
    angles=np.full(4,np.nan);angles[:len(best.slopes)]=best.angles_deg
    displacement=np.linalg.norm(last_mobile['x']-initial['x'],axis=1)
    top_cap=initial_y>=np.quantile(initial_y,.95)
    unchanged=top_cap&(displacement<=options.displacement_threshold*options.d_eq)
    x_break=float(initial['x'][unchanged,0].max()) if unchanged.any() else np.nan
    collapse95=t95(geometry.t,geometry.x_front)
    end=dict(form=form,case=str(case.case_root),t_end=qs,t_end_raw=raw_qs,t_geometry=last.t,
        t_collapse=collapse95,t95=collapse95,t_collapse_energy=qs,
        collapse_time_criterion='Front reaches 95% of final observed advance',
        L_end=last.L_t,l=(last.L_t-options.l0/options.d_eq)/(options.l0/options.d_eq),
        H_end=last.H_t,h=last.H_t/(options.h0/options.d_eq),phi_end=last.phi_t,
        u_front_max=float(geometry.u_front.max()),theta=float(abs(best.angles_deg[theta_index])),
        theta_1=angles[0],theta_2=angles[1],theta_3=angles[2],theta_4=angles[3],
        k_best=len(best.slopes),n_punkte=len(edge),Pi_S=PI_S[form],quasistatic=bool(np.isfinite(qs)),
        x_break=x_break,x_break_fit=float(best.breakpoints[0]) if len(best.breakpoints) else np.nan,
        n_unchanged_top=int(unchanged.sum()),chi=options.h0/options.l0,
        slope_selection_criterion=scores.selection_criterion.iloc[0])
    if not np.isfinite(qs):issues.append('Available energy tail does not meet the quasistatic criterion; final geometry is an observed completed-run snapshot')
    if not geometry.complete.all():issues.append('Some alpha/slice geometries are incomplete or use a convex-hull fallback')
    pd.DataFrame([end]).to_csv(folder/'endzustand.csv',index=False)
    end_profiles=slices[slices.counter==int(last.counter)]
    spread=dict(t_geometry=last.t,quasistatic=end['quasistatic'])
    for key in ('L','H','A','phi'):
        value=end_profiles[key]
        spread[key+'_mean']=value.mean();spread[key+'_std']=value.std(ddof=0)
        spread[key+'_normalized_std']=value.std(ddof=0)/abs(value.mean()) if value.mean() else np.nan
    pd.DataFrame([spread]).to_csv(folder/'spanwise_2d.csv',index=False)
    sensitivity=[]
    for alpha in sorted(set([.2,.4,.6,.8,1.,options.alpha])):
        altered=CollapseOptions(**{**options.__dict__,'alpha':alpha})
        _,profile,_,_= _geometry_job((paths[-1],skip,lz,altered))
        row=profile.iloc[np.argmin(abs(profile.z-lz/2))]
        sensitivity.append(dict(alpha=alpha,L_end=row.L/options.d_eq,H_end=row.H/options.d_eq,
            A_end=row.A/options.d_eq**2,phi_end=1-profile.A_solid.sum()/profile.A.sum(),
            t_geometry=last.t,fallback=bool(profile.fallback.any())))
    sensitivity=pd.DataFrame(sensitivity);reference=sensitivity[np.isclose(sensitivity.alpha,options.alpha)].iloc[0]
    for metric in ('L_end','H_end','A_end','phi_end'):
        sensitivity[metric+'_relative_difference']=sensitivity[metric]/reference[metric]-1
    sensitivity.to_csv(folder/'alpha_sensitivity.csv',index=False)
    plateau=sensitivity[sensitivity.alpha.between(.4,.8)]
    alpha_stable=bool((plateau[['L_end_relative_difference','A_end_relative_difference']].abs()<=.05).all().all())
    end['alpha_reference_stable_5pct']=alpha_stable
    if not alpha_stable:issues.append('alpha=0.6 is the common reference, but the 0.4–0.8 geometry sweep is not stable within 5%; alpha uncertainty must accompany interpretation')
    orientation=orientation_statistics(last_mobile,form)
    orientation['particle_row']=np.arange(skip+1,skip+len(last_mobile)+1)
    orientation['t']=last.t;orientation.to_csv(folder/'orientation.csv',index=False)
    pd.DataFrame([dict(form=form,t=last.t,mean_angle_to_flow=orientation.angle_to_flow.mean(),
        std_angle_to_flow=orientation.angle_to_flow.std(ddof=1),
        mean_angle_to_vertical=orientation.angle_to_vertical.mean(),
        order_parameter_flow=float(np.mean((3*orientation.dx**2-1)/2)),n_particles=len(orientation))]).to_csv(folder/'orientation_summary.csv',index=False)
    coll=case.coll/f'ell_coll_{int(last.counter):06d}_3.bin'
    if coll.exists():
        ct,counts,pairs=contact_statistics(coll,skip,len(last_mobile))
        if not np.isclose(ct,last.time,atol=1e-4,rtol=1e-6):raise ValueError('Contact/TRN export times mismatch')
        counts.to_csv(folder/'contact_counts.csv',index=False)
        pd.DataFrame([dict(t=ct*options.u_s/options.d_eq,mean_contact_count=counts.contact_count.mean(),
            n_pairs=len(pairs),definition='Unique nonzero RK3 impulse pairs, including fixed bed')]).to_csv(folder/'contact_summary.csv',index=False)
        sources[str(coll.resolve())]=sha256(coll)
    else:
        pd.DataFrame([dict(t=last.t,mean_contact_count=np.nan,n_pairs=np.nan,
            definition='Unavailable: no synchronous RK3 collision export')]).to_csv(folder/'contact_summary.csv',index=False)
        issues.append('Contact statistics unavailable: synchronous collision exports are absent')
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        bins=list(pool.map(_porosity_bins_job,[(last_mobile,float(z),lz,options.alpha) for z in end_profiles.z]))
    pd.concat(bins,ignore_index=True).to_csv(folder/'porosity_bins_end.csv',index=False)
    budget=geometry[['t','E_kin_trans','E_kin_rot','E_kin_rot_raw','E_kin_tot','E_pot']].copy()
    budget['particle_mechanical_residual']=budget.E_pot.iloc[0]+budget.E_kin_tot.iloc[0]-budget.E_pot-budget.E_kin_tot
    budget['Dissipation']=np.nan
    budget['complete_system_budget']=False
    budget.to_csv(folder/'energy_budget.csv',index=False)
    issues.append('True dissipation unavailable: complete fluid energy and boundary/IBM work terms are not exported as an integrated budget')
    pressure,pressure_sources=pressure_series(case,paths,times,lz,options)
    if not pressure.empty:pressure.to_csv(folder/'excess_pressure.csv',index=False)
    else:issues.append('No synchronous pressure exports for same-height ambient-referenced pressure')
    sources.update(pressure_sources)
    pd.DataFrame([end]).to_csv(folder/'endzustand.csv',index=False)
    status=dict(form=form,case=str(case.case_root),results=str(case.results),n_native_frames=len(paths),
        n_geometry_frames=len(geometry),initial_history_available=bool(times[0]==0),end=end,
        issues=issues,rotation_filter_samples=int((energy.rotation_excluded_monitor!=0).sum()),
        pressure_samples=len(pressure),contact_available=coll.exists(),alpha_stable=alpha_stable)
    (folder/'status.json').write_text(json.dumps(status,indent=2,default=str)+'\n')
    (folder/'herkunft.json').write_text(json.dumps(dict(sha256=sources,options=options.__dict__,
        scripts={str(Path(__file__).resolve()):sha256(__file__)},energy_metadata='energy_herkunft.json',
        frame_header_manifest_sha256=sha256(folder/'frame_inventory.csv'),
        frame_provenance='All native TRN headers hashed in frame_inventory.csv; full selected geometry frames hashed in sha256'),indent=2)+'\n')
    write_case_readme(folder,status,options)
    (folder/'run.log').write_text(f'Completed {form}; native frames={len(paths)}, geometry frames={len(geometry)}.\n'+'\n'.join(issues)+'\n')
    return status


def write_case_readme(folder,status,options):
    Path(folder/'README.md').write_text(f'''# P0–P2: {status['form']}

Quelle: `{status['case']}`; Daten: `{status['results']}`.

Zeit t*=t·u_s/D_eq, u_s={options.u_s}; D_eq={options.d_eq};
E*=E/(rho_p·g·D_eq^4), rho_p={options.rho_p}, g={options.gravity}.
Geometrie: zentrale z-Schnittebene; feste Ursprünge x={options.x_origin},
y={options.y_origin}. L_t=x_front-x_origin; h=H_end/H0, l=(L_end-L0)/L0,
L0={options.l0}, H0={options.h0}. `span` ist separat die bewegliche Polygonbreite.
Porosität: orientierte Ellipsoid-Schnitte in {options.z_slices} periodischen z-Ebenen,
flächengewichtete Porosität über z. Festes Bett ausgeschlossen; Festfläche auf
Alpha-Kontrollvolumen beschnitten. Endzustand zusätzlich in x/y-Bins (50 x-Bins,
vertikale Binweite 1 D_eq). Ellipsenpolygone: vorhandene 10-Punkt-Konvention.

Endzeit: nach der letzten beobachteten Überschreitung nach dem Energiemaximum
muss E_kin_tot <= {options.energy_fraction*100:g}% des Maximums über mindestens
{options.hold_time:g} zusammenhängende t*-Einheiten und drei Samples gelten.
Fehlende Werte unterbrechen ein solches Fenster. Spätere Datenlücken werden
ausdrücklich gemeldet und belegen keine fortgesetzte Ruhe.
`t_collapse`/`t95` ist die erste Erreichung von 95% des finalen beobachteten
Frontvorstoßes; `t_collapse_energy` ist separat die energiebezogene Endzeit.
Finale Geometrie stammt vom letzten gespeicherten
Frame (`t_geometry`); Endzeit beschreibt den Beginn der quasistatischen Phase.
Wenn das Energiekriterium nicht erfüllt ist, bleibt t_end/t_collapse_energy leer und die
finale Geometrie wird ausdrücklich als beobachteter Zustand gekennzeichnet.

Rotationsfilter: Anteil > {options.rotation_share_threshold:g} an der Rohsumme;
einmaliger Vergleich je Zeitschritt. Rohrotation und ausgeschlossene Beiträge
bleiben gespeichert. Hauptkurven gefiltert; mechanischer Partikelrest ist keine
Dissipation. `Dissipation` bleibt leer, bis ein vollständiges Systembudget und
Arbeitsterme vorliegen. Routine `dissipation_from_complete_budget` nimmt diese
explizit entgegen.

Oberkante: höchster Polygonpunkt je x-Bin, euklidischer Abstand <=2 D_eq zum
letzten akzeptierten Punkt. Stetige Regression mit 1–4 Abschnitten; AIC/BIC/AICc
werden exportiert; AICc bei n/k<40, sonst BIC. Theta für Formvergleich: Betrag
 des Winkels des längsten abfallenden Abschnitts. Abbruchkante: größtes initiales
x der obersten 5% Partikel, die sich höchstens {options.displacement_threshold:g}
D_eq bewegt haben; erster Regressionsknick separat (`x_break_fit`).

Orientierung: Prolat-Längsachse bzw. Oblat-Symmetrieachsen-Normale, ohne
Vorzeichen; Winkel zu Fließrichtung x und Vertikalen y.
Kontakte: eindeutige Paare mit nichtnull RK3-Impuls, einschließlich Bettkontakten;
fehlender Export bedeutet unbekannt, nicht null. Dies ist eine Momentaufnahme
aktiver Kollisionsimpulse, keine vollständige persistente Koordinationszahl.
Druck: z=Lz/2; gleicher Höhenbezug in entferntem Fluid x=90..95 D_eq,
Kontrollbereich x=0..L0 und y=0.5..3 D_eq. Differenz entfernt Druck-Gauge und
hydrostatische Höhe; keine Behauptung einer getrennten Poren-/Feststoffmaske.
Normierung entspricht dem vorhandenen PRIME-Druckreader.

Grenzen:
'''+ '\n'.join('- '+s for s in status['issues'])+'\n')
