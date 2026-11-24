"""Render one camera view as MP4 with physically timed, matched fluid frames."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from granular_analysis.config import RENDER_BOUNDS
from granular_analysis.geometry.ellipsoid import _rotation_matrix
from granular_analysis.io.loader import _read_bin_time, load_frame
from granular_analysis.io.fluid_slice import load_fluid_slice, load_fluid_volume, matching_record, restart_records


def physical_time_scale(deq_mm=.615, gravity=9.81):
    if not all(math.isfinite(v) and v > 0 for v in (deq_mm, gravity)):
        raise ValueError('Require positive finite particle diameter and gravity')
    return math.sqrt(deq_mm * 1e-3 / gravity)


def select_video_frames(case, *, component=None, time_start=None, time_end=None):
    """Return (dimensionless time, particle counter), restricted to exact pairs."""
    records = restart_records(case.results) if component else []
    prefix = 'pre' if component == 'pressure' else 'uvw'
    frames = []
    for path in case.require_trn().glob('ell_trn_*.bin'):
        time = _read_bin_time(path)
        if not math.isfinite(time):
            continue
        if time_start is not None and time < time_start:
            continue
        if time_end is not None and time > time_end:
            continue
        if component:
            try:
                row = matching_record(records, time)
            except ValueError:
                continue
            if not (case.results / 'fluid' / f'{prefix}_{int(row["counter"]):06d}.bin').is_file():
                continue
        frames.append((time, int(path.stem.rsplit('_', 1)[1])))
    # At identical restart times, keep the largest particle counter.
    unique = dict(sorted(frames))
    frames = sorted(unique.items())
    if len(frames) < 2:
        raise ValueError('Video requires at least two distinct particle times with matching fluid checkpoints when enabled')
    return frames


def playback_schedule(frames, *, fps=30, speed=1, deq_mm=.615, gravity=9.81):
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(speed) or speed <= 0:
        raise ValueError('Video FPS and speed must be positive and finite')
    scale = physical_time_scale(deq_mm, gravity)
    times = np.array([row[0] for row in frames]) * scale
    duration = (times[-1] - times[0]) / speed
    n_frames = max(2, int(math.ceil(duration * fps)) + 1)
    targets = np.minimum(times[0] + np.arange(n_frames) / fps * speed, times[-1])
    upper = np.clip(np.searchsorted(times, targets), 1, len(times)-1)
    indices = np.where(targets-times[upper-1] <= times[upper]-targets, upper-1, upper)
    return indices, targets, scale


def find_ffmpeg():
    executable = os.environ.get('GRANULAR_FFMPEG') or shutil.which('ffmpeg')
    if executable:
        return executable
    bundled = Path(__file__).resolve().parents[3] / ".tools" / "video" / "imageio_ffmpeg" / "binaries"
    candidates = sorted(bundled.glob("ffmpeg-*"))
    if candidates:
        return str(candidates[0])
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('MP4 requires FFmpeg: install granular-analysis[video] or set GRANULAR_FFMPEG') from None


def run_renderview_video(*, case, out_dir, component=None, view='isometric',
                        fps=30, speed=1, deq_mm=.615, gravity=9.81,
                        window_size=(1920, 1344), lz=5, xlim=None, ylim=None,
                        time_start=None, time_end=None, fluid_clim=None, force=False, legend_scale=1.0, fluid_z=0.0, elevation=15.0, vortex_threshold=None, match_renderview=False, vortex_percentile=99.0, fluid_plane=False, fluid_stride=1):
    if not isinstance(fluid_stride, int) or fluid_stride < 1:
        raise ValueError("Fluid stride must be positive")
    if vortex_threshold is not None and (component not in ("q", "lambda2") or not np.isfinite(vortex_threshold) or not 0 <= vortex_threshold <= 1):
        raise ValueError("Vortex threshold requires Q/lambda2 and a fraction in [0, 1]")
    if not math.isfinite(elevation) or not 0 <= elevation < 90:
        raise ValueError("Video elevation must be finite and in [0, 90)")
    if not math.isfinite(legend_scale) or legend_scale <= 0:
        raise ValueError('Legend scale must be positive and finite')
    if view not in ('isometric', 'top', 'front', 'side'):
        raise ValueError('Unknown video view')
    if len(window_size) != 2 or any(not isinstance(v, int) or v < 2 or v % 2 for v in window_size):
        raise ValueError('MP4 resolution requires positive even WIDTH HEIGHT')
    if not math.isfinite(fluid_z):
        raise ValueError('Fluid plane position must be finite')
    if not math.isfinite(lz) or lz <= 0:
        raise ValueError('Require positive finite LZ')
    if any(v is not None and not math.isfinite(v) for v in (time_start, time_end)):
        raise ValueError('Time limits must be finite')
    if time_start is not None and time_end is not None and time_start > time_end:
        raise ValueError('Time start must not exceed time end')
    for limits in (xlim, ylim, fluid_clim):
        if limits is not None and (len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]):
            raise ValueError('Limits must be finite and increasing')
    if not np.isfinite(vortex_percentile) or not 0 < vortex_percentile <= 100:
        raise ValueError("Vortex percentile must be in (0, 100]")
    volume = vortex_threshold is not None and not fluid_plane
    if match_renderview:
        elevation = None
        legend_scale = 1.0
        fluid_z = lz / 2
    frames = select_video_frames(case, component=component, time_start=time_start, time_end=time_end)
    indices, targets, scale = playback_schedule(frames, fps=fps, speed=speed, deq_mm=deq_mm, gravity=gravity)
    target = Path(out_dir) / 'renderview_video'
    target.mkdir(parents=True, exist_ok=True)
    suffix = (f'_fluid_volume_{component}' if volume else f'_fluid_{component}_z{fluid_z:.2f}') if component else ''
    if volume and fluid_stride > 1:
        suffix += f"_stride{fluid_stride}"
    if match_renderview:
        suffix += '_renderview'
    if vortex_threshold is not None:
        suffix += f"_vortex{100*vortex_threshold:g}pct_refp{vortex_percentile:g}"
    output = target / f'particles_{view}{suffix}.mp4'
    if output.exists() and not force:
        raise FileExistsError(f'{output} already exists; use --force to replace')
    def load_fluid(counter):
        cached = Path(cache_folder) / f'fluid_{counter:06d}.npz'
        if cached.is_file():
            with np.load(cached, allow_pickle=False) as archive:
                data = {}
                for key in archive.files:
                    value = archive[key]
                    data[key] = value.item() if value.ndim == 0 else value
                return data
        fluid = (load_fluid_volume(case, counter, component,
                                  **({"stride": fluid_stride} if fluid_stride > 1 else {}),
                                  **({"xlim": xlim} if xlim is not None else {}),
                                  **({"ylim": ylim} if ylim is not None else {})) if volume else
                 load_fluid_slice(case, counter, fluid_z, component))
        np.savez(cached, **fluid)
        return fluid
    encoder = find_ffmpeg()
    from granular_analysis.plotting.particle_3d_plotter import ellipsoids_to_polydata, render_particle_view
    from granular_analysis.plotting import plotstyle
    selected_indices = np.unique(indices)
    bounds = list((*RENDER_BOUNDS[:4], 0.0, lz))
    with tempfile.TemporaryDirectory(prefix='fluid-cache-', dir=target) as cache_folder:
        minimum, maximum = math.inf, -math.inf
        vortex_maximum = 0.0
        print(f'Video: {len(indices)} playback frames from {len(selected_indices)} stored frames; t* scale={scale:.9g} s, speed={speed:g}, fps={fps:g}', flush=True)
        from granular_analysis.io.fluid_slice import VortexReferencePool
        with VortexReferencePool(target) as reference_pool:
            # Prepass for a fixed framing and colour scale throughout the movie.
            for preparation_index, index in enumerate(selected_indices):
                time, counter = frames[index]
                particles = load_frame(case.require_trn(), f'ell_trn_{counter:06d}.bin')
                for p in particles:
                    rotation = _rotation_matrix(dict(p, q=np.array(p.get('q', [1.,0,0,0]), copy=True)))
                    radius = np.sqrt((rotation**2) @ np.array([p[k] for k in 'abc'])**2)
                    for axis in range(3):
                        bounds[2*axis] = min(bounds[2*axis], p['x'][axis]-radius[axis])
                        bounds[2*axis+1] = max(bounds[2*axis+1], p['x'][axis]+radius[axis])
                if component:
                    fluid = load_fluid(counter)
                    bounds[:4] = [min(bounds[0], fluid['x_edges'][0]), max(bounds[1], fluid['x_edges'][-1]),
                                  min(bounds[2], fluid['y_edges'][0]), max(bounds[3], fluid['y_edges'][-1])]
                    values = np.asarray(fluid['values'])
                    print(f"Prepared fluid {preparation_index+1}/{len(selected_indices)}: grid {values.shape}", flush=True)
                    if vortex_threshold is not None:
                        strength = values if component == 'q' else -values
                        reference_strength = strength
                        if xlim is not None:
                            reference_strength = np.take(reference_strength, np.flatnonzero((fluid['x'] >= xlim[0]) & (fluid['x'] <= xlim[1])), axis=-1)
                        if ylim is not None:
                            reference_strength = np.take(reference_strength, np.flatnonzero((fluid['y'] >= ylim[0]) & (fluid['y'] <= ylim[1])), axis=-2)
                        reference_pool.add(reference_strength)
                    if component in ('magnitude', 'u', 'v', 'w'):
                        values = values / plotstyle.VELOCITY_SCALE
                    elif component == 'q':
                        values = values / plotstyle.Q_SCALE
                    if component in ('q', 'lambda2'):
                        values = values if component == 'q' else -values
                        values = np.where(values > 0, values, np.nan)
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        minimum, maximum = min(minimum, float(finite.min())), max(maximum, float(finite.max()))
            vortex_raw_maximum = reference_pool.maximum
            vortex_sample_count = int(reference_pool.count)
            if vortex_threshold is not None:
                vortex_maximum = reference_pool.percentile(vortex_percentile)
                print(f'Vortex reference P{vortex_percentile:g}={vortex_maximum:.9g}; absolute maximum={vortex_raw_maximum:.9g}; threshold={vortex_threshold*vortex_maximum:.9g}; {vortex_sample_count} positive samples', flush=True)
        if xlim is not None:
            bounds[:2] = xlim
        if ylim is not None:
            bounds[2:4] = ylim
        if component and fluid_clim is None:
            if not math.isfinite(minimum):
                raise ValueError('No finite fluid values in selected frames')
            limit = max(abs(minimum), abs(maximum), np.finfo(float).eps)
            if component in ('q', 'lambda2'):
                if vortex_threshold is not None:
                    maximum = vortex_maximum / (plotstyle.Q_SCALE if component == "q" else 1)
                lower = minimum
                if vortex_threshold is not None and vortex_threshold > 0:
                    lower = max(lower, vortex_threshold * vortex_maximum / (plotstyle.Q_SCALE if component == 'q' else 1))
                fluid_clim = (lower if lower < maximum else maximum / 10, maximum)
            else:
                fluid_clim = (0, limit) if component == 'magnitude' else (-limit, limit)
        metadata = dict(fluid_compute_xlim=xlim if volume else None, fluid_compute_ylim=ylim if volume else None, fluid_gradient_halo_cells=3 if volume and (xlim is not None or ylim is not None) else 0, fluid_stride=fluid_stride, match_renderview=match_renderview, fluid_volume=volume, color_scale="log" if component in ("q", "lambda2") else "linear", view=view, camera_elevation_deg=elevation if view == "isometric" else None, legend_scale=legend_scale, fps=fps, speed=speed, deq_mm=deq_mm, gravity=gravity,
                        seconds_per_t_star=scale, resolution=list(window_size), fluid=component,
                        fluid_plane_z=fluid_z if component and not volume else None, fluid_clim=fluid_clim,
                        vortex_threshold_fraction=vortex_threshold, vortex_reference_value=vortex_maximum,
                        vortex_reference_percentile=vortex_percentile, vortex_raw_maximum=vortex_raw_maximum,
                        vortex_sample_count=vortex_sample_count,
                        vortex_reference_method="exact percentile of positive sampled strengths inside specified x/y limits over rendered source frames",
                        bounds=bounds, physical_duration_s=(frames[-1][0]-frames[0][0])*scale,
                        encoded_duration_s=len(indices)/fps,
                        timing='nearest available exact particle/fluid pair at uniform physical playback times; no interpolation',
                        frames=[dict(counter=frames[i][1], t_star=frames[i][0],
                                     source_time_s=frames[i][0]*scale, playback_time_s=float(targets[j]))
                                for j, i in enumerate(indices)])
        with tempfile.TemporaryDirectory(prefix='granular-video-', dir=target) as temporary:
            temporary = Path(temporary)
            rendered = {}
            for position, index in enumerate(selected_indices):
                time, counter = frames[index]
                particles = load_frame(case.require_trn(), f'ell_trn_{counter:06d}.bin')
                mesh = ellipsoids_to_polydata(particles, nu=32, nv=24)
                fluid = load_fluid(counter) if component else None
                if vortex_threshold is not None:
                    from granular_analysis.io.fluid_slice import filter_vortex_slice
                    fluid = filter_vortex_slice(fluid, vortex_threshold, reference_max=vortex_maximum, reference_percentile=vortex_percentile)
                path = temporary / f'source_{counter:06d}.png'
                render_particle_view(mesh, path, view, domain_bounds=bounds, fit_axes=(xlim is None, ylim is None, True) if match_renderview else (False,False,False),
                                     fluid=fluid, fluid_z=fluid_z, fluid_clim=fluid_clim, window_size=window_size,
                                     export_document=False, legend_scale=legend_scale, camera_elevation=elevation, time_label=f't = {time*scale:.3f} s')
                rendered[index] = path
                print(f'Rendered {position+1}/{len(selected_indices)}: counter {counter:06d}', flush=True)
            for j, index in enumerate(indices):
                os.link(rendered[index], temporary / f'frame_{j:06d}.png')
            encoded = temporary / 'movie.mp4'
            subprocess.run([encoder, '-y', '-loglevel', 'error', '-framerate', str(fps),
                            '-i', str(temporary / 'frame_%06d.png'), '-c:v', 'libx264',
                            '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(encoded)], check=True)
            encoded.replace(output)
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Saved video: {output}', flush=True)
        return output
