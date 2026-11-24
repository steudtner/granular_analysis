#!/usr/bin/env python3
"""Run the supported granular-analysis pipelines for one simulation case."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback
from typing import Callable, Optional

from granular_analysis.config import (
    ALPHA_2D,
    RENDER_BOUNDS,
    LZ,
    SKIP_FIRST_PARTICLES,
    USE_PARTICLE_SKIP,
)
from granular_analysis.io.loader import load_frame, load_last_frame
from granular_analysis.paths import (
    CasePaths,
    latest_common_collision_counter,
    latest_trn_counter,
    output_directory,
)
from granular_analysis.plotting.slice_plotter import plot_exact_slice_from_ellipsoids
from granular_analysis.scripts.run_2d_analysis import main as run_2d
from granular_analysis.scripts.run_collision_distances import main as run_collision_distances
from granular_analysis.scripts.run_energy_analysis import main as run_energy
from granular_analysis.scripts.run_slice_comparison import main as run_slices
from granular_analysis.scripts.run_slope_analysis import main as run_slope
from granular_analysis.scripts.run_alpha_sensitivity import run_sensitivity
from granular_analysis.scripts.run_slope_alpha_sensitivity import run_slope_alpha_sensitivity
from granular_analysis.scripts.run_alignment_analysis import run_alignment

PIPELINE_NAMES = (
    "2d",
    "slices",
    "energy",
    "slope",
    "collision_distances",
    "renderview",
    "alpha_sensitivity",
    "slope_alpha_sensitivity",
    "alignment",
    "porosity_time",
    "particle_fluid",
    "particle_shape",
    "deposit_orientation",
    "renderview_video",
)

# Sensitivity sweeps are deliberately opt-in: they run many alpha values and
# can take substantially longer than the standard per-case analysis.
DEFAULT_PIPELINE_NAMES = PIPELINE_NAMES[:6]


def _resolve_data_folder(path: Optional[str] = None) -> str:
    """Compatibility wrapper returning the canonical TRN directory."""
    return str(CasePaths.from_input(path).trn)


def _resolve_ellipsoid_root(path: str) -> str:
    return str(CasePaths.from_input(path).ellipsoid)


def _resolve_trn_folder(path: str) -> str:
    return str(CasePaths.from_input(path).trn)


def _extract_counter_from_filename(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    for part in reversed(stem.split("_")):
        if part.isdigit():
            return int(part)
    return None


def _load_particles(case: CasePaths, counter: Optional[int]):
    trn = case.require_trn()
    if counter is None:
        return load_last_frame(trn)
    return load_frame(trn, f"ell_trn_{counter:06d}.bin")


def _run_exact_slice_plot(args, case: CasePaths, out_dir: Path, counter: Optional[int]):
    from granular_analysis.io.fluid_slice import load_fluid_slice
    if counter is None:
        counter = latest_trn_counter(case)
    if counter is None:
        raise FileNotFoundError(f"No particle frames in {case.trn}")
    ellipsoids = _load_particles(case, counter)
    z_value = args.z if args.z is not None else LZ / 2
    skip = (
        args.skip_points
        if args.skip_points is not None
        else SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0
    )
    use_upper_edge = bool(args.upper_edge)
    use_alphashape = bool(args.alphashape or use_upper_edge)
    suffix = "_upperEdge" if use_upper_edge else ""
    fluid = load_fluid_slice(case, counter, z_value, args.fluid) if args.fluid else None
    if args.vortex_threshold is not None and fluid is not None:
        from granular_analysis.io.fluid_slice import filter_vortex_slice
        fluid = filter_vortex_slice(fluid, args.vortex_threshold, reference_percentile=args.vortex_percentile)
    fluid_suffix = f"_{args.fluid}" if args.fluid else ""
    if args.vortex_threshold is not None:
        fluid_suffix += f"_vortex{100*args.vortex_threshold:g}pct"
    if args.fluid_grid:
        fluid_suffix += "_grid"
    if args.velocity_directions:
        fluid_suffix += "_arrows"
    output_path = out_dir / "2d" / f"alphashape_slice_{counter:06d}_z{z_value:.2f}{fluid_suffix}{suffix}.svg"

    return plot_exact_slice_from_ellipsoids(
        ellipsoids,
        z_plane=z_value,
        output_path=output_path,
        alphashape=use_alphashape,
        alpha=args.alpha if args.alpha is not None else ALPHA_2D,
        upper_edge=use_upper_edge,
        upper_edge_bins=args.upper_edge_bins,
        densify_max_segment_length=args.densify,
        skip_points=skip,
        xlim=tuple(args.xlim) if args.xlim else (0.0, 60.0),
        ylim=tuple(args.ylim) if args.ylim else (0.0, 20.0),
        periodic_length=LZ,
        fluid=fluid,
        fluid_grid=args.fluid_grid,
        velocity_colors=not args.no_velocity_arrows,
        velocity_directions=args.velocity_directions,
    )


def run_renderview(
    *,
    case: CasePaths,
    counter: Optional[int],
    out_dir: Path,
    force: bool = False,
    xlim=None,
    ylim=None,
    fluid_component=None,
    fluid_midplane=False,
    lz=LZ,
    window_size=(8000, 5600),
    vortex_threshold=None,
    vortex_percentile=99.0,
    fluid_stride=1,
    views=None,
    red_mobile_particles=False,
    skip_first=None,
):
    """Render requested views of one transient ellipsoid frame."""
    if vortex_threshold is not None:
        if fluid_component not in ("q", "lambda2") or not math.isfinite(vortex_threshold) or not 0 <= vortex_threshold <= 1:
            raise ValueError("Vortex threshold requires Q/lambda2 and a fraction in [0, 1]")
    if fluid_component and vortex_threshold is None:
        fluid_midplane = True
    counter = counter if counter is not None else latest_trn_counter(case)
    if counter is None:
        raise FileNotFoundError(f"No TRN frames found in {case.trn}")

    if len(window_size) != 2 or any(size <= 0 for size in window_size):
        raise ValueError("Render resolution must contain positive WIDTH HEIGHT")
    render_dir = out_dir / "renderview"
    prefix = f"particles_{counter:06d}"
    mobile_start = ((SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0)
                    if skip_first is None else skip_first) if red_mobile_particles else None
    if mobile_start is not None and mobile_start < 0:
        raise ValueError('Mobile particle start must be nonnegative')
    if red_mobile_particles:
        prefix += '_mobile_red'
    if fluid_midplane:
        prefix += f"_fluid_midplane_{fluid_component or 'magnitude'}"
    if fluid_component and not fluid_midplane:
        prefix += f"_fluid_volume_{fluid_component}"
    views = tuple(views or ("isometric", "top", "front", "side"))
    if vortex_threshold is not None:
        prefix += f"_vortex{100*vortex_threshold:g}pct_refp{vortex_percentile:g}"
        if fluid_stride > 1:
            prefix += f"_stride{fluid_stride}"
    bounds = tuple(float(value) for value in (
        *(xlim if xlim is not None else RENDER_BOUNDS[:2]),
        *(ylim if ylim is not None else RENDER_BOUNDS[2:4]),
        *RENDER_BOUNDS[4:],
    ))
    if any(not (low < high) for low, high in zip(bounds[::2], bounds[1::2])):
        raise ValueError("Render limits must have increasing, finite bounds")
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Render limits must have increasing, finite bounds")
    expected = [render_dir / f"{prefix}_{view}{ext}"
                for view in views for ext in (".png", ".pdf", ".svg")]
    bounds_file = render_dir / f"{prefix}_bounds.json"
    cache_key = {"version": 11, "bounds": list(bounds),
                 "fluid": fluid_component, "fluid_z": lz / 2 if fluid_midplane else None,
                 "fit_axes": [xlim is None, ylim is None, True],
                 "window_size": list(window_size), "vortex_threshold": vortex_threshold, "vortex_percentile": vortex_percentile, "fluid_stride": fluid_stride, "views": list(views)}
    if red_mobile_particles:
        cache_key['mobile_start'] = mobile_start
    try:
        cached_bounds = json.loads(bounds_file.read_text())
    except (OSError, ValueError):
        cached_bounds = None
    if not force and cached_bounds == cache_key and all(path.exists() for path in expected):
        print(f"Renderview {counter:06d} already exists; skipping.")
        return

    from granular_analysis.plotting.particle_3d_plotter import render_particle_views

    particles = _load_particles(case, counter)
    fluid = None
    if fluid_midplane or fluid_component:
        from granular_analysis.io.fluid_slice import load_fluid_slice, load_fluid_volume
        fluid = (load_fluid_slice(case, counter, lz / 2, fluid_component or "magnitude")
                 if fluid_midplane else load_fluid_volume(case, counter, fluid_component, stride=fluid_stride, xlim=xlim, ylim=ylim))
        if vortex_threshold is not None:
            from granular_analysis.io.fluid_slice import filter_vortex_slice
            fluid = filter_vortex_slice(fluid, vortex_threshold, reference_percentile=vortex_percentile)
    render_particle_views(
        particles,
        output_dir=render_dir,
        views=views,
        filename_prefix=prefix,
        domain_bounds=bounds,
        fluid=fluid,
        fluid_z=lz / 2 if fluid_midplane else None,
        fit_axes=(xlim is None, ylim is None, True),
        window_size=tuple(window_size),
        mobile_start=mobile_start,
    )
    bounds_file.write_text(json.dumps(cache_key) + "\n")


def run_pipeline(name: str, action: Callable[[], object]):
    print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")
    return action()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=PIPELINE_NAMES,
                        help="Run only these pipelines (default: 2d, slices, energy, slope, collision_distances, renderview).")
    parser.add_argument("--exclude", nargs="+", choices=PIPELINE_NAMES, default=[],
                        help="Remove these pipelines from the selected set.")
    parser.add_argument("--path", help="Case root or any supported results subdirectory.")
    parser.add_argument("--folder", help="Deprecated alias for --path.")
    parser.add_argument("--out", help="Analysis output directory.")
    parser.add_argument("--compare", nargs="+", metavar="CASE",
                        help="Compare energy, 2d, slices or slope with additional cases; primary case is --path or current directory.")
    parser.add_argument("--compare-labels", nargs="+", metavar="LABEL",
                        help="Legend labels: primary case first, then --compare cases.")
    parser.add_argument("--energy-component", choices=("kinetic", "translational", "rotational"), default="kinetic",
                        help="Energy quantity for plots and comparisons (default: total kinetic).")
    parser.add_argument("--rotation-share-threshold", type=float, default=0.5,
                        help="Exclude dominant particle rotation above this raw energy share (0.5 to 1; 1 disables).")
    parser.add_argument("--energy-raw", action="store_true", help="Compare unnormalized energy against simulation time.")
    parser.add_argument("--potential-energy", action="store_true", help="Show the initial potential-energy reference.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--counter", type=int,
        help="Simulation frame counter; for energy, a zero-based histogram timestep index.",
    )
    selection.add_argument("--time", type=float,
                           help="Select the nearest stored simulation time; with --fluid, the nearest exact particle/fluid pair.")
    parser.add_argument(
        "--histogram-timesteps",
        help=(
            "Comma-separated monitor timestep indices for energy histograms "
            "(e.g. 0,100,200,-1). Overrides --counter for energy. "
            "Defaults to the final timestep when neither option is given."
        ),
    )
    parser.add_argument("--porosity-frames", type=int, default=15,
                        help="Times sampled for porosity_time (0 = all available frames).")
    parser.add_argument("--porosity-profiles", type=int, default=15,
                        help="Maximum time-coloured z profiles in porosity_time.")
    parser.add_argument("--porosity-z-step", type=float, default=0.1,
                        help="Maximum periodic z spacing for porosity_time.")
    parser.add_argument("--time-start", type=float,
                        help="First included physical time (porosity_time; also restricts video frame selection).")
    parser.add_argument("--time-end", type=float,
                        help="Last included physical time (porosity_time; also restricts video frame selection).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even when a valid cache or render exists.")
    parser.add_argument("--velocity-arrows", "--particle-velocity-arrows",
                        dest="velocity_directions", action="store_true",
                        help="Show in-plane particle velocity directions in exact 2D slices; colours show speed.")
    parser.add_argument("--no-velocity-colors", "--no-velocity-arrows",
                        dest="no_velocity_arrows", action="store_true",
                        help="Disable white-to-red particle speed colouring in exact 2D slice plots.")
    parser.add_argument("--orientation-mode", choices=("auto", "prolate", "oblate"), default="auto",
                        help="Deposit orientation axis: auto detects shape, prolate uses a axis, oblate c axis.")
    parser.add_argument("--orientation-bins", type=int, default=6,
                        help="Number of equal-width x-bins for deposit_orientation (default: 6).")
    parser.add_argument("--orientation-renderview", metavar="PNG",
                        help="Also combine deposit orientation insets below an existing renderview PNG.")
    parser.add_argument("--orientation-angle-step", type=float, default=10.0,
                        help="Angle bin width in degrees for deposit_orientation (default: 10).")
    parser.add_argument("--alignment-mode", choices=("prolate", "oblate"), default="prolate",
                        help="Alignment to longest axis (prolate), or to c axis (oblate).")
    parser.add_argument("--alignment-min-speed", type=float, default=0.0,
                        help="Exclude speeds at or below this value from alignment angles.")
    parser.add_argument("--cluster-eps", type=float, default=0.3,
                        help="DBSCAN radius in standardized angle/linear-speed coordinates.")
    parser.add_argument("--cluster-min-samples", type=int, default=20,
                        help="DBSCAN minimum samples per cluster (default: 20).")
    parser.add_argument("--vortex-percentile", type=float, default=99.0, help="Robust positive vortex strength reference percentile (default: 99; 100 = maximum).")
    parser.add_argument("--video-fluid-stride", "--fluid-stride", type=int, default=1, help="3D preview grid stride (1 = full resolution; 4 = coarser complete domain).")
    parser.add_argument("--video-fluid-plane", action="store_true", help="Render the fluid slice rather than the vortex volume in videos.")
    parser.add_argument("--vortex-threshold", type=float, help="Hide weak vortex cells below this fraction of max strength, e.g. 0.2; requires --fluid q or lambda2.")
    parser.add_argument("--render-views", nargs="+", choices=("isometric", "top", "front", "side"), help="Static renderview views to render (default: all).")
    parser.add_argument("--red-mobile-particles", action="store_true",
                        help="Color free particles red in renderview; initial --skip-points wall particles stay gray (default: 2000).")
    parser.add_argument("--fluid", nargs="?", const="magnitude",
                        type=str.lower, choices=("magnitude", "u", "v", "w", "pressure", "lambda2", "q"),
                        help="Overlay velocity magnitude (default), u/v/w component, pressure, lambda2 or Q at exactly the particle time.")
    parser.add_argument("--video-match-renderview", action="store_true", help="Use the static renderview camera and legend size.")
    parser.add_argument("--video-elevation", type=float, default=15.0, help="Isometric video camera elevation above the xz plane in degrees (default: 15; 0 = horizontal).")
    parser.add_argument("--video-view", choices=("isometric", "top", "front", "side"), default="isometric",
                        help="Camera view for renderview_video (default: isometric).")
    parser.add_argument("--video-fluid-z", type=float, default=0.0, help="Video fluid plane z-position (default: 0, behind particles in the standard view).")
    parser.add_argument("--video-legend-scale", type=float, default=1.0, help="Video legend size factor (default: 1.0).")
    parser.add_argument("--video-fps", type=float, default=30,
                        help="Video frame rate (default: 30).")
    parser.add_argument("--video-speed", type=float, default=1, help="Playback speed relative to physical time (1 = real time).")
    parser.add_argument("--video-resolution", nargs=2, type=int, default=(1920, 1344), metavar=("WIDTH", "HEIGHT"),
                        help="Video resolution in pixels; even values (default: 1920 1344).")
    parser.add_argument("--video-deq-mm", type=float, default=.615,
                        help="D_eq in millimeters for the video time conversion t = t* sqrt(D_eq/g) (default: 0.615).")
    parser.add_argument("--video-gravity", type=float, default=9.81,
                        help="Gravity in m/s^2 for the video time conversion (default: 9.81).")
    parser.add_argument("--render-resolution", nargs=2, type=int, default=(8000, 5600),
                        metavar=("WIDTH", "HEIGHT"),
                        help="Renderview resolution in pixels (default: 8000 5600).")
    parser.add_argument("--fluid-midplane", action="store_true",
                        help="Overlay an x-y fluid slice at z=--lz/2 in all renderviews; defaults to --fluid magnitude.")
    parser.add_argument("--fluid-grid", action="store_true",
                        help="Overlay actual fluid cell boundaries; defaults to --fluid magnitude.")
    parser.add_argument("--fluid-clim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Fixed fluid colour scale for particle_fluid; use identical values across cases.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue after a failed pipeline; the exit status stays unsuccessful.")
    parser.add_argument("--dry", action="store_true", help="Print actions without writing files.")
    parser.add_argument(
        "--alpha", type=float,
        help="Alpha-shape parameter for a single 2d or slices analysis.",
    )
    parser.add_argument(
        "--lz", type=float, default=LZ,
        help="Periodic z-length for 2d and sensitivity analyses.",
    )
    parser.add_argument(
        "--alpha-start", type=float, default=0.1,
        help="First alpha value for a sensitivity sweep (default: 0.1).",
    )
    parser.add_argument(
        "--alpha-stop", type=float, default=1.0,
        help="Last alpha value for a sensitivity sweep (default: 1.0).",
    )
    parser.add_argument(
        "--alpha-step", type=float, default=0.1,
        help="Positive alpha increment for a sensitivity sweep (default: 0.1).",
    )
    parser.add_argument("--alphashape", action="store_true",
                        help="Draw the alpha-shape in the exact-slice plot; with --only 2d, also request that plot.")
    parser.add_argument("--upper-edge", "--upper_edge", dest="upper_edge", action="store_true",
                        help="Draw the binned upper edge and alpha-shape in the exact-slice plot; with --only 2d, also request that plot.")
    parser.add_argument("--upper-edge-bins", "--upper_edge_bins", dest="upper_edge_bins", type=int, default=50,
                        help="Upper-edge x-bin count (default: 50).")
    parser.add_argument("--densify", type=float,
                        help="Maximum boundary-segment length when densifying slice geometry.")
    parser.add_argument("--length-unit-mm", type=float,
                        help="Millimeters per input length unit for DIN particle_shape sieve curves (configured default: 0.615).")
    parser.add_argument("--skip-points", "--skip_points", "--skip-particles", "--skip_particles", dest="skip_points", type=int,
                        help="Exclude the first N particles: wall particles in geometry pipelines, input rows in particle_shape and energy (default: configured 2000).")
    parser.add_argument("--z", type=float,
                        help="z-plane for 2d, exact-slice plots and sensitivity sweeps (default: Lz/2).")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("XMIN", "XMAX"),
                        help="Exact-slice and renderview x-axis limits.")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"),
                        help="Exact-slice and renderview y-axis limits; also caps video vortex previews and deposit bins.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if any(size <= 0 for size in args.render_resolution):
        parser.error("--render-resolution requires positive WIDTH HEIGHT")
    if args.path and args.folder:
        parser.error("Use either --path or --folder, not both.")
    if args.compare and (not args.only or set(args.only) - {"energy", "2d", "slices", "slope"}):
        parser.error("--compare supports --only energy 2d slices slope")

    case = CasePaths.from_input(args.path or args.folder)
    out_dir = output_directory(args.out)
    selected = list(args.only or DEFAULT_PIPELINE_NAMES)
    selected = [name for name in selected if name not in args.exclude]
    if not selected:
        print("No pipelines selected.")
        return 0
    if "deposit_orientation" in selected:
        if args.orientation_bins < 1:
            parser.error("--orientation-bins must be positive")
        if not math.isfinite(args.orientation_angle_step) or not 0 < args.orientation_angle_step <= 90:
            parser.error("--orientation-angle-step must be finite and in (0, 90]")
        if args.skip_points is not None and args.skip_points < 0:
            parser.error("--skip-points must be nonnegative")
    if not math.isfinite(args.vortex_percentile) or not 0 < args.vortex_percentile <= 100:
        parser.error("--vortex-percentile must be in (0, 100]")
    if args.vortex_threshold is not None:
        if not math.isfinite(args.vortex_threshold) or not 0 <= args.vortex_threshold <= 1:
            parser.error("--vortex-threshold must be in [0, 1]")
        if args.fluid not in ("q", "lambda2"):
            parser.error("--vortex-threshold requires --fluid q or lambda2")
        if not any(name in selected for name in ("2d", "slices", "renderview", "renderview_video")):
            parser.error("--vortex-threshold requires 2d, slices, renderview or renderview_video")
    if args.fluid_clim is not None:
        if not any(name in selected for name in ("particle_fluid", "renderview_video")):
            parser.error("--fluid-clim requires particle_fluid or renderview_video")
        if not all(math.isfinite(v) for v in args.fluid_clim) or args.fluid_clim[0] >= args.fluid_clim[1]:
            parser.error("--fluid-clim requires finite increasing MIN MAX")
    if "porosity_time" in selected:
        if args.counter is not None or args.time is not None:
            parser.error("porosity_time uses --time-start/--time-end, not --counter/--time")
        if args.porosity_frames < 0 or args.porosity_profiles < 1:
            parser.error("Require --porosity-frames >= 0 and --porosity-profiles >= 1")
        if not math.isfinite(args.lz) or not 0 < args.porosity_z_step < args.lz:
            parser.error("Require finite --lz > --porosity-z-step > 0")
        if any(t is not None and not math.isfinite(t) for t in (args.time_start, args.time_end)):
            parser.error("Time limits must be finite")
        if args.time_start is not None and args.time_end is not None and args.time_start > args.time_end:
            parser.error("--time-start must not exceed --time-end")
    elif "renderview_video" not in selected and (args.time_start is not None or args.time_end is not None):
        parser.error("--time-start/--time-end require porosity_time or renderview_video")
    if args.alpha_step <= 0:
        parser.error("--alpha-step must be positive")
    if args.alpha_stop < args.alpha_start:
        parser.error("--alpha-stop must be greater than or equal to --alpha-start")

    if args.velocity_directions and (args.compare or not any(name in selected for name in ("2d", "slices"))):
        parser.error("--velocity-arrows requires 2d or slices without --compare")
    if args.compare:
        if args.fluid_grid:
            parser.error("--fluid-grid is not supported in comparison mode")
        if args.fluid:
            parser.error("Fluid overlays are not supported in comparison mode")
        if selected == ["energy"] and (args.counter is not None or args.time is not None):
            parser.error("Energy comparison uses full curves; omit --counter and --time")
        if args.histogram_timesteps is not None:
            parser.error("Comparison does not use --histogram-timesteps")
        paths = [str(case.case_root), *args.compare]
        if args.dry:
            print(f"Compare {selected}: {paths}; time={args.time}, counter={args.counter}")
            return 0
        try:
            if "energy" in selected:
                from granular_analysis.scripts.run_energy_comparison import run_energy_comparison
                run_energy_comparison(paths, out_dir / "comparison/energy", labels=args.compare_labels,
                    component=args.energy_component, normalized=not args.energy_raw, force=args.force,
                    skip_first=args.skip_points, potential_energy=args.potential_energy,
                    rotation_share_threshold=args.rotation_share_threshold)
            geometry = [name for name in selected if name != "energy"]
            if geometry:
                from granular_analysis.scripts.run_geometry_comparison import run_geometry_comparison
                run_geometry_comparison(paths, out_dir / "comparison/geometry", geometry,
                    labels=args.compare_labels, counter=args.counter, time=args.time, z=args.z,
                    alpha=args.alpha if args.alpha is not None else ALPHA_2D, lz=args.lz,
                    skip_first=args.skip_points, densify=args.densify,
                    upper_edge_bins=args.upper_edge_bins, force=args.force)
        except Exception as exc:
            print(f"[ERROR] Comparison: {type(exc).__name__}: {exc}")
            return 1
        print("All requested comparisons completed successfully.")
        return 0

    if "renderview_video" in selected and (args.counter is not None or args.time is not None):
        parser.error("renderview_video uses --time-start/--time-end, not --counter/--time")
    if args.fluid_midplane:
        if "renderview" not in selected or args.compare:
            parser.error("--fluid-midplane requires renderview without --compare")
        if not math.isfinite(args.lz) or args.lz <= 0:
            parser.error("--fluid-midplane requires finite --lz > 0")
        if args.fluid is None:
            args.fluid = "magnitude"

    if args.fluid_grid:
        if not any(name in selected for name in ("2d", "slices", "particle_fluid")):
            parser.error("--fluid-grid requires 2d, slices, or particle_fluid")
        if args.fluid is None:
            args.fluid = "magnitude"
    if "particle_fluid" in selected and args.fluid is None:
        args.fluid = "magnitude"
    counter = args.counter
    if args.fluid and not args.fluid_midplane and not any(name in selected for name in ("2d", "slices", "particle_fluid", "renderview", "renderview_video")):
        parser.error("--fluid requires 2d, slices, or particle_fluid")
    if args.time is not None and not any(name in selected for name in ("2d", "slices", "renderview", "alignment", "particle_fluid", "deposit_orientation")):
        parser.error("--time requires 2d, slices, renderview, alignment, particle_fluid, or deposit_orientation")
    chosen_time = None
    fluid_counter = None
    try:
        from granular_analysis.io.fluid_slice import (
            latest_common_counter, matching_record, nearest_frame, restart_records,
        )
        from granular_analysis.io.loader import _read_bin_time
        if args.time is not None:
            chosen_time, counter = nearest_frame(case, args.time, args.fluid)
        elif counter is None and args.fluid and selected != ["renderview_video"]:
            counter = latest_common_counter(case, args.fluid)
        if args.fluid and selected != ["renderview_video"]:
            chosen_time = _read_bin_time(case.trn / f"ell_trn_{counter:06d}.bin")
            try:
                row = matching_record(restart_records(case.results), chosen_time)
                fluid_counter = int(row["counter"])
                prefix = "pre" if args.fluid == "pressure" else "uvw"
                fluid_path = case.results / "fluid" / f"{prefix}_{fluid_counter:06d}.bin"
                if not fluid_path.is_file():
                    raise ValueError(f"Missing fluid file: {fluid_path}")
            except ValueError as exc:
                alternative_time, alternative_counter = nearest_frame(case, chosen_time, args.fluid)
                raise ValueError(
                    f"{exc}\nNearest matching pair: --counter {alternative_counter} "
                    f"at t={alternative_time:.12g}.\n"
                    f"Or use --time {chosen_time:.12g} to select the nearest matching pair automatically."
                ) from None
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[ERROR] Frame selection: {exc}")
        return 1
    if counter is None and any(name in selected for name in ("2d", "slices", "renderview")):
        counter = latest_trn_counter(case)

    print(f"CASE       = {case.case_root}")
    print(f"TRN        = {case.trn}")
    print(f"OUTPUT     = {out_dir}")
    print(f"PIPELINES  = {', '.join(selected)}")
    print(f"COUNTER    = {counter if counter is not None else 'automatic'}")
    if chosen_time is not None:
        print(f"TIME       = {chosen_time:.12g}" +
              (f" (requested {args.time:.12g}, difference {chosen_time - args.time:+.6g})"
               if args.time is not None else ""))
    if fluid_counter is not None:
        print(f"FLUID      = {fluid_counter:06d} (same simulation time)")

    if args.dry:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []

    for name in selected:
        try:
            if name == "2d":
                action = lambda: run_2d(
                    force=args.force,
                    path=str(case.trn),
                    out=str(out_dir),
                    z_value=args.z if args.z is not None else LZ / 2,
                    densify=args.densify,
                    upper_edge_bins=args.upper_edge_bins,
                    alpha=args.alpha if args.alpha is not None else ALPHA_2D,
                    lz=args.lz,
                    counter=counter,
                )
            elif name == "slices":
                action = lambda: run_slices(
                    force=args.force,
                    path=str(case.trn),
                    out=str(out_dir),
                    z_value=args.z,
                    alpha=args.alpha if args.alpha is not None else ALPHA_2D,
                    skip_first=args.skip_points,
                    densify=args.densify,
                    upper_edge_bins=args.upper_edge_bins,
                    counter=counter,
                )
            elif name == "particle_fluid":
                from granular_analysis.scripts.render_particle_fluid import run_particle_fluid
                action = lambda: run_particle_fluid(
                    case, out_dir / "particle_fluid", counter=counter,
                    z=args.z if args.z is not None else 2.25,
                    component=args.fluid, skip_first=args.skip_points,
                    bounds=(*(args.xlim or (0.,50.)), *(args.ylim or (0.,30.))),
                    clim=args.fluid_clim, fluid_grid=args.fluid_grid,
                )
            elif name == "porosity_time":
                from granular_analysis.scripts.run_porosity_time import run_porosity_time
                action = lambda: run_porosity_time(
                    case, out_dir / "porosity_time", n_frames=args.porosity_frames,
                    n_profiles=args.porosity_profiles, time_start=args.time_start,
                    time_end=args.time_end, z_step=args.porosity_z_step, lz=args.lz,
                    alpha=args.alpha if args.alpha is not None else ALPHA_2D,
                    skip_first=args.skip_points, force=args.force,
                )
            elif name == "particle_shape":
                from granular_analysis.scripts.run_particle_shape import run_particle_shape
                action = lambda: run_particle_shape(case=case, out_dir=out_dir, skip_first=args.skip_points, length_unit_mm=args.length_unit_mm)
            elif name == "renderview_video":
                from granular_analysis.scripts.run_renderview_video import run_renderview_video
                action = lambda: run_renderview_video(
                    case=case, out_dir=out_dir, component=args.fluid, view=args.video_view,
                    fps=args.video_fps, speed=args.video_speed, deq_mm=args.video_deq_mm,
                    legend_scale=args.video_legend_scale, fluid_z=args.video_fluid_z, elevation=args.video_elevation,
                    vortex_threshold=args.vortex_threshold, match_renderview=args.video_match_renderview,
                    vortex_percentile=args.vortex_percentile, fluid_plane=args.video_fluid_plane, fluid_stride=args.video_fluid_stride,
                    gravity=args.video_gravity, window_size=tuple(args.video_resolution),
                    lz=args.lz, xlim=args.xlim, ylim=args.ylim,
                    time_start=args.time_start, time_end=args.time_end, fluid_clim=args.fluid_clim,
                    force=args.force,
                )
            elif name == "deposit_orientation":
                from granular_analysis.scripts.run_deposit_orientation import run_deposit_orientation
                action = lambda: run_deposit_orientation(
                    case=case, out_dir=out_dir, counter=counter,
                    n_bins=args.orientation_bins, angle_step=args.orientation_angle_step,
                    z_plane=args.z, lz=args.lz,
                    mode=args.orientation_mode, xlim=args.xlim, ylim=args.ylim,
                    skip_first=args.skip_points,
                    renderview_image=args.orientation_renderview,
                )
            elif name == "alignment":
                action = lambda: run_alignment(
                    case=case, out_dir=out_dir, counter=counter,
                    mode=args.alignment_mode, min_speed=args.alignment_min_speed,
                    cluster_eps=args.cluster_eps, cluster_min_samples=args.cluster_min_samples,
                )
            elif name == "energy":
                action = lambda: run_energy(
                    force=args.force,
                    skip_first=args.skip_points,
                    potential_energy=args.potential_energy,
                    rotation_share_threshold=args.rotation_share_threshold,
                    path=str(case.trn),
                    out=str(out_dir / "energy.csv"),
                    histogram_timesteps=(
                        args.histogram_timesteps
                        if args.histogram_timesteps is not None
                        else str(args.counter) if args.counter is not None else None
                    ),
                )
            elif name == "slope":
                action = lambda: run_slope(
                    force=args.force,
                    out=str(out_dir),
                    alpha=args.alpha,
                    xlim=tuple(args.xlim) if args.xlim else (0.0, 60.0),
                    ylim=tuple(args.ylim) if args.ylim else (0.0, 20.0),
                )
            elif name == "collision_distances":
                collision_counter = (
                    args.counter if args.counter is not None
                    else latest_common_collision_counter(case)
                )
                if collision_counter is None:
                    collision_files = list(case.coll.glob("ell_coll_*.bin"))
                    if not collision_files:
                        raise FileNotFoundError(
                            "No collision files found in "
                            f"{case.coll}. Expected files named "
                            "ell_coll_<counter>_<stage>.bin. PRIME must be "
                            "compiled with _ellipsoid_write_collision_ and "
                            "must write COLL data during the simulation."
                        )
                    raise FileNotFoundError(
                        "TRN and collision files exist, but no counter has "
                        "a TRN frame and all required collision RK stages."
                    )
                action = lambda counter=collision_counter: run_collision_distances(
                    counter=counter,
                    path=str(case.ellipsoid),
                    out=str(out_dir),
                )
            elif name == "alpha_sensitivity":
                action = lambda: run_sensitivity(
                    case.case_root,
                    out_dir / "alpha_sensitivity",
                    z_value=args.z if args.z is not None else args.lz / 2,
                    lz=args.lz,
                    alpha_start=args.alpha_start,
                    alpha_stop=args.alpha_stop,
                    alpha_step=args.alpha_step,
                    skip_first=(
                        args.skip_points if args.skip_points is not None
                        else SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0
                    ),
                )
            elif name == "slope_alpha_sensitivity":
                action = lambda: run_slope_alpha_sensitivity(
                    case.case_root,
                    out_dir / "slope_alpha_sensitivity",
                    z_value=args.z if args.z is not None else args.lz / 2,
                    lz=args.lz,
                    alpha_start=args.alpha_start,
                    alpha_stop=args.alpha_stop,
                    alpha_step=args.alpha_step,
                    skip_first=(
                        args.skip_points if args.skip_points is not None
                        else SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0
                    ),
                    upper_edge_bins=args.upper_edge_bins,
                )
            else:
                action = lambda: run_renderview(
                    case=case,
                    counter=counter,
                    out_dir=out_dir,
                    force=args.force,
                    xlim=tuple(args.xlim) if args.xlim else None,
                    ylim=tuple(args.ylim) if args.ylim else None,
                    fluid_component=args.fluid,
                    window_size=tuple(args.render_resolution),
                    vortex_threshold=args.vortex_threshold, vortex_percentile=args.vortex_percentile, fluid_stride=args.video_fluid_stride, views=args.render_views,
                    fluid_midplane=args.fluid_midplane,
                    lz=args.lz,
                    red_mobile_particles=args.red_mobile_particles,
                    skip_first=args.skip_points,
                )
            run_pipeline(name, action)
        except Exception as exc:
            errors.append((name, exc))
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if not args.continue_on_error:
                break

    run_exact_slice = (
        not errors
        and (
            args.only is None
            or "slices" in selected
            or "2d" in selected
        )
    )
    if run_exact_slice:
        try:
            run_pipeline(
                "exact_slice_plot",
                lambda: _run_exact_slice_plot(args, case, out_dir, counter),
            )
        except Exception as exc:
            errors.append(("exact_slice_plot", exc))
            print(f"[ERROR] exact_slice_plot: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    if errors:
        print("\nFailed steps:")
        for name, exc in errors:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
        return 1

    print("\nAll requested steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
