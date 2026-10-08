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
    fluid_suffix = f"_{args.fluid}" if args.fluid else ""
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
):
    """Render four views of one transient ellipsoid frame."""
    counter = counter if counter is not None else latest_trn_counter(case)
    if counter is None:
        raise FileNotFoundError(f"No TRN frames found in {case.trn}")

    render_dir = out_dir / "renderview"
    prefix = f"particles_{counter:06d}"
    views = ("isometric", "top", "front", "side")
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
    try:
        cached_bounds = json.loads(bounds_file.read_text())
    except (OSError, ValueError):
        cached_bounds = None
    if not force and cached_bounds == list(bounds) and all(path.exists() for path in expected):
        print(f"Renderview {counter:06d} already exists; skipping.")
        return

    from granular_analysis.plotting.particle_3d_plotter import render_particle_views

    particles = _load_particles(case, counter)
    render_particle_views(
        particles,
        output_dir=render_dir,
        views=views,
        filename_prefix=prefix,
        domain_bounds=bounds,
    )
    bounds_file.write_text(json.dumps(bounds) + "\n")


def run_pipeline(name: str, action: Callable[[], object]):
    print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")
    return action()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=PIPELINE_NAMES)
    parser.add_argument("--exclude", nargs="+", choices=PIPELINE_NAMES, default=[])
    parser.add_argument("--path", help="Case root or any supported results subdirectory.")
    parser.add_argument("--folder", help="Deprecated alias for --path.")
    parser.add_argument("--out", help="Analysis output directory.")
    parser.add_argument("--compare", nargs="+", metavar="CASE",
                        help="Compare energy, 2d, slices or slope with additional cases; primary case is --path or current directory.")
    parser.add_argument("--compare-labels", nargs="+", metavar="LABEL",
                        help="Legend labels: primary case first, then --compare cases.")
    parser.add_argument("--energy-component", choices=("kinetic", "translational", "rotational"), default="kinetic")
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
    parser.add_argument("--time-start", type=float, help="First included time for porosity_time.")
    parser.add_argument("--time-end", type=float, help="Last included time for porosity_time.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--velocity-arrows", "--particle-velocity-arrows",
                        dest="velocity_directions", action="store_true",
                        help="Show in-plane particle velocity directions in exact 2D slices; colours show speed.")
    parser.add_argument("--no-velocity-colors", "--no-velocity-arrows",
                        dest="no_velocity_arrows", action="store_true",
                        help="Disable white-to-red particle speed colouring in exact 2D slice plots.")
    parser.add_argument("--alignment-mode", choices=("prolate", "oblate"), default="prolate",
                        help="Alignment to longest axis (prolate), or to c axis (oblate).")
    parser.add_argument("--alignment-min-speed", type=float, default=0.0,
                        help="Exclude speeds at or below this value from alignment angles.")
    parser.add_argument("--cluster-eps", type=float, default=0.3,
                        help="DBSCAN radius in standardized angle/linear-speed coordinates.")
    parser.add_argument("--cluster-min-samples", type=int, default=20)
    parser.add_argument("--fluid", nargs="?", const="magnitude",
                        type=str.lower, choices=("magnitude", "u", "v", "w", "pressure", "lambda2", "q"),
                        help="Overlay velocity, pressure, lambda2 or Q at exactly the particle time.")
    parser.add_argument("--fluid-grid", action="store_true",
                        help="Overlay actual fluid cell boundaries; defaults to --fluid magnitude.")
    parser.add_argument("--fluid-clim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="Fixed fluid colour scale for particle_fluid; use identical values across cases.")
    parser.add_argument("--continue-on-error", action="store_true")
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
    parser.add_argument("--alphashape", action="store_true")
    parser.add_argument("--upper-edge", "--upper_edge", dest="upper_edge", action="store_true")
    parser.add_argument("--upper-edge-bins", "--upper_edge_bins", dest="upper_edge_bins", type=int, default=50)
    parser.add_argument("--densify", type=float)
    parser.add_argument("--length-unit-mm", type=float,
                        help="Millimeters per input length unit for DIN particle_shape sieve curves (configured default: 0.615).")
    parser.add_argument("--skip-points", "--skip_points", "--skip-particles", "--skip_particles", dest="skip_points", type=int)
    parser.add_argument("--z", type=float)
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("XMIN", "XMAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
    if args.fluid_clim is not None:
        import math
        if "particle_fluid" not in selected:
            parser.error("--fluid-clim requires --only particle_fluid")
        if not all(math.isfinite(v) for v in args.fluid_clim) or args.fluid_clim[0] >= args.fluid_clim[1]:
            parser.error("--fluid-clim requires finite increasing MIN MAX")
    if "porosity_time" in selected:
        if args.counter is not None or args.time is not None:
            parser.error("porosity_time uses --time-start/--time-end, not --counter/--time")
        if args.porosity_frames < 0 or args.porosity_profiles < 1:
            parser.error("Require --porosity-frames >= 0 and --porosity-profiles >= 1")
        import math
        if not math.isfinite(args.lz) or not 0 < args.porosity_z_step < args.lz:
            parser.error("Require finite --lz > --porosity-z-step > 0")
        if any(t is not None and not math.isfinite(t) for t in (args.time_start, args.time_end)):
            parser.error("Time limits must be finite")
        if args.time_start is not None and args.time_end is not None and args.time_start > args.time_end:
            parser.error("--time-start must not exceed --time-end")
    elif args.time_start is not None or args.time_end is not None:
        parser.error("--time-start/--time-end require --only porosity_time")
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

    if args.fluid_grid:
        if not any(name in selected for name in ("2d", "slices", "particle_fluid")):
            parser.error("--fluid-grid requires 2d, slices, or particle_fluid")
        if args.fluid is None:
            args.fluid = "magnitude"
    if "particle_fluid" in selected and args.fluid is None:
        args.fluid = "magnitude"
    counter = args.counter
    if args.fluid and not any(name in selected for name in ("2d", "slices", "particle_fluid")):
        parser.error("--fluid requires 2d, slices, or particle_fluid")
    if args.time is not None and not any(name in selected for name in ("2d", "slices", "renderview", "alignment", "particle_fluid")):
        parser.error("--time requires 2d, slices, renderview, alignment, or particle_fluid")
    chosen_time = None
    fluid_counter = None
    try:
        from granular_analysis.io.fluid_slice import (
            latest_common_counter, matching_record, nearest_frame, restart_records,
        )
        from granular_analysis.io.loader import _read_bin_time
        if args.time is not None:
            chosen_time, counter = nearest_frame(case, args.time, args.fluid)
        elif counter is None and args.fluid:
            counter = latest_common_counter(case, args.fluid)
        if args.fluid:
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
