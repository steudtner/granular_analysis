#!/usr/bin/env python3
"""
Master runner for granular_analysis pipelines.

Supports:
    - standard simulation result folders
    - settling-matrix case roots
    - explicit results folders
    - nested ellipsoid / ellipsoid_trn data structures

Examples
--------
Standard case:

    python -m granular_analysis.scripts.run_all \
        --only 2d \
        --path /path/to/case/results

Settling-matrix case root:

    python -m granular_analysis.scripts.run_all \
        --only 2d \
        --path /path/to/case

Collision distances:

    python -m granular_analysis.scripts.run_all \
        --only collision_distances \
        --path /path/to/case

Exact slice:

    python -m granular_analysis.scripts.run_all \
        --only slices \
        --path /path/to/case

Full analysis:

    python -m granular_analysis.scripts.run_all \
        --path /path/to/case
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import sys
import traceback


# ============================================================================
# CONFIG
# ============================================================================

from granular_analysis.config import (
    FOLDER,
    OUTDIR_2D,
    RESULTS_DIR,
    ALPHA_2D,
    LZ,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
)


# ============================================================================
# PIPELINE IMPORTS
# ============================================================================

from granular_analysis.scripts.run_2d_analysis import main as run_2d
from granular_analysis.scripts.run_slice_comparison import main as run_slices
from granular_analysis.scripts.run_energy_analysis import main as run_energy
from granular_analysis.scripts.run_slope_analysis import main as run_slope
from granular_analysis.scripts.run_collision_distances import (
    main as run_collision_distances
)

from granular_analysis.plotting.slice_plotter import (
    plot_exact_slice_from_ellipsoids
)

from granular_analysis.scripts.run_slice_comparison import load_particles

from granular_analysis.io.loader import load_last_frame

from granular_analysis.plotting.particle_3d_plotter import (
    render_particle_views
)


# ============================================================================
# PATH RESOLUTION
# ============================================================================

def _resolve_data_folder(path: str | None = None) -> str:
    """
    Resolve the input data directory.

    Accepted input forms:

        <case>
        <case>/results
        <case>/results/ellipsoid
        <case>/results/ellipsoid/ellipsoid_trn

    The TRN directory is returned unchanged if supplied directly.
    """

    # ---------------------------------------------------------------
    # Select source
    # ---------------------------------------------------------------

    if path:
        candidate = path

    elif os.environ.get("GRANULAR_DATA_FOLDER"):
        candidate = os.environ["GRANULAR_DATA_FOLDER"]

    else:
        candidate = FOLDER

    p = pathlib.Path(
        candidate
    ).expanduser().resolve()

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # The configured default is already:
    #
    #   .../results/ellipsoid/ellipsoid_trn
    #
    # Do NOT append anything to it.
    # ---------------------------------------------------------------

    if p.name == "ellipsoid_trn":
        return str(p)

    # ---------------------------------------------------------------
    # Ellipsoid root
    # ---------------------------------------------------------------

    if p.name == "ellipsoid":
        return str(p)

    # ---------------------------------------------------------------
    # Results directory
    # ---------------------------------------------------------------

    if p.name == "results":
        return str(p)

    # ---------------------------------------------------------------
    # Case root
    # ---------------------------------------------------------------

    results_dir = p / "results"

    if results_dir.is_dir():
        return str(results_dir)

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    return str(p)
def _resolve_ellipsoid_root(data_folder: str) -> str:
    """
    Resolve the ellipsoid root directory.

    Accepted input:

        .../results
        .../results/ellipsoid
        .../results/ellipsoid/ellipsoid_trn
    """

    p = pathlib.Path(
        data_folder
    ).expanduser().resolve()

    # Already ellipsoid root
    if p.name == "ellipsoid":
        return str(p)

    # Already TRN directory
    if p.name == "ellipsoid_trn":
        return str(p.parent)

    # Standard results directory
    candidate = p / "ellipsoid"

    if candidate.is_dir():
        return str(candidate)

    return str(candidate)


def _resolve_trn_folder(data_folder: str) -> str:
    """
    Resolve the directory containing ell_trn_XXXXXX.bin files.

    Expected structure:

        .../results/ellipsoid/ellipsoid_trn/
    """

    p = pathlib.Path(
        data_folder
    ).expanduser().resolve()

    # ---------------------------------------------------------------
    # Already the TRN directory
    # ---------------------------------------------------------------

    if p.name == "ellipsoid_trn":
        return str(p)

    # ---------------------------------------------------------------
    # Supplied ellipsoid root
    # ---------------------------------------------------------------

    if p.name == "ellipsoid":
        trn_dir = p / "ellipsoid_trn"

        if trn_dir.is_dir():
            return str(trn_dir)

        return str(trn_dir)

    # ---------------------------------------------------------------
    # Standard results/ellipsoid/ellipsoid_trn structure
    # ---------------------------------------------------------------

    trn_dir = p / "ellipsoid" / "ellipsoid_trn"

    if trn_dir.is_dir():
        return str(trn_dir)

    # ---------------------------------------------------------------
    # Supplied case/results directory
    # ---------------------------------------------------------------

    trn_dir = p / "ellipsoid" / "ellipsoid_trn"

    return str(trn_dir)

# ============================================================================
# COUNTER RESOLUTION
# ============================================================================

def _extract_counter_from_filename(filename: str) -> int | None:
    """
    Extract the final six-digit counter from a filename.

    Examples
    --------
    ell_trn_000123.bin
        -> 123

    ell_coll_000456.bin
        -> 456

    collision_000789.dat
        -> 789
    """

    name = os.path.basename(filename)

    matches = re.findall(r"(\d{6})(?:\.[^.]+)?$", name)

    if matches:
        return int(matches[-1])

    # More general fallback
    matches = re.findall(r"(\d+)", name)

    if matches:
        return int(matches[-1])

    return None


def _resolve_latest_trn_counter(data_folder: str) -> int | None:
    """
    Find the newest TRN counter.

    Searches the nested settling-matrix structure:

        results/ellipsoid/ellipsoid_trn/ell_trn_XXXXXX.bin

    and also several backwards-compatible locations.
    """

    trn_folder = pathlib.Path(
        _resolve_trn_folder(data_folder)
    )

    candidates = []

    if trn_folder.is_dir():
        candidates.extend(
            trn_folder.glob("ell_trn_*.bin")
        )

    # Backwards-compatible alternatives
    p = pathlib.Path(data_folder).expanduser().resolve()

    search_patterns = [
        p / "ellipsoid_trn" / "ell_trn_*.bin",
        p / "ellipsoid" / "ellipsoid_trn" / "ell_trn_*.bin",
        p / "ellipsoid" / "ell_trn_*.bin",
        p / "ell_trn_*.bin",
    ]

    for pattern in search_patterns:
        candidates.extend(
            pathlib.Path(x)
            for x in glob.glob(str(pattern))
        )

    # Remove duplicates
    unique = {}

    for candidate in candidates:
        unique[str(candidate.resolve())] = candidate

    candidates = list(unique.values())

    if not candidates:
        return None

    # Prefer highest counter rather than filesystem order.
    counters = []

    for candidate in candidates:
        counter = _extract_counter_from_filename(
            candidate.name
        )

        if counter is not None:
            counters.append(counter)

    if not counters:
        return None

    return max(counters)


def _resolve_latest_collision_counter(data_folder: str) -> int | None:
    """
    Find the newest collision counter.

    Searches recursively because settling-matrix cases may have:

        results/
            ellipsoid/
                ellipsoid_coll/
                ellipsoid_trn/

    or other collision-related output locations.

    If no collision-specific file can be found, the newest TRN counter
    is used as fallback because collision_distances requires the same
    simulation frame for TRN and COLL.
    """

    p = pathlib.Path(data_folder).expanduser().resolve()

    patterns = [
        "*coll*.bin",
        "*coll*.dat",
        "*collision*.bin",
        "*collision*.dat",
        "*counter*.bin",
        "*counter*.dat",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(p.rglob(pattern))

    # Also explicitly search common COLL directory
    explicit_dirs = [
        p / "ellipsoid" / "ellipsoid_coll",
        p / "ellipsoid_coll",
        p / "ellipsoid" / "coll",
        p / "coll",
    ]

    for directory in explicit_dirs:
        if directory.is_dir():
            for item in directory.iterdir():
                if item.is_file():
                    candidates.append(item)

    # Remove duplicates
    unique = {}

    for candidate in candidates:
        unique[str(candidate.resolve())] = candidate

    candidates = list(unique.values())

    counters = []

    for candidate in candidates:
        counter = _extract_counter_from_filename(
            candidate.name
        )

        if counter is not None:
            counters.append(counter)

    if counters:
        return max(counters)

    # ------------------------------------------------------------------
    # Fallback:
    # collision_distances requires TRN and COLL at the same counter.
    # If no separately named collision file can be found, use latest TRN.
    # ------------------------------------------------------------------

    return _resolve_latest_trn_counter(data_folder)


# ============================================================================
# EXACT SLICE
# ============================================================================

def _run_exact_slice_plot(
    args,
    data_folder: str,
    out_dir: pathlib.Path,
):
    """
    Generate the exact ellipsoid slice plot.

    The plot can optionally contain:
        - alpha shape
        - upper edge
    """

    print("=" * 60)
    print("Running: exact slice plot")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Resolve actual TRN directory
    # ---------------------------------------------------------------

    trn_path = pathlib.Path(
        _resolve_trn_folder(data_folder)
    ).resolve()

    print("Loading particles from:")
    print(f"  {trn_path}")

    if not trn_path.is_dir():
        raise FileNotFoundError(
            f"TRN directory not found:\n{trn_path}"
        )

    bin_files = list(trn_path.glob("*.bin"))

    if not bin_files:
        raise FileNotFoundError(
            f"No .bin files found in:\n{trn_path}"
        )

    # ---------------------------------------------------------------
    # Load particles
    # ---------------------------------------------------------------

    ellipsoids = load_last_frame(trn_path)

    print(f"Loaded {len(ellipsoids)} particles.")

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------

    slice_out = out_dir / "2d"

    slice_out.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------

    z = (
        args.z
        if args.z is not None
        else LZ / 2
    )

    alpha = (
        args.alpha
        if args.alpha is not None
        else ALPHA_2D
    )

    upper_edge_bins = (
        args.upper_edge_bins
        if args.upper_edge_bins is not None
        else 50
    )

    skip_points = (
        args.skip_points
        if args.skip_points is not None
        else (
            SKIP_FIRST_PARTICLES
            if USE_PARTICLE_SKIP
            else 0
        )
    )

    xlim = (
        tuple(args.xlim)
        if args.xlim is not None
        else (0.0, 60.0)
    )

    ylim = (
        tuple(args.ylim)
        if args.ylim is not None
        else (0.0, 20.0)
    )

    # ---------------------------------------------------------------
    # Plot options
    # ---------------------------------------------------------------

    use_alphashape = args.alphashape
    use_upper_edge = args.upper_edge

    # Upper edge requires an alpha shape.
    if use_upper_edge:
        use_alphashape = True

    print()
    print("Exact slice parameters:")
    print(f"  z               = {z}")
    print(f"  alpha           = {alpha}")
    print(f"  alphashape      = {use_alphashape}")
    print(f"  upper_edge      = {use_upper_edge}")
    print(f"  upper_edge_bins = {upper_edge_bins}")
    print(f"  skip_points     = {skip_points}")
    print(f"  xlim            = {xlim}")
    print(f"  ylim            = {ylim}")
    print()

    # ---------------------------------------------------------------
    # Output filename
    # ---------------------------------------------------------------

    suffix = (
        "_upperEdge"
        if use_upper_edge
        else ""
    )

    output_path = (
        slice_out
        / f"alphashape_slice_{z:.2f}{suffix}.svg"
    )

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------

    result = plot_exact_slice_from_ellipsoids(
        ellipsoids,
        z_plane=z,
        output_path=output_path,
        alphashape=use_alphashape,
        alpha=alpha,
        upper_edge=use_upper_edge,
        upper_edge_bins=upper_edge_bins,
        densify_max_segment_length=(
            args.densify
            if args.densify
            else None
        ),
        skip_points=skip_points,
        xlim=xlim,
        ylim=ylim,
    )

    print()
    print(
        f"Saved exact slice plot to:\n"
        f"  {result['output_path']}"
    )

    return result

# ============================================================================
# RENDERVIEW
# ============================================================================

def _resolve_latest_bin_path(data_folder: str) -> str | None:
    """
    Find the newest TRN binary file.
    """

    trn_folder = pathlib.Path(
        _resolve_trn_folder(data_folder)
    )

    if not trn_folder.is_dir():
        return None

    files = list(
        trn_folder.glob("ell_trn_*.bin")
    )

    if not files:
        return None

    files.sort(
        key=lambda p: (
            _extract_counter_from_filename(p.name)
            if _extract_counter_from_filename(p.name) is not None
            else -1
        )
    )

    return str(files[-1])


def run_renderview(
    counter=None,
    path=None,
    out=None,
    force=False,
    **kwargs,
):
    """
    Render 3D particle views for one TRN frame.
    """

    data_folder = _resolve_data_folder(path)

    if counter is None:
        counter = _resolve_latest_trn_counter(
            data_folder
        )

    if counter is None:
        raise RuntimeError(
            "Could not determine TRN counter."
        )

    out_dir = pathlib.Path(
        out if out else RESULTS_DIR
    )

    render_dir = out_dir / "renderview"

    render_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Expected output files
    # ---------------------------------------------------------------

    expected = [
        render_dir / f"particles_{counter:06d}_1.png",
        render_dir / f"particles_{counter:06d}_2.png",
        render_dir / f"particles_{counter:06d}_3.png",
        render_dir / f"particles_{counter:06d}_4.png",
    ]

    if (
        not force
        and all(p.exists() for p in expected)
    ):
        print(
            f"Renderview for counter "
            f"{counter:06d} already exists. Skipping."
        )
        return

    # ---------------------------------------------------------------
    # load_data()
    #
    # read_ellipsoid_data.load_data(data="trn", ...)
    # expects:
    #
    #   <path>/ellipsoid_trn/ell_trn_XXXXXX.bin
    #
    # Therefore path must be the ellipsoid root.
    # ---------------------------------------------------------------

    load_path = _resolve_ellipsoid_root(
        data_folder
    )

    print("Renderview paths:")
    print(f"  DATA ROOT = {data_folder}")
    print(f"  LOAD PATH = {load_path}")
    print(f"  COUNTER   = {counter}")
    print(f"  OUTPUT    = {render_dir}")

    particles = load_data(
        data="trn",
        counter=counter,
        path=load_path,
    )

    render_particle_views(
        particles,
        output_dir=str(render_dir),
        counter=counter,
    )


# ============================================================================
# PIPELINES
# ============================================================================

PIPELINES = {
    "2d": run_2d,
    "slices": run_slices,
    "energy": run_energy,
    "slope": run_slope,
    "collision_distances": run_collision_distances,
    "renderview": run_renderview,
}


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_pipeline(
    name,
    func,
    force=False,
    kwargs=None,
):
    """
    Execute one pipeline.

    Some older pipeline functions do not accept force=.
    In that case, retry without force.
    """

    kwargs = kwargs or {}

    print()
    print("=" * 60)
    print(f"Running: {name}")
    print("=" * 60)

    try:
        return func(
            force=force,
            **kwargs,
        )

    except TypeError as exc:
        message = str(exc)

        if (
            "unexpected keyword argument 'force'"
            not in message
        ):
            raise

        print(
            f"[INFO] Pipeline '{name}' does not accept "
            f"force=. Retrying without force."
        )

        return func(**kwargs)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run granular_analysis pipelines."
        )
    )

    # ---------------------------------------------------------------
    # Pipeline selection
    # ---------------------------------------------------------------

    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(PIPELINES.keys()),
        default=None,
        help="Run only the specified pipelines.",
    )

    parser.add_argument(
        "--exclude",
        nargs="+",
        choices=list(PIPELINES.keys()),
        default=[],
        help="Exclude specified pipelines.",
    )

    # ---------------------------------------------------------------
    # General execution
    # ---------------------------------------------------------------

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation / overwrite cached outputs.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with remaining pipelines after an error.",
    )

    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print planned execution without running pipelines.",
    )

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------

    parser.add_argument(
        "--path",
        default=None,
        help=(
            "Simulation case root or results directory. "
            "A case root containing results/ is resolved automatically."
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output directory.",
    )

    parser.add_argument(
        "--folder",
        default=None,
        help="Alias for overriding the input data folder.",
    )

    # ---------------------------------------------------------------
    # Frame / counter
    # ---------------------------------------------------------------

    parser.add_argument(
        "--counter",
        type=int,
        default=None,
        help="Explicit simulation counter.",
    )

    # ---------------------------------------------------------------
    # Alpha shape / slices
    # ---------------------------------------------------------------

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha-shape parameter.",
    )

    parser.add_argument(
        "--alphashape",
        action="store_true",
        help="Enable alpha-shape analysis where supported.",
    )

    parser.add_argument(
        "--upper_edge",
        action="store_true",
        help="Enable upper-edge analysis.",
    )

    parser.add_argument(
        "--upper_edge_bins",
        type=int,
        default=None,
        help="Number of bins for upper-edge analysis.",
    )

    parser.add_argument(
        "--densify",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--skip_points",
        type=int,
        default=None,
        help="Number of initial points to skip.",
    )

    # ---------------------------------------------------------------
    # Slice geometry
    # ---------------------------------------------------------------

    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Slice z-coordinate.",
    )

    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="x-axis limits.",
    )

    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="y-axis limits.",
    )

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ========================================================================
    # RESOLVE PATHS
    # ========================================================================

    data_folder = _resolve_data_folder(
        args.path
    )

    # --folder explicitly overrides --path
    if args.folder:
        data_folder = _resolve_data_folder(
            args.folder
        )

    out_dir = pathlib.Path(
        args.out
        if args.out
        else RESULTS_DIR
    ).expanduser().resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================================
    # PRINT PATHS
    # ========================================================================

    print()
    print("=" * 60)
    print("PATHS")
    print("=" * 60)

    print(f"DATA    = {data_folder}")
    print(f"RESULTS = {out_dir}")

    print("=" * 60)
    print()

    # ========================================================================
    # SELECT PIPELINES
    # ========================================================================

    if args.only:
        selected = list(args.only)
    else:
        selected = list(PIPELINES.keys())

    selected = [
        name
        for name in selected
        if name not in args.exclude
    ]

    # Validate
    unknown = [
        name
        for name in selected
        if name not in PIPELINES
    ]

    if unknown:
        parser.error(
            f"Unknown pipeline(s): {unknown}"
        )

    if not selected:
        print("No pipelines selected.")
        return

    print("Selected pipelines:")
    for name in selected:
        print(f"  - {name}")

    print()

    # ========================================================================
    # COUNTER
    # ========================================================================

    counter = args.counter

    if counter is None:

        # Renderview needs TRN
        if "renderview" in selected:
            counter = _resolve_latest_trn_counter(
                data_folder
            )

        # Collision distances need a frame for TRN + COLL
        elif "collision_distances" in selected:
            counter = _resolve_latest_collision_counter(
                data_folder
            )

        # Slices / slope use a simulation frame as well
        elif (
            "slices" in selected
            or "slope" in selected
        ):
            counter = _resolve_latest_trn_counter(
                data_folder
            )

    if counter is not None:
        print(
            f"COUNTER = {counter:06d}"
        )
    else:
        print(
            "COUNTER = None "
            "(pipeline may determine its own counter)"
        )

    print()

    # ========================================================================
    # COMMON KWARGS
    # ========================================================================

    common_kwargs = {
        "out": str(out_dir),
    }

    # ========================================================================
    # RUN PIPELINES
    # ========================================================================

    errors = []

    for name in selected:

        func = PIPELINES[name]

        kwargs = dict(
            common_kwargs
        )

        # --------------------------------------------------------------------
        # 2D
        # --------------------------------------------------------------------

        if name == "2d":

            kwargs["path"] = data_folder

            if args.z is not None:
                kwargs["z_value"] = args.z

            if args.densify:
                kwargs["densify"] = args.densify

            if args.upper_edge_bins is not None:
                kwargs["upper_edge_bins"] = (
                    args.upper_edge_bins
                )

        # --------------------------------------------------------------------
        # SLICES
        # --------------------------------------------------------------------

        elif name == "slices":

            kwargs["path"] = data_folder

            if counter is not None:
                kwargs["counter"] = counter

            if args.z is not None:
                kwargs["z"] = args.z

            if args.alpha is not None:
                kwargs["alpha"] = args.alpha

            if args.skip_points is not None:
                kwargs["skip_points"] = (
                    args.skip_points
                )

        # --------------------------------------------------------------------
        # ENERGY
        # --------------------------------------------------------------------

        elif name == "energy":

            kwargs["out"] = str(
                out_dir / "energy.csv"
            )

        # --------------------------------------------------------------------
        # SLOPE
        # --------------------------------------------------------------------

        elif name == "slope":

            kwargs["path"] = data_folder

            if counter is not None:
                kwargs["counter"] = counter

        # --------------------------------------------------------------------
        # COLLISION DISTANCES
        # --------------------------------------------------------------------

        elif name == "collision_distances":

            # IMPORTANT:
            #
            # run_collision_distances.py eventually calls:
            #
            #   load_data(
            #       data="trn",
            #       path=path
            #   )
            #
            # read_ellipsoid_data expects:
            #
            #   <path>/ellipsoid_trn/ell_trn_XXXXXX.bin
            #
            # Therefore path must be:
            #
            #   .../results/ellipsoid
            #
            kwargs["path"] = _resolve_ellipsoid_root(
                data_folder
            )

            if counter is not None:
                kwargs["counter"] = counter

        # --------------------------------------------------------------------
        # RENDERVIEW
        # --------------------------------------------------------------------

        elif name == "renderview":

            kwargs["path"] = data_folder

            if counter is not None:
                kwargs["counter"] = counter

        # ====================================================================
        # DRY RUN
        # ====================================================================

        if args.dry:

            print("=" * 60)
            print(f"DRY RUN: {name}")
            print("=" * 60)

            print(f"FUNCTION = {func}")
            print(f"FORCE    = {args.force}")
            print("KWARGS:")

            for key, value in kwargs.items():
                print(
                    f"  {key} = {value}"
                )

            print()

            continue

        # ====================================================================
        # EXECUTE
        # ====================================================================

        try:

            run_pipeline(
                name=name,
                func=func,
                force=args.force,
                kwargs=kwargs,
            )

            print()
            print(
                f"[OK] Pipeline '{name}' completed."
            )

        except Exception as exc:

            print()
            print(
                f"[ERROR] Pipeline '{name}' failed:"
            )
            print(
                f"  {type(exc).__name__}: {exc}"
            )

            traceback.print_exc()

            errors.append(
                (name, exc)
            )

            if not args.continue_on_error:
                print()
                print(
                    "Stopping because "
                    "--continue-on-error was not specified."
                )
                break

    # ========================================================================
    # EXACT SLICE PLOT
    # ========================================================================
    #
    # IMPORTANT:
    #
    # --only 2d
    #     -> ONLY 2D, no exact slice plot.
    #
    # --only slices
    #     -> slices pipeline + exact slice plot.
    #
    # full run
    #     -> exact slice plot after all selected pipelines.
    #
    # This is orchestration only. No individual pipeline is modified.
    # ========================================================================

    run_exact_slice = False

    if not args.dry:

            # Explicit slices selection
        if (
            args.only is not None
            and (
                "slices" in selected
                or (
                    "2d" in selected
                    and (
                        args.alphashape
                        or args.upper_edge
                    )
                )
            )
        ):
            run_exact_slice = True

    elif args.only is None:
        run_exact_slice = True

    if run_exact_slice:

        try:

            _run_exact_slice_plot(
                args=args,
                data_folder=data_folder,
                out_dir=out_dir,
            )

            print()
            print(
                "[OK] Exact slice plot completed."
            )

        except Exception as exc:

            print()
            print(
                "[WARNING] Exact slice plot failed:"
            )
            print(
                f"  {type(exc).__name__}: {exc}"
            )

            traceback.print_exc()

            errors.append(
                ("exact_slice_plot", exc)
            )

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if errors:

        print(
            f"Failed steps: {len(errors)}"
        )

        for name, exc in errors:
            print(
                f"  [FAILED] {name}: "
                f"{type(exc).__name__}: {exc}"
            )

    else:

        print(
            "All requested steps completed successfully."
        )

    print("=" * 60)
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()