from __future__ import annotations

import argparse
import glob
import os
import pathlib
import re
import sys
import traceback


# ============================================================
# PATH RESOLUTION
# ============================================================

def _resolve_data_folder(path: str | None = None) -> str:
    """
    Resolve the data folder used by the analysis.

    Priority:
        1. Explicit --path
        2. Environment variable GRANULAR_ANALYSIS_FOLDER
        3. Configured FOLDER
    """
    if path:
        return str(pathlib.Path(path).expanduser().resolve())

    env_path = os.environ.get("GRANULAR_ANALYSIS_FOLDER")
    if env_path:
        return str(pathlib.Path(env_path).expanduser().resolve())

    from granular_analysis.config import FOLDER

    return str(pathlib.Path(FOLDER).expanduser().resolve())


def _resolve_latest_collision_counter(
    path: str | None = None,
) -> int | None:
    """
    Try to determine the latest collision counter from filenames.

    Returns None if no counter can be determined.
    """

    folder = pathlib.Path(path or ".").expanduser()

    if not folder.exists():
        return None

    candidates = []

    patterns = [
        "*collision*",
        "*coll*",
        "*counter*",
    ]

    for pattern in patterns:
        candidates.extend(folder.glob(pattern))

    numbers = []

    for item in candidates:
        match = re.findall(r"\d+", item.name)

        if not match:
            continue

        for value in match:
            try:
                numbers.append(int(value))
            except ValueError:
                pass

    if not numbers:
        return None

    return max(numbers)


def _resolve_latest_trn_counter(
    path: str | None = None,
) -> int | None:
    """
    Determine the latest available TRN timestep
    from ell_trn_XXXXXX.bin files.
    """

    folder = pathlib.Path(path or ".").expanduser()

    if not folder.exists():
        return None

    counters = []

    for item in folder.glob("ell_trn_*.bin"):

        match = re.search(
            r"ell_trn_(\d+)\.bin$",
            item.name,
        )

        if match:
            counters.append(
                int(match.group(1))
            )

    if not counters:
        return None

    return max(counters)

# ============================================================
# CONFIG
# ============================================================

from granular_analysis.config import (
    FOLDER,
    OUTDIR_2D,
    RESULTS_DIR,
)


# ============================================================
# PIPELINE IMPORTS
# ============================================================

from granular_analysis.scripts.run_2d_analysis import (
    main as run_2d,
)

from granular_analysis.scripts.run_slice_comparison import (
    main as run_slices,
)

from granular_analysis.scripts.run_energy_analysis import (
    main as run_energy,
)

from granular_analysis.scripts.run_slope_analysis import (
    main as run_slope,
)

from granular_analysis.scripts.run_collision_distances import (
    main as run_collision_distances,
)

from granular_analysis.plotting.slice_plotter import (
    plot_exact_slice_from_ellipsoids,
)

from granular_analysis.scripts.run_slice_comparison import (
    load_particles,
)

from granular_analysis.io.read_ellipsoid_data import (
    load_data,
)

from granular_analysis.plotting.particle_3d_plotter import (
    render_particle_views,
)


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_pipeline(
    name: str,
    func,
    force: bool = False,
    **kwargs,
):
    """
    Run one analysis pipeline.

    Some older pipeline functions may not accept force=.
    Therefore a fallback without force is provided.
    """

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

        if "unexpected keyword argument 'force'" not in message:
            raise

        print(
            f"[INFO] Pipeline '{name}' does not accept force=. "
            f"Retrying without force."
        )

        return func(**kwargs)


# ============================================================
# BIN FILE
# ============================================================

def _resolve_latest_bin_path(
    folder: str | pathlib.Path,
):
    """
    Find the latest .bin file in a folder.

    Returns None if no .bin file exists.
    """

    folder = pathlib.Path(folder)

    if not folder.exists():
        return None

    files = list(folder.glob("*.bin"))

    if not files:
        return None

    def sort_key(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0

    files.sort(
        key=sort_key
    )

    return files[-1]


# ============================================================
# EXACT SLICE PLOT
# ============================================================

def _run_exact_slice_plot(
    args,
    out_dir,
):
    """
    Run the exact ellipsoid slice plot.

    Uses the existing slice_plotter.py implementation to draw:
        - all intersected ellipses
        - optional alpha-shape
        - optional upper edge
    """

    print("=" * 60)
    print("Running: exact slice plot")
    print("=" * 60)

    try:

        # --------------------------------------------------------
        # Input path
        # --------------------------------------------------------

        path = (
            args.path
            or args.folder
            or FOLDER
        )

        # --------------------------------------------------------
        # Slice position
        # --------------------------------------------------------

        z_value = (
            args.z
            if args.z is not None
            else 2.5
        )

        # --------------------------------------------------------
        # Load particles
        # --------------------------------------------------------

        print(
            "Loading particles from:"
        )

        print(
            f"  {path}"
        )

        print()

        ellipsoids = load_particles(
            path
        )

        print(
            f"Loaded {len(ellipsoids)} particles."
        )

        print()

        # --------------------------------------------------------
        # Output directory
        #
        # Keep the exact slice plots together with the 2D results.
        # --------------------------------------------------------

        slice_out_dir = pathlib.Path(
            out_dir
        ) / "2d"

        slice_out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            slice_out_dir
            / f"alphashape_slice_{z_value:.2f}.svg"
        )

        # --------------------------------------------------------
        # Optional parameters
        # --------------------------------------------------------

        alpha = (
            args.alpha
            if args.alpha is not None
            else 0.6
        )

        upper_edge_bins = (
            args.upper_edge_bins
            if args.upper_edge_bins is not None
            else 50
        )

        skip_points = (
            args.skip_points
            if args.skip_points is not None
            else 0
        )

        densify = (
            args.densify
            if args.densify is not None
            else None
        )

        xlim = (
            tuple(args.xlim)
            if args.xlim is not None
            else (0, 60)
        )

        ylim = (
            tuple(args.ylim)
            if args.ylim is not None
            else (0, 20)
        )

        # --------------------------------------------------------
        # Run existing plotter
        # --------------------------------------------------------

        result = plot_exact_slice_from_ellipsoids(
            ellipsoids,
            z_plane=z_value,
            output_path=output_path,
            alphashape=args.alphashape,
            alpha=alpha,
            upper_edge=args.upper_edge,
            upper_edge_bins=upper_edge_bins,
            densify_max_segment_length=densify,
            skip_points=skip_points,
            xlim=xlim,
            ylim=ylim,
        )

        # --------------------------------------------------------
        # Report
        # --------------------------------------------------------

        print()

        print(
            "Exact slice plot completed successfully."
        )

        print()

        print(
            f"  Output      = {result['output_path']}"
        )

        print(
            f"  z           = {z_value}"
        )

        print(
            f"  Ellipses    = {result['num_ellipses']}"
        )

        print(
            f"  Alpha pts   = {result['num_alpha_points']}"
        )

        print(
            f"  Area        = {result['area']}"
        )

        print()

    except Exception as exc:

        print(
            f"[WARNING] Exact slice plot failed: {exc}"
        )

        if args.continue_on_error:
            traceback.print_exc()
        else:
            raise


# ============================================================
# RENDER VIEW
# ============================================================

def run_renderview(
    path,
    out,
    counter=None,
    force=False,
):
    """
    Render 3D particle views from an ellipsoid TRN frame.
    """

    print("=" * 60)
    print("3D PARTICLE RENDER")
    print("=" * 60)

    # --------------------------------------------------------
    # Counter
    #
    # The counter is resolved centrally in main(), exactly
    # like for the other pipelines.
    # --------------------------------------------------------

    if counter is None:
        raise ValueError(
            "No timestep/counter available for renderview."
        )

    print(
        f"TRN counter = {counter:06d}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        pathlib.Path(out)
        / "renderview"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Cache check
    # --------------------------------------------------------

    expected_files = [
        output_dir / "particles_3d_isometric.png",
        output_dir / "particles_3d_top.png",
        output_dir / "particles_3d_front.png",
        output_dir / "particles_3d_side.png",
    ]

    if (
        not force
        and all(
            file.exists()
            for file in expected_files
        )
    ):

        print(
            "[INFO] Rendered views already exist. "
            "Use --force to regenerate."
        )

        for file in expected_files:
            print(
                f"  {file}"
            )

        print()

        return {
            "counter": counter,
            "output_dir": output_dir,
            "files": expected_files,
        }

    # --------------------------------------------------------
    # Load ellipsoid TRN data
    # --------------------------------------------------------

    print(
        f"Loading ellipsoid TRN frame "
        f"{counter:06d} ..."
    )

    load_path = pathlib.Path(path)

    if load_path.name == "ellipsoid_trn":
        load_path = load_path.parent

    ellipsoids = load_data(
        data="trn",
        counter=counter,
        path=str(load_path),
    )

    if not ellipsoids:
        raise RuntimeError(
            f"TRN frame {counter:06d} contains "
            "no ellipsoids."
        )

    print(
        f"Particles = {len(ellipsoids)}"
    )

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    print()
    print(
        "Rendering views ..."
    )

    rendered = render_particle_views(
        ellipsoids=ellipsoids,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        "Rendered files:"
    )

    for view, filename in rendered.items():

        print(
            f"  {view:10s}: {filename}"
        )

    print()

    return {
        "counter": counter,
        "output_dir": output_dir,
        "files": rendered,
    }


# ============================================================
# PIPELINES
# ============================================================

PIPELINES = {
    "2d": run_2d,
    "slices": run_slices,
    "energy": run_energy,
    "slope": run_slope,
    "collision_distances": run_collision_distances,
    "renderview": run_renderview,
}


# ============================================================
# MAIN
# ============================================================

def main(args=None) -> None:
    """
    Main orchestrator.

    Important:
        argparse is handled here only when args is None.

    This allows run_all.py to be called normally from the shell
    while keeping all command-line arguments available through
    the args namespace.
    """

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    if args is None:
        args = build_parser().parse_args()

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    data_folder = _resolve_data_folder(
        args.path
    )

    if args.out:

        out_dir = (
            pathlib.Path(args.out)
            .expanduser()
            .resolve()
        )

    else:

        out_dir = (
            pathlib.Path(RESULTS_DIR)
            .expanduser()
            .resolve()
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Apply explicit folder override
    # --------------------------------------------------------

    if args.folder:

        data_folder = str(
            pathlib.Path(args.folder)
            .expanduser()
            .resolve()
        )

    # --------------------------------------------------------
    # Print paths
    # --------------------------------------------------------

    print("=" * 60)
    print("PATHS")
    print("=" * 60)

    print(
        f"DATA    = {data_folder}"
    )

    print(
        f"RESULTS = {out_dir}"
    )

    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry:

        print(
            "DRY RUN"
        )

        print()

    # --------------------------------------------------------
    # Determine selected pipelines
    # --------------------------------------------------------

    if args.only:

        selected = list(
            args.only
        )

    else:

        selected = list(
            PIPELINES.keys()
        )

    # --------------------------------------------------------
    # Exclude pipelines
    # --------------------------------------------------------

    if args.exclude:

        selected = [
            name
            for name in selected
            if name not in args.exclude
        ]

    # --------------------------------------------------------
    # Validate pipeline names
    # --------------------------------------------------------

    unknown = [
        name
        for name in selected
        if name not in PIPELINES
    ]

    if unknown:

        raise ValueError(
            "Unknown pipeline(s): "
            + ", ".join(unknown)
            + "\nAvailable pipelines: "
            + ", ".join(PIPELINES.keys())
        )

    # --------------------------------------------------------
    # Collision counter
    # --------------------------------------------------------

    counter = args.counter

    if counter is None:

        if "renderview" in selected:
            counter = _resolve_latest_trn_counter(
                data_folder
            )

            if counter is not None:
                print(f"TRN counter = {counter}")
                print()

        else:
            counter = _resolve_latest_collision_counter(
                data_folder
            )

            if counter is not None:
                print(f"Collision counter = {counter}")
                print()

    # --------------------------------------------------------
    # Common kwargs
    # --------------------------------------------------------

    common_kwargs = {
        "out": str(out_dir),
    }

    # --------------------------------------------------------
    # Run pipelines
    # --------------------------------------------------------

    errors = []

    for name in selected:

        func = PIPELINES[name]

        # ------------------------------------------------
        # Pipeline-specific arguments
        # ------------------------------------------------

        kwargs = dict(
            common_kwargs
        )

        # ------------------------------------------------
        # 2D
        # ------------------------------------------------

        if name == "2d":

            kwargs.update(
                {
                    "path": data_folder,
                }
            )

            if args.z is not None:

                kwargs["z_value"] = args.z

            if args.densify is not None:

                kwargs["densify"] = args.densify

            if args.upper_edge_bins is not None:

                kwargs["upper_edge_bins"] = (
                    args.upper_edge_bins
                )

        # ------------------------------------------------
        # Slices
        # ------------------------------------------------

        elif name == "slices":

            kwargs.update(
                {
                    "path": data_folder,
                }
            )

            if counter is not None:

                kwargs["counter"] = counter

        # ------------------------------------------------
        # Energy
        # ------------------------------------------------

        elif name == "energy":

            kwargs.update(
                {
                    "out": str(
                        out_dir / "energy.csv"
                    ),
                }
            )

        # ------------------------------------------------
        # Slope
        # ------------------------------------------------

        elif name == "slope":

            kwargs.update(
                {
                    "path": data_folder,
                }
            )

            if counter is not None:

                kwargs["counter"] = counter

        # ------------------------------------------------
        # Collision distances
        # ------------------------------------------------

        elif name == "collision_distances":

            kwargs.update(
                {
                    "path": data_folder,
                }
            )

            if counter is not None:

                kwargs["counter"] = counter

        # ------------------------------------------------
        # Render view
        # ------------------------------------------------

        elif name == "renderview":

            kwargs.update(
                {
                    "path": data_folder,
                }
            )

            if counter is not None:

                kwargs["counter"] = counter

        # ------------------------------------------------
        # Dry run
        # ------------------------------------------------

        if args.dry:

            print("=" * 60)
            print(
                f"DRY RUN: {name}"
            )
            print("=" * 60)

            print(
                f"Function : {func}"
            )

            print(
                f"force    : {args.force}"
            )

            print(
                f"kwargs   : {kwargs}"
            )

            print()

            continue

        # ------------------------------------------------
        # Execute
        # ------------------------------------------------

        try:

            run_pipeline(
                name,
                func,
                force=args.force,
                **kwargs,
            )

            print()
            print(
                f"[OK] Pipeline '{name}' completed."
            )
            print()

        except Exception as exc:

            print()
            print(
                f"[ERROR] Pipeline '{name}' failed:"
            )

            print(
                exc
            )

            print()

            errors.append(
                (name, exc)
            )

            if not args.continue_on_error:

                raise

            traceback.print_exc()

    # --------------------------------------------------------
    # Exact slice plot
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This must NOT run for:
    #
    #     --only energy
    #
    # because the exact slice plot is unrelated to the energy
    # pipeline and requires slice-specific arguments.
    #
    # --------------------------------------------------------

    if (
        not args.dry
        and (
            args.only is None
            or "2d" in selected
            or "slices" in selected
        )
    ):

        _run_exact_slice_plot(
            args,
            out_dir,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if errors:

        print(
            f"{len(errors)} pipeline(s) failed:"
        )

        for name, exc in errors:

            print(
                f"  - {name}: {exc}"
            )

    else:

        print(
            "All selected pipelines completed successfully."
        )

    print("=" * 60)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run granular collapse post-processing "
            "pipelines."
        )
    )

    # --------------------------------------------------------
    # Pipeline selection
    # --------------------------------------------------------

    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(
            PIPELINES.keys()
        ),
        default=None,
        help=(
            "Run only the selected pipeline(s). "
            "Example: --only energy"
        ),
    )

    parser.add_argument(
        "--exclude",
        nargs="+",
        choices=list(
            PIPELINES.keys()
        ),
        default=None,
        help=(
            "Exclude selected pipeline(s)."
        ),
    )

    # --------------------------------------------------------
    # General execution
    # --------------------------------------------------------

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Force recomputation even if cached results exist."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue with the remaining pipelines if one "
            "pipeline fails."
        ),
    )

    parser.add_argument(
        "--dry",
        action="store_true",
        help=(
            "Print what would be executed without running "
            "the analysis."
        ),
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "Input data folder."
        ),
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output directory."
        ),
    )

    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help=(
            "Alternative input folder override."
        ),
    )

    # --------------------------------------------------------
    # Collision counter
    # --------------------------------------------------------

    parser.add_argument(
        "--counter",
        type=int,
        default=None,
        help=(
            "Explicit collision counter."
        ),
    )

    # --------------------------------------------------------
    # Alpha shape / slice options
    # --------------------------------------------------------

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Alpha value for alpha-shape calculation."
        ),
    )

    parser.add_argument(
        "--alphashape",
        action="store_true",
        default=False,
        help=(
            "Enable alpha-shape processing."
        ),
    )

    parser.add_argument(
        "--upper_edge",
        action="store_true",
        default=False,
        help=(
            "Use upper-edge extraction."
        ),
    )

    parser.add_argument(
        "--upper_edge_bins",
        type=int,
        default=None,
        help=(
            "Number of bins for upper-edge extraction."
        ),
    )

    parser.add_argument(
        "--densify",
        type=float,
        default=None,
        help=(
            "Maximum segment length for alpha-shape densification."
        ),
    )

    parser.add_argument(
        "--skip_points",
        type=int,
        default=None,
        help=(
            "Number of points to skip in slice analysis."
        ),
    )

    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help=(
            "Slice position in z direction."
        ),
    )

    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help=(
            "X-axis limits."
        ),
    )

    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help=(
            "Y-axis limits."
        ),
    )

    return parser


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()