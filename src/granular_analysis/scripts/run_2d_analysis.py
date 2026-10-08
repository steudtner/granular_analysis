import os
from pathlib import Path

import numpy as np
import pandas as pd

from granular_analysis.paths import CasePaths

from granular_analysis.config import (
    FOLDER,
    OUTDIR_2D,
    RESULTS_DIR,
    ALPHA_2D,
    LZ,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
)

from granular_analysis.io.loader import load_frame, get_all_bin_paths
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.utils.state_manager import build_fingerprint, load_state, save_state


def write_surface_summary(result, outdir, frame, z_value, *,
                          initial_result=None, initial_frame=None):
    """Persist and print the surface dimensions alongside the upper-edge CSV."""
    keys = ("x_min", "x_max", "max_runout_length", "remaining_height")
    summary = {"frame": frame.name, "z_slice": z_value,
               **{key: result[key] for key in keys}}
    target = Path(outdir) / "surface_metrics.csv"
    rows = []
    if initial_frame is not None:
        rows.append({"state": "initial", "frame": initial_frame.name,
                     "z_slice": z_value,
                     **{key: initial_result[key] if initial_result is not None else None
                        for key in keys}})
    rows.append({"state": "selected", **summary})
    pd.DataFrame(rows).to_csv(target, index=False)
    print(f"  Böschungskante: {Path(outdir) / 'alphashape_last_timestep.csv'}")
    print(f"  Maximale Auslauflänge (x_max - x_min): {summary['max_runout_length']}")
    print(f"  Verbleibende Resthöhe (über y=0): {summary['remaining_height']}")
    print(f"  Oberflächenkennwerte: {target}")


def resolve_data_folder(data_folder):
    """Resolve and validate the canonical transient-data directory."""
    return str(CasePaths.from_input(data_folder).require_trn())


def main(
    force: bool = False,
    path: str = None,
    out: str = None,
    z_value: float = LZ / 2,
    densify: float = None,
    upper_edge_bins: int = 50,
    alpha: float = ALPHA_2D,
    lz: float = LZ,
    counter: int = None,
):
    """
    Run the 2D slice/porosity analysis.

    Parameters
    ----------
    force : bool
        Recompute even if a cached result exists.

    path : str
        Dataset path. Can point to:
            .../results
            .../results/ellipsoid
            .../results/ellipsoid/ellipsoid_trn

    out : str
        Writable output root directory.

    z_value : float
        z-coordinate of the slice used for the porosity calculation
        and upper-edge extraction.

        Default:
            LZ / 2

    densify : float, optional
        Maximum segment length used for boundary densification.

    upper_edge_bins : int
        Number of bins for upper-edge extraction.
    """

    # ------------------------------------------------------------
    # Resolve input data directory
    # ------------------------------------------------------------

    data_folder = (
        path
        if path
        else os.environ.get("GRANULAR_DATA_FOLDER")
        or FOLDER
    )

    data_folder = resolve_data_folder(data_folder)

    # ------------------------------------------------------------
    # Resolve writable output directory
    # ------------------------------------------------------------

    if out:
        results_dir = os.path.abspath(out)
        outdir = os.path.join(results_dir, "2d")
    else:
        results_dir = os.path.abspath(RESULTS_DIR)
        outdir = os.path.abspath(OUTDIR_2D)

    os.makedirs(outdir, exist_ok=True)

    print()
    print("2D ANALYSIS PATHS")
    print("------------------------------------------------------------")
    print(f"DATA       = {data_folder}")
    print(f"RESULTS    = {results_dir}")
    print(f"2D OUTPUT  = {outdir}")
    print("------------------------------------------------------------")

    # ------------------------------------------------------------
    # Check cache
    # ------------------------------------------------------------

    initial_frame = get_all_bin_paths(data_folder)[0]
    frame = (Path(data_folder) / f"ell_trn_{counter:06d}.bin" if counter is not None
             else max(Path(data_folder).glob("ell_trn_*.bin")))
    frame_stat = frame.stat()
    cache_fingerprint = build_fingerprint(
        frame=str(frame.resolve()),
        frame_size=frame_stat.st_size,
        frame_mtime_ns=frame_stat.st_mtime_ns,
        z_value=z_value,
        alpha=alpha,
        lz=lz,
        skip_first=SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0,
        densify=densify,
        upper_edge_bins=upper_edge_bins,
        initial_frame=str(initial_frame.resolve()),
        initial_frame_size=initial_frame.stat().st_size,
        initial_frame_mtime_ns=initial_frame.stat().st_mtime_ns,
        surface_metrics_version=2,
    )
    cached = None if force else load_state(
        "2d", results_dir, fingerprint=cache_fingerprint
    )

    def export_surface_summary(result):
        initial_result = result if initial_frame == frame else compute_slice_porosity(
            load_frame(data_folder, initial_frame.name),
            z0=z_value,
            alpha=alpha,
            lz=lz,
            densify_max_segment_length=densify,
            n_bins=upper_edge_bins,
            skip_first=SKIP_FIRST_PARTICLES if USE_PARTICLE_SKIP else 0,
            write_upper_edge=False,
        )
        write_surface_summary(
            result, outdir, frame, z_value,
            initial_result=initial_result, initial_frame=initial_frame,
        )

    if cached:
        print()
        print("2D Porosity result (cached):")

        for k, v in cached.items():
            if k != "timestamp":
                print(f"  {k} = {v}")

        print()
        print(
            f"Extracting upper edge at z0={z_value} "
            "(cached calculation)..."
        )

        ellipsoids = load_frame(data_folder, frame.name)

        result = compute_slice_porosity(
            ellipsoids,
            z0=z_value,
            alpha=alpha,
            lz=lz,
            densify_max_segment_length=densify,
            n_bins=upper_edge_bins,
            skip_first=(
                SKIP_FIRST_PARTICLES
                if USE_PARTICLE_SKIP
                else 0
            ),
            output_dir=outdir,
            write_upper_edge=True,
        )
        if result is not None:
            export_surface_summary(result)

        return

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    print()
    print(f"Loading ellipsoid frame {frame.name}...")

    ellipsoids = load_frame(data_folder, frame.name)

    particles = ellipsoids

    print(f"Loaded {len(particles)} particles.")

    # ------------------------------------------------------------
    # Main 2D computation
    #
    # Slice position is determined automatically from LZ:
    # z0 = LZ / 2
    # ------------------------------------------------------------

    print()
    print(f"Computing 2D porosity at z0={z_value} ...")

    result = compute_slice_porosity(
        particles,
        z0=z_value,
        alpha=alpha,
        lz=lz,
        densify_max_segment_length=densify,
        n_bins=upper_edge_bins,
        skip_first=(
            SKIP_FIRST_PARTICLES
            if USE_PARTICLE_SKIP
            else 0
        ),
        output_dir=outdir,
        write_upper_edge=True,
    )

    if result is None:
        print("No valid slice computed.")
        return

    # ------------------------------------------------------------
    # Convert numpy values to Python-native types
    # for JSON serialization
    # ------------------------------------------------------------

    result_serializable = {}

    for k, v in result.items():

        if isinstance(v, np.ndarray):
            result_serializable[k] = v.tolist()

        elif isinstance(v, np.number):
            result_serializable[k] = v.item()

        else:
            result_serializable[k] = v

    # ------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------

    print()
    print("2D Porosity result:")
    print(f"  porosity = {result['porosity']}")
    print(f"  A_ell    = {result['A_ell']}")
    print(f"  A_alpha  = {result['A_alpha']}")
    print(f"  n        = {result['n']}")
    export_surface_summary(result)

    # ------------------------------------------------------------
    # Save cache in the writable output directory
    # ------------------------------------------------------------

    save_state(
        "2d",
        result_serializable,
        results_dir,
        fingerprint=cache_fingerprint,
    )

    print()
    print("2D analysis completed successfully.")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
