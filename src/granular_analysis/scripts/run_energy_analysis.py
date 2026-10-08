#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from granular_analysis.config import D_EQ, FOLDER, RESULTS_DIR, SKIP_FIRST_PARTICLES
from granular_analysis.paths import CasePaths
from granular_analysis.plotting import plotstyle

from granular_analysis.io.monitor_loader import (
    get_monitor_files,
    read_monitor_file,
    monitor_source_paths,
)

from granular_analysis.analysis.energy.kinetic_energy import (
    kinetic_energy,
    rotational_energy,
    translational_rotational_ratio,
)

from granular_analysis.plotting.plots import (
    plot_energy_components,
    plot_energy_ratio,
    plot_energy_histogram,
    plot_energy_time_series,
)

from granular_analysis.utils.physics import (
    load_ellipsoid_properties,
    read_density_first_particle,
    read_gravity,
    initial_potential_energy,
)

from granular_analysis.utils.state_manager import (
    build_fingerprint,
    load_state,
    save_state,
)


# ============================================================
# INPUT PATH RESOLUTION
# ============================================================

def _auto_input_paths(folder=FOLDER):
    """
    Automatically determine the input files from FOLDER.

    Expected structure:

        <case>/
            input.dat
            input_ellipsoid_data.dat
            results/
                ellipsoid/
                    ellipsoid_trn/

    Therefore:

        FOLDER
          = <case>/results/ellipsoid/ellipsoid_trn

        parent
          = <case>/results/ellipsoid

        parent.parent
          = <case>/results

        parent.parent.parent
          = <case>
    """

    case_root = CasePaths.from_input(folder).case_root

    return {
        "global": case_root / "input.dat",
        "ellipsoid": case_root / "input_ellipsoid_data.dat",
    }


# ============================================================
# MONITOR FOLDER
# ============================================================

def _monitor_folder(folder=FOLDER):
    """
    Monitor files are located one directory above ellipsoid_trn.

    Expected:

        .../ellipsoid/
            ellipsoid_trn/
            ellipsoid_monitor/
            ...
    """

    case = CasePaths.from_input(folder)
    return case.monitor if case.monitor.is_dir() else case.ellipsoid


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():
    """
    Arguments for direct execution of this script.

    IMPORTANT:
    run_all.py does NOT call this parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute total kinetic energy "
            "from ellipsoid monitor files."
        )
    )

    parser.add_argument(
        "--path",
        default=None,
        help="Case root or any supported results subdirectory.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV path. "
            "If omitted, use RESULTS_DIR/energy.csv."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation.",
    )

    parser.add_argument(
        "--histogram-timesteps",
        default=None,
        help=(
            "Comma-separated timestep indices "
            "for particle-energy histograms. "
            "Example: 0,100,200,-1"
        ),
    )

    parser.add_argument("--rotation-share-threshold", type=float, default=0.5,
                        help="Exclude a particle rotation contribution above this fraction of raw rotational energy (0.5 to 1; 1 disables).")
    parser.add_argument("--potential-energy", action="store_true")
    parser.add_argument(
        "--no-physical-mass",
        action="store_true",
        help=(
            "Disable physical masses. "
            "Not supported for monitor-based energy."
        ),
    )

    return parser.parse_args()


# ============================================================
# HISTOGRAM TIMESTEP PARSER
# ============================================================

def _parse_histogram_timesteps(
    arg,
    total_rows,
):
    """
    Convert a comma-separated timestep specification
    into valid dataframe indices.

    Examples:

        "0,100,200"
        "-1"
        "0,-1,100"
    """

    if arg is None:
        return []

    indices = []

    for value in arg.split(","):

        value = value.strip()

        if not value:
            continue

        index = int(value)

        if index < 0:
            index += total_rows

        if 0 <= index < total_rows:
            indices.append(index)

    return sorted(set(indices))


# ============================================================
# LOAD MONITOR SERIES
# ============================================================

def _load_monitor_series(
    folder,
    particle_ids,
    masses,
    semi_axes,
    skip_first=0,
    rotation_share_threshold=1.0,
    diagnostics=None,
):
    """
    Load all ellipsoid monitor files and calculate:

        translational kinetic energy
        rotational kinetic energy
        total kinetic energy

    Energies are interpolated onto the time grid
    of the first monitor file.
    """

    if not 0.5 <= rotation_share_threshold <= 1.0:
        raise ValueError("rotation_share_threshold must be between 0.5 and 1")

    files = [(number, path) for number, path in get_monitor_files(folder)
             if number > skip_first or number < 1]

    if not files:
        raise RuntimeError(
            f"No monitor files remain in {folder} after skipping {skip_first} input rows"
        )

    # --------------------------------------------------------
    # Reference time grid
    # --------------------------------------------------------

    _, first_path = files[0]

    first_data = read_monitor_file(
        first_path
    )

    times = first_data[:, 1].copy()

    if len(times) < 2:
        raise ValueError(
            f"Monitor file contains fewer than "
            f"two timesteps: {first_path}"
        )

    if np.any(np.diff(times) <= 0):
        raise ValueError(
            f"Non-increasing time grid in "
            f"{first_path}"
        )

    # --------------------------------------------------------
    # Allocate arrays
    # --------------------------------------------------------

    translational_energy = np.zeros(
        len(times)
    )

    rotational_energy_total = np.zeros(
        len(times)
    )

    particle_energies = np.empty(
        (
            len(files),
            len(times),
        )
    )

    largest_rotation = np.zeros(len(times))
    largest_particle = np.zeros(len(times), dtype=int)

    valid = np.ones(
        len(times),
        dtype=bool,
    )

    # --------------------------------------------------------
    # Process particles
    # --------------------------------------------------------

    for particle_index, (
        monitor_number,
        path,
    ) in enumerate(files):

        # ----------------------------------------------------
        # Map the one-based input row, even when monitor files are missing.
        # ----------------------------------------------------

        property_row = monitor_number - 1
        if not 0 <= property_row < len(particle_ids):
            raise RuntimeError(
                f"Monitor number {monitor_number} in {path} refers to input "
                f"particle row {monitor_number}, but the input contains "
                f"{len(particle_ids)} particles"
            )

        # ----------------------------------------------------
        # Load monitor data
        # ----------------------------------------------------

        data = (
            first_data
            if particle_index == 0
            else read_monitor_file(path)
        )

        monitor_times = data[:, 1]

        # ----------------------------------------------------
        # Validate time grid
        # ----------------------------------------------------

        if len(monitor_times) < 2:

            raise ValueError(
                f"Monitor file contains fewer "
                f"than two timesteps: {path}"
            )

        if np.any(
            np.diff(monitor_times) <= 0
        ):

            raise ValueError(
                f"Non-increasing time grid in "
                f"{path}"
            )

        # ----------------------------------------------------
        # Mass
        # ----------------------------------------------------

        mass = np.full(
            len(monitor_times),
            masses[property_row],
        )

        # ----------------------------------------------------
        # Semi axes
        # ----------------------------------------------------

        axes = np.repeat(
            semi_axes[property_row][None, :],
            len(monitor_times),
            axis=0,
        )

        # ----------------------------------------------------
        # Translational energy
        #
        # data[:, 5:8] = velocity
        # ----------------------------------------------------

        translational = kinetic_energy(
            mass,
            data[:, 5:8],
        )

        # ----------------------------------------------------
        # Rotational energy
        #
        # data[:, 8:11] = angular velocity
        # ----------------------------------------------------

        rotational = rotational_energy(
            mass,
            axes,
            data[:, 8:11],
        )

        # ----------------------------------------------------
        # Total particle energy
        # ----------------------------------------------------

        energy = (
            translational
            + rotational
        )

        # ----------------------------------------------------
        # Determine common time range
        # ----------------------------------------------------

        inside = (
            (times >= monitor_times[0])
            & (
                times
                <= monitor_times[-1]
            )
        )

        valid &= inside

        # ----------------------------------------------------
        # Interpolate particle energy
        # ----------------------------------------------------

        particle_energies[
            particle_index
        ] = np.interp(
            times,
            monitor_times,
            energy,
        )

        # ----------------------------------------------------
        # Interpolate translational energy
        # ----------------------------------------------------

        translational_energy += np.interp(
            times,
            monitor_times,
            translational,
        )

        # ----------------------------------------------------
        # Interpolate rotational energy
        # ----------------------------------------------------

        interpolated_rotation = np.interp(times, monitor_times, rotational)
        rotational_energy_total += interpolated_rotation
        larger = interpolated_rotation > largest_rotation
        largest_rotation[larger] = interpolated_rotation[larger]
        largest_particle[larger] = particle_index

    # Compare once against the unfiltered total; never iteratively remove particles.
    raw_rotation = rotational_energy_total.copy()
    excluded = largest_rotation > rotation_share_threshold * raw_rotation
    removed = np.where(excluded, largest_rotation, 0.0)
    rotational_energy_total -= removed
    columns = np.flatnonzero(excluded)
    particle_energies[largest_particle[columns], columns] -= removed[columns]
    if diagnostics is not None:
        monitor_numbers = np.array([number for number, _ in files])
        diagnostics.update(
            rotational_energy_raw=raw_rotation[valid],
            rotational_energy_excluded=removed[valid],
            rotation_excluded_monitor=np.where(excluded, monitor_numbers[largest_particle], 0)[valid],
        )

    # ========================================================
    # Apply common valid time range
    # ========================================================

    times = times[valid]

    particle_energies = (
        particle_energies[:, valid]
    )

    translational_energy = (
        translational_energy[valid]
    )

    rotational_energy_total = (
        rotational_energy_total[valid]
    )

    # ========================================================
    # Total energy
    # ========================================================

    total_energy = (
        translational_energy
        + rotational_energy_total
    )

    return (
        times,
        translational_energy,
        rotational_energy_total,
        total_energy,
        particle_energies,
        files,
    )


# ============================================================
# ENERGY DATAFRAME
# ============================================================

def compute_energy_dataframe(
    folder,
    use_physical_mass=True,
    skip_first=None,
    input_paths=None,
    rotation_share_threshold=0.5,
):
    """Calculate the kinetic-energy time series as a pandas DataFrame."""
    df, _, _ = _compute_energy_data(
        folder, use_physical_mass, skip_first, input_paths, rotation_share_threshold
    )
    return df


def _compute_energy_data(
    folder,
    use_physical_mass=True,
    skip_first=None,
    input_paths=None,
    rotation_share_threshold=0.5,
):
    """
    Calculate the time series and retain per-particle energies for histograms.

    Returns (DataFrame, particle_energies, monitor_files). DataFrame columns:

        time
        time_norm

        translational_energy
        rotational_energy
        kinetic_energy

        translational_energy_norm
        rotational_energy_norm
        kinetic_energy_norm
        translational_rotational_ratio
    """

    if not 0.5 <= rotation_share_threshold <= 1.0:
        raise ValueError("rotation_share_threshold must be between 0.5 and 1")
    skip_first = SKIP_FIRST_PARTICLES if skip_first is None else skip_first
    if skip_first < 0:
        raise ValueError("skip_first must be nonnegative")

    if input_paths is None:
        input_paths = _auto_input_paths(folder)

    # --------------------------------------------------------
    # Physical mass is mandatory
    # --------------------------------------------------------

    if not use_physical_mass:

        raise ValueError(
            "Monitor-based energy analysis "
            "requires physical masses."
        )

    # --------------------------------------------------------
    # Read physical parameters
    # --------------------------------------------------------

    gravity = read_gravity(
        input_paths["global"]
    )

    rho_ref = read_density_first_particle(
        input_paths["ellipsoid"]
    )

    (
        particle_ids,
        masses,
        a,
        b,
        c,
    ) = load_ellipsoid_properties(
        input_paths["ellipsoid"]
    )

    # --------------------------------------------------------
    # Load monitor data
    # --------------------------------------------------------

    diagnostics = {}
    (
        times,
        translational,
        rotational,
        total,
        particle_energies,
        files,
    ) = _load_monitor_series(
        _monitor_folder(folder),
        particle_ids,
        masses,
        np.column_stack(
            (a, b, c)
        ),
        skip_first=skip_first,
        rotation_share_threshold=rotation_share_threshold,
        diagnostics=diagnostics,
    )

    # ========================================================
    # Energy normalization
    #
    # E* = E / (rho_ref * g * D_eq^4)
    # ========================================================

    normalization = (
        rho_ref
        * gravity
        * D_EQ**4
    )

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame({

        "time":
            times,

        "time_norm":
            times
            * np.sqrt(
                gravity / D_EQ
            ),

        "translational_energy":
            translational,

        "rotational_energy":
            rotational,

        "kinetic_energy":
            total,

        "translational_energy_norm":
            translational
            / normalization,

        "rotational_energy_norm":
            rotational
            / normalization,

        "kinetic_energy_norm":
            total
            / normalization,
    })
    try:
        potential = initial_potential_energy(input_paths["global"], input_paths["ellipsoid"], skip_first=skip_first)
    except ValueError as exc:
        print(f"WARNING: Initial potential-energy reference unavailable: {exc}")
        potential = np.nan
    df["translational_rotational_ratio"] = translational_rotational_ratio(
        translational, rotational
    )
    df["initial_potential_energy"] = potential
    df["initial_potential_energy_norm"] = potential / normalization
    for name, values in diagnostics.items():
        df[name] = values
    df["rotation_share_threshold"] = rotation_share_threshold
    df["rotational_energy_raw_norm"] = df["rotational_energy_raw"] / normalization
    df["skip_first_particles"] = skip_first
    return df, particle_energies, files


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def main(
    force: bool = False,
    path: Optional[str] = None,
    out: Optional[str] = None,
    histogram_timesteps=None,
    no_physical_mass: bool = False,
    skip_first=None,
    potential_energy=False,
    rotation_share_threshold=0.5,
):
    """
    Run the energy analysis.

    This function can be called by run_all.py without
    parsing sys.argv again.

    Parameters
    ----------
    force:
        Ignore existing cached results.

    out:
        Explicit output CSV path.

        If None:
            RESULTS_DIR/energy.csv

    histogram_timesteps:
        Comma-separated timestep indices.

    no_physical_mass:
        Reject physical-mass calculation.
    """

    # ========================================================
    # Input validation
    # ========================================================

    if not 0.5 <= rotation_share_threshold <= 1.0:
        raise ValueError("rotation_share_threshold must be between 0.5 and 1")
    skip_first = SKIP_FIRST_PARTICLES if skip_first is None else skip_first
    if skip_first < 0:
        raise ValueError("skip_first must be nonnegative")
    if no_physical_mass:

        raise ValueError(
            "--no-physical-mass is incompatible "
            "with monitor files."
        )

    # ========================================================
    # Determine output CSV
    # ========================================================

    if out is not None:

        output_csv = Path(out)

    else:

        output_csv = (
            Path(RESULTS_DIR)
            / "energy.csv"
        )

    output_dir = (
        output_csv.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Energy output directory = "
        f"{output_dir}"
    )

    print(
        f"Energy CSV              = "
        f"{output_csv}"
    )

    energy_png = (
        output_csv.with_suffix(".png")
    )

    # ========================================================
    # Input paths and cache
    # ========================================================

    data_folder = path or FOLDER
    input_paths = _auto_input_paths(data_folder)
    monitored = [(number, file) for number, file in get_monitor_files(_monitor_folder(data_folder))
                 if number > skip_first]
    print(f"Energy particle selection: skipping first {skip_first} input rows; {len(monitored)} monitors retained.")
    source_files = [input_paths["global"], input_paths["ellipsoid"]]
    source_files.extend(source for _, path in monitored for source in monitor_source_paths(path))
    signatures = []
    for source in source_files:
        stat = Path(source).stat()
        signatures.append((str(Path(source).resolve()), stat.st_size, stat.st_mtime_ns))
    cache_fingerprint = build_fingerprint(
        sources=signatures,
        plot_style="cm_t1_10pt_svg_pdf_powers_legend2_v2",
        monitor_mapping="input_row_v1",
        energy_ratio="translational_over_rotational_v1",
        rotation_share_threshold=rotation_share_threshold,
        rotation_filter="dominant_raw_share_v1",
        potential_reference="selected_input_particles_y_height_v3",
        potential_energy=potential_energy,
        skip_first=skip_first,
        d_eq=D_EQ,
        histogram_timesteps=histogram_timesteps,
    )
    cached = None if force else load_state(
        "energy", str(output_dir), fingerprint=cache_fingerprint
    )
    components_png = output_dir / f"{output_csv.stem}_components.png"
    ratio_png = output_dir / f"{output_csv.stem}_ratio.png"
    if (cached and output_csv.is_file() and energy_png.is_file()
            and components_png.is_file() and ratio_png.is_file()
            and all(path.with_suffix(ext).is_file()
                    for path in (energy_png, components_png, ratio_png)
                    for ext in (".pdf", ".svg"))):
        print("Energy Analysis (cached)")
        return

    print(f"Energy input FOLDER      = {data_folder}")
    print(f"Global input             = {input_paths['global']}")
    print(f"Ellipsoid input          = {input_paths['ellipsoid']}")

    # ========================================================
    # Required CSV columns
    # ========================================================

    required_columns = {
        "time",
        "time_norm",
        "translational_energy",
        "rotational_energy",
        "kinetic_energy",
        "translational_energy_norm",
        "rotational_energy_norm",
        "kinetic_energy_norm",
        "translational_rotational_ratio",
    }

    needs_recompute = True
    particle_energies = None

    # ========================================================
    # Existing CSV
    # ========================================================

    if (
        output_csv.exists()
        and not force
        and cached is not None
    ):

        cached_df = pd.read_csv(
            output_csv
        )

        needs_recompute = not (
            required_columns.issubset(
                cached_df.columns
            )
        )

        if not needs_recompute:

            df = cached_df

    # ========================================================
    # Compute energy
    # ========================================================

    if needs_recompute:

        print(f"Reading {len(monitored)} monitor files and computing energy...")
        df, particle_energies, files = _compute_energy_data(
            data_folder,
            input_paths=input_paths,
            skip_first=skip_first,
            rotation_share_threshold=rotation_share_threshold,
        )
        print(f"Rotation filter: {int((df.rotation_excluded_monitor != 0).sum())} samples excluded; threshold={rotation_share_threshold}")

        df.to_csv(
            output_csv,
            index=False,
        )

        print(
            f"Wrote energy data to "
            f"{output_csv}"
        )

    # ========================================================
    # Energy time series
    # ========================================================

    plot_energy_time_series(
        df,
        energy_png,
        use_normalized=True,
        potential_energy=potential_energy,
    )

    print(
        f"Wrote energy plot to "
        f"{energy_png}"
    )

    # ========================================================
    # Energy components
    # ========================================================

    plot_energy_components(
        df,
        components_png,
        use_normalized=True,
        potential_energy=potential_energy,
    )

    print(
        f"Wrote energy component plot to "
        f"{components_png}"
    )

    # ========================================================
    # Particle energy histogram
    # ========================================================

    plot_energy_ratio(df, ratio_png, use_normalized=True)
    print(f"Wrote energy ratio plot to {ratio_png}")

    if particle_energies is None:
        print(f"Reading {len(monitored)} monitor files for energy histograms...")
        particle_ids, masses, a, b, c = load_ellipsoid_properties(
            input_paths["ellipsoid"]
        )
        _, _, _, _, particle_energies, files = _load_monitor_series(
            _monitor_folder(data_folder),
            particle_ids,
            masses,
            np.column_stack((a, b, c)),
            skip_first=skip_first,
        )

    # ========================================================
    # Histogram indices
    # ========================================================

    indices = (
        _parse_histogram_timesteps(
            histogram_timesteps,
            len(df),
        )
    )

    # Default:
    # final timestep
    if not indices and len(df):

        indices = [
            len(df) - 1
        ]

    # ========================================================
    # Histogram normalization
    # ========================================================

    normalization = (
        read_density_first_particle(
            input_paths["ellipsoid"]
        )
    )

    normalization *= (
        read_gravity(
            input_paths["global"]
        )
        * D_EQ**4
    )

    # ========================================================
    # Generate histograms
    # ========================================================

    if indices:

        print(f"Creating {len(indices)} energy histogram(s)...")

        hist_dir = (
            output_dir
            / "energy_histograms"
        )

        hist_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index in indices:

            label = (
                f"t="
                f"{df.iloc[index]['time']:.4f}"
            )

            out_path = (
                hist_dir
                / (
                    "particle_energy_hist_"
                    f"{index:06d}.png"
                )
            )

            plot_energy_histogram(
                particle_energies[
                    :, index
                ],
                label,
                out_path,
                normalization=normalization,
            )

    # ========================================================
    # Save analysis state
    # ========================================================

    stats = {

        "E_max":
            float(
                df[
                    "kinetic_energy"
                ].max()
            ),

        "E_final":
            float(
                df[
                    "kinetic_energy"
                ].iloc[-1]
            ),

        "n_frames":
            len(df),

        "use_physical_mass":
            True,

        "n_monitor_files":
            len(files),
    }

    save_state(
        "energy",
        stats,
        str(output_dir),
        fingerprint=cache_fingerprint,
    )

    print(
        "Energy analysis done"
    )


# ============================================================
# DIRECT CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    main(
        force=args.force,
        path=args.path,
        out=args.out,
        histogram_timesteps=(
            args.histogram_timesteps
        ),
        no_physical_mass=(
            args.no_physical_mass
        ),
        potential_energy=args.potential_energy,
        rotation_share_threshold=args.rotation_share_threshold,
    )
