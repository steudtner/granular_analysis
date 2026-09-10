#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from granular_analysis.config import (
    D_EQ,
    FOLDER,
    RESULTS_DIR,
)

from granular_analysis.io.monitor_loader import (
    get_monitor_files,
    read_monitor_file,
)

from granular_analysis.analysis.energy.kinetic_energy import (
    kinetic_energy,
    rotational_energy,
)

from granular_analysis.plotting.plots import (
    plot_energy_components,
    plot_energy_histogram,
    plot_energy_time_series,
)

from granular_analysis.utils.physics import (
    load_ellipsoid_properties,
    read_density_first_particle,
    read_gravity,
)

from granular_analysis.utils.state_manager import (
    load_state,
    save_state,
)


# ============================================================
# INPUT PATH RESOLUTION
# ============================================================

def _auto_input_paths():
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

    case_root = (
        Path(FOLDER)
        .resolve()
        .parent
        .parent
        .parent
    )

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

    return Path(folder).resolve().parent


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
    masses,
    semi_axes,
):
    """
    Load all ellipsoid monitor files and calculate:

        translational kinetic energy
        rotational kinetic energy
        total kinetic energy

    Energies are interpolated onto the time grid
    of the first monitor file.
    """

    files = get_monitor_files(folder)

    if not files:
        raise RuntimeError(
            f"No monitor files found in {folder}"
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

    valid = np.ones(
        len(times),
        dtype=bool,
    )

    # --------------------------------------------------------
    # Process particles
    # --------------------------------------------------------

    for particle_index, (
        particle_id,
        path,
    ) in enumerate(files):

        # ----------------------------------------------------
        # Check particle ID
        # ----------------------------------------------------

        if (
            particle_id < 1
            or particle_id > len(masses)
        ):
            raise RuntimeError(
                f"Monitor ID {particle_id} "
                f"has no matching input particle"
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
            masses[particle_id - 1],
        )

        # ----------------------------------------------------
        # Semi axes
        # ----------------------------------------------------

        axes = np.repeat(
            semi_axes[
                particle_id - 1
            ][None, :],
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

        rotational_energy_total += np.interp(
            times,
            monitor_times,
            rotational,
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
    skip_first=0,
    input_paths=None,
):
    """
    Calculate the complete kinetic-energy time series.

    Returns a pandas DataFrame containing:

        time
        time_norm

        translational_energy
        rotational_energy
        kinetic_energy

        translational_energy_norm
        rotational_energy_norm
        kinetic_energy_norm
    """

    # Kept for API compatibility.
    del skip_first

    if input_paths is None:
        input_paths = _auto_input_paths()

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
        _,
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

    (
        times,
        translational,
        rotational,
        total,
        _,
        _,
    ) = _load_monitor_series(
        _monitor_folder(folder),
        masses,
        np.column_stack(
            (a, b, c)
        ),
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

    return pd.DataFrame({

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


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def main(
    force: bool = False,
    out: Optional[str] = None,
    histogram_timesteps=None,
    no_physical_mass: bool = False,
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
    # Cache
    # ========================================================

    cached = None

    if not force:

        cached = load_state(
            "energy",
            str(output_dir),
        )

    if cached:

        print(
            "Energy Analysis (cached)"
        )

        return

    # ========================================================
    # Input paths
    # ========================================================

    input_paths = (
        _auto_input_paths()
    )

    print(
        f"Energy input FOLDER      = "
        f"{FOLDER}"
    )

    print(
        f"Global input             = "
        f"{input_paths['global']}"
    )

    print(
        f"Ellipsoid input          = "
        f"{input_paths['ellipsoid']}"
    )

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
    }

    needs_recompute = True

    # ========================================================
    # Existing CSV
    # ========================================================

    if (
        output_csv.exists()
        and not force
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

        df = compute_energy_dataframe(
            FOLDER,
            input_paths=input_paths,
        )

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
    )

    print(
        f"Wrote energy plot to "
        f"{energy_png}"
    )

    # ========================================================
    # Energy components
    # ========================================================

    components_png = (
        output_dir
        / f"{output_csv.stem}_components.png"
    )

    plot_energy_components(
        df,
        components_png,
        use_normalized=True,
    )

    print(
        f"Wrote energy component plot to "
        f"{components_png}"
    )

    # ========================================================
    # Particle energy histogram
    # ========================================================

    (
        _,
        masses,
        a,
        b,
        c,
    ) = load_ellipsoid_properties(
        input_paths["ellipsoid"]
    )

    (
        _,
        _,
        _,
        _,
        particle_energies,
        files,
    ) = _load_monitor_series(
        _monitor_folder(),
        masses,
        np.column_stack(
            (a, b, c)
        ),
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
        out=args.out,
        histogram_timesteps=(
            args.histogram_timesteps
        ),
        no_physical_mass=(
            args.no_physical_mass
        ),
    )
