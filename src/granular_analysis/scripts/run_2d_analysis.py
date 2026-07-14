import os
import numpy as np

from granular_analysis.config import (
    FOLDER,
    OUTDIR_2D,
    RESULTS_DIR,
    ALPHA_2D,
    LZ,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
)
from granular_analysis.io.loader import load_last_frame, filter_particles
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.utils.state_manager import load_state, save_state


def main(force: bool = False):
    # ensure output directory exists
    os.makedirs(OUTDIR_2D, exist_ok=True)

    # Check cache
    cached = None if force else load_state("2d", RESULTS_DIR)
    if cached:
        print("2D Porosity result (cached):")
        for k, v in cached.items():
            if k != "timestamp":
                print(f"  {k} = {v}")
        return

    # load data
    ellipsoids = load_last_frame(FOLDER)

    # FIX: correct filter_particles signature
    particles = filter_particles(
        ellipsoids,
        USE_PARTICLE_SKIP,
        SKIP_FIRST_PARTICLES,
    )

    # computation
    result = compute_slice_porosity(
        particles,
        z0=0.0,
        alpha=ALPHA_2D,
        lz=LZ,
    )

    if result is None:
        print("No valid slice computed.")
        return

    # Convert numpy values to python native types for JSON serialization
    result_serializable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            result_serializable[k] = v.tolist()
        elif isinstance(v, np.number):
            result_serializable[k] = v.item()
        else:
            result_serializable[k] = v

    # output
    print("2D Porosity result:")
    print(f"  porosity = {result['porosity']}")
    print(f"  A_ell    = {result['A_ell']}")
    print(f"  A_alpha  = {result['A_alpha']}")
    print(f"  n        = {result['n']}")

    # Save to cache
    save_state("2d", result_serializable, RESULTS_DIR)


if __name__ == "__main__":
    main()