import os

from granular_analysis.config import (
    FOLDER,
    OUTDIR_2D,
    ALPHA_2D,
    LZ,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
)
from granular_analysis.io.loader import load_last_frame, filter_particles
from granular_analysis.analysis.porosity2d import compute_slice_porosity


def main():
    # ensure output directory exists
    os.makedirs(OUTDIR_2D, exist_ok=True)

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

    # output
    print("2D Porosity result:")
    print(f"  porosity = {result['porosity']}")
    print(f"  A_ell    = {result['A_ell']}")
    print(f"  A_alpha  = {result['A_alpha']}")
    print(f"  n        = {result['n']}")


if __name__ == "__main__":
    main()