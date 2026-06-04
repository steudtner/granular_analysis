import os

from granular_analysis.config import (
    FOLDER,
    OUTDIR_3D,
    USE_PARTICLE_SKIP,
    SKIP_FIRST_PARTICLES,
    ALPHA_2D,
)

from granular_analysis.io.loader import load_last_frame, filter_particles
from granular_analysis.analysis.porosity3d import compute_porosity_3d
from granular_analysis.scripts.visualize_3d_slices import plot_alpha_variants


def main():

    os.makedirs(OUTDIR_3D, exist_ok=True)

    ellipsoids = load_last_frame(FOLDER)

    particles = filter_particles(
        ellipsoids,
        USE_PARTICLE_SKIP,
        SKIP_FIRST_PARTICLES,
    )

    print(f"Loaded particles: {len(particles)}")

    # ----------------------------
    # 3D POROSITY
    # ----------------------------
    result = compute_porosity_3d(
        particles,
        alpha=ALPHA_2D
    )

    print("\n3D Porosity result:")
    print(f"  porosity = {result['porosity']}")
    print(f"  V_ell    = {result['V_ell']}")
    print(f"  V_alpha  = {result['V_alpha']}")

    # ----------------------------
    # VISUALIZATION
    # ----------------------------
    pts = result["mesh"].vertices

    plot_alpha_variants(
        pts,
        alpha=ALPHA_2D,
        save_dir=OUTDIR_3D
    )


if __name__ == "__main__":
    main()