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
from granular_analysis.analysis.porosity3d_opentop import compute_porosity_3d_opentop
from granular_analysis.utils.logging_utils import suppress_noise


def main():
    suppress_noise()

    os.makedirs(OUTDIR_3D, exist_ok=True)

    ellipsoids = load_last_frame(FOLDER)
    particles = filter_particles(ellipsoids, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES)
    print(f"Loaded particles: {len(particles)}")

    classic = compute_porosity_3d(particles, alpha=ALPHA_2D)
    opentop = compute_porosity_3d_opentop(particles, alpha=ALPHA_2D, z_min=0.0, z_max=2.5)

    print()
    print("=" * 35)
    print("POROSITY COMPARISON")
    print("=" * 35)

    print("\nCLASSIC")
    print(f"  porosity = {classic['porosity']:.6f}")
    print(f"  V_alpha  = {classic['V_alpha']:.6f}")
    print(f"  V_ell    = {classic['V_ell']:.6f}")

    print("\nOPENTOP CLOSED")
    print(f"  porosity = {opentop['porosity']:.6f}")
    print(f"  V_alpha  = {opentop['V_alpha']:.6f}")
    print(f"  V_ell    = {opentop['V_ell']:.6f}")

    print(f"\nΔporosity = {abs(classic['porosity'] - opentop['porosity']):.6f}")

    classic_path = os.path.join(OUTDIR_3D, "alpha_classic.ply")
    opentop_path = os.path.join(OUTDIR_3D, "alpha_opentop_closed.ply")

    classic["mesh"].export(classic_path)
    opentop["mesh"].export(opentop_path)

    print(f"\nSaved:\n  {classic_path}\n  {opentop_path}")


if __name__ == "__main__":
    main()
