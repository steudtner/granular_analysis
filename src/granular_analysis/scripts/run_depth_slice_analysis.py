import os
import pandas as pd

from granular_analysis.config import (
    FOLDER, OUTDIR_2D, ALPHA_2D, LZ,
    USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES, Z_LEVELS,
)
from granular_analysis.io.loader import load_last_frame, filter_particles
from granular_analysis.analysis.porosity2d import compute_slice_porosity
from granular_analysis.plotting.plots import plot_shape, plot_profile


def main():
    os.makedirs(OUTDIR_2D, exist_ok=True)

    ellipsoids = load_last_frame(FOLDER)
    particles = filter_particles(ellipsoids, USE_PARTICLE_SKIP, SKIP_FIRST_PARTICLES)

    results = []

    for z in Z_LEVELS:
        res = compute_slice_porosity(particles, z, ALPHA_2D, LZ)
        if res is None:
            continue
        out = os.path.join(OUTDIR_2D, f"z_{z:.2f}.png")
        plot_shape(res["shape"], out, f"z={z:.2f} phi={res['porosity']:.3f}")
        results.append({"z": z, "porosity": res["porosity"]})

    df = pd.DataFrame(results)
    csv = os.path.join(OUTDIR_2D, "porosity.csv")
    df.to_csv(csv, index=False)
    plot_profile(df.sort_values("z"), os.path.join(OUTDIR_2D, "profile.png"))
    print("2D slice analysis done")


if __name__ == "__main__":
    main()
