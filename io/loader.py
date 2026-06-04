import os
from granular_analysis.read_ellipsoid_data import read_ellipsoid_trn


def load_last_frame(folder):

    files = sorted(f for f in os.listdir(folder) if f.endswith(".bin"))

    if not files:
        raise RuntimeError("no files")

    return read_ellipsoid_trn(os.path.join(folder, files[-1]))


def get_all_bin_files(folder):

    return sorted(f for f in os.listdir(folder) if f.endswith(".bin"))


def filter_particles(ellipsoids, use_skip, skip):

    if not use_skip:
        return ellipsoids

    return ellipsoids[skip:]