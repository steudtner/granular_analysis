import numpy as np


def slice_particles_by_z(pts_all, z0, z1):
    mask = (pts_all[:, 2] >= z0) & (pts_all[:, 2] < z1)
    return pts_all[mask]


def project_xy(pts):
    return pts[:, :2]
