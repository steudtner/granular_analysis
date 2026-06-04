import numpy as np


def kinetic_energy(mass, velocities):

    return 0.5 * mass * np.sum(velocities**2, axis=1)


def total_kinetic_energy(mass, velocities):

    return np.sum(kinetic_energy(mass, velocities))


def normalize_energy(E, rho_ref, g):

    return E / (rho_ref * g)