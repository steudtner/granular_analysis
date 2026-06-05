import numpy as np


def kinetic_energy(mass, velocities):
    """Per-particle kinetic energy: 0.5 * m * |v|^2."""
    return 0.5 * mass * np.sum(velocities**2, axis=1)


def total_kinetic_energy(mass, velocities):
    """Sum of per-particle kinetic energies."""
    return np.sum(kinetic_energy(mass, velocities))


def normalize_energy(E, rho_ref, g):
    """Normalize energy by reference density and gravity."""
    return E / (rho_ref * g)
