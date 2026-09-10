import numpy as np


def kinetic_energy(mass, velocities):
    """Per-particle kinetic energy: 0.5 * m * |v|^2."""
    return 0.5 * mass * np.sum(velocities**2, axis=1)


def total_kinetic_energy(mass, velocities):
    """Sum of per-particle kinetic energies."""
    return np.sum(kinetic_energy(mass, velocities))


def rotational_energy(mass, semi_axes, angular_velocity):
    """Per-particle rotational energy for homogeneous ellipsoids."""
    a, b, c = np.asarray(semi_axes).T
    inertia = mass[:, None] * np.column_stack((
        b**2 + c**2,
        a**2 + c**2,
        a**2 + b**2,
    )) / 5.0
    return 0.5 * np.sum(inertia * np.asarray(angular_velocity)**2, axis=1)


def normalize_energy(E, rho_ref, g):
    """Normalize energy by reference density and gravity."""
    return E / (rho_ref * g)
