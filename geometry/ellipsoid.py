import numpy as np


def ellipsoid_volume(e):
    return (4/3) * np.pi * e["a"] * e["b"] * e["c"]


def ellipsoid_slice(e, z0, shift=0.0):

    x0, y0, zc = e["x"]
    a, b, c = e["a"], e["b"], e["c"]

    # ---- SAFETY CHECK ----
    if c is None or c <= 0 or not np.isfinite(c):
        return None

    dz = (zc + shift) - z0

    k = 1.0 - (dz * dz) / (c * c)

    if not np.isfinite(k) or k <= 0:
        return None

    ax = a * np.sqrt(k)
    by = b * np.sqrt(k)

    if not np.isfinite(ax) or not np.isfinite(by):
        return None

    return x0, y0, ax, by


def ellipse_points(x0, y0, ax_e, by_e, n=60):

    t = np.linspace(0, 2*np.pi, n)

    x = x0 + ax_e * np.cos(t)
    y = y0 + by_e * np.sin(t)

    return np.column_stack([x, y])


def ellipsoid_surface_points(e, nu=10, nv=10):

    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)

    uu, vv = np.meshgrid(u, v)

    x0, y0, z0 = e["x"]

    x = x0 + e["a"] * np.cos(uu) * np.sin(vv)
    y = y0 + e["b"] * np.sin(uu) * np.sin(vv)
    z = z0 + e["c"] * np.cos(vv)

    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])
