import numpy as np


def ellipsoid_volume(e):
    return (4.0 / 3.0) * np.pi * e["a"] * e["b"] * e["c"]


def _rotation_matrix(e):
    """Lokale Hauptachsen -> globale Koordinaten."""
    if "R" in e:
        R = np.asarray(e["R"], dtype=float)
        if R.shape == (9,):
            R = R.reshape(3, 3)
        if R.shape != (3, 3):
            raise ValueError("R muss eine 3x3-Rotationsmatrix sein.")
        return R

    q = np.asarray(e.get("q", [1.0, 0.0, 0.0, 0.0]), dtype=float)
    if q.shape != (4,):
        raise ValueError("q muss [w, x, y, z] enthalten.")

    q /= np.linalg.norm(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def ellipsoid_slice(e, z_plane, shift=0.0):
    """
    Schnitt eines beliebig orientierten Ellipsoids mit z = z_plane.

        Rückgabe:
            x0, y0 : Mittelpunkt der Schnittellipse
            ax, by : große/kleine Halbachse
            theta  : Orientierung der großen Halbachse gegen global-x [rad]
    """
    center = np.asarray(e["x"], dtype=float).copy()
    center[2] += shift

    a, b, c = float(e["a"]), float(e["b"]), float(e["c"])
    if min(a, b, c) <= 0 or not np.all(np.isfinite([a, b, c])):
        return None

    R = _rotation_matrix(e)

    # (p-center)^T Q (p-center) = 1
    Q = R @ np.diag([1.0 / a**2, 1.0 / b**2, 1.0 / c**2]) @ R.T

    dz = float(z_plane - center[2])
    A = Q[:2, :2]
    q = Q[:2, 2]
    qzz = Q[2, 2]

    try:
        A_inv_q = np.linalg.solve(A, q)
    except np.linalg.LinAlgError:
        return None

    # Mittelpunkt der Schnittellipse in der Ebene
    xy_offset = -dz * A_inv_q

    # Rechte Seite nach quadratischer Ergänzung
    radius_sq = 1.0 - qzz * dz**2 + dz**2 * (q @ A_inv_q)
    if not np.isfinite(radius_sq) or radius_sq <= 0:
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    if np.any(eigenvalues <= 0):
        return None

    # Kleine Eigenwerte entsprechen großen Halbachsen.
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    ax = np.sqrt(radius_sq / eigenvalues[0])
    by = np.sqrt(radius_sq / eigenvalues[1])

    direction_major = eigenvectors[:, 0]
    theta = np.arctan2(direction_major[1], direction_major[0])

    return center[0] + xy_offset[0], center[1] + xy_offset[1], ax, by, theta


def ellipse_points(x0, y0, ax_e, by_e, theta=0.0, n=10):
    """Punkte einer in der x-y-Ebene gedrehten Ellipse, gleichmäßig über den
       tatsächlichen Umfang (Bogenlänge) verteilt.

    Eine gleichmäßige Verteilung im Winkel (alter Ansatz) häuft Punkte an den
    Spitzen der großen Halbachse und dünnt sie an der kurzen Halbachse aus,
    besonders bei exzentrischen Ellipsen. Hier wird stattdessen zunächst dicht
    im Winkel abgetastet, die kumulative Bogenlänge bestimmt und dann `n`
    Punkte mit gleichem Bogenlängenabstand daraus interpoliert.
    """
    # Dichte Winkelabtastung zur numerischen Bogenlängen-Approximation.
    n_dense = max(360, n * 20)
    t_dense = np.linspace(0.0, 2.0 * np.pi, n_dense)
    x_dense = ax_e * np.cos(t_dense)
    y_dense = by_e * np.sin(t_dense)

    seg_len = np.hypot(np.diff(x_dense), np.diff(y_dense))
    arc_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc_len[-1]

    # n gleich lange Bogenabschnitte, geschlossene Kurve (kein doppelter Endpunkt nötig).
    target_len = np.linspace(0.0, total_len, n, endpoint=False)
    t_uniform = np.interp(target_len, arc_len, t_dense)

    x_local = ax_e * np.cos(t_uniform)
    y_local = by_e * np.sin(t_uniform)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = x0 + cos_t * x_local - sin_t * y_local
    y = y0 + sin_t * x_local + cos_t * y_local
    return np.column_stack([x, y])

def ellipsoid_surface_points(e, nu=30, nv=10):
    """Oberflächenpunkte unter Berücksichtigung der Orientierung."""
    u = np.linspace(0.0, 2.0 * np.pi, nu)
    v = np.linspace(0.0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)

    local = np.column_stack([
        e["a"] * np.cos(uu).ravel() * np.sin(vv).ravel(),
        e["b"] * np.sin(uu).ravel() * np.sin(vv).ravel(),
        e["c"] * np.cos(vv).ravel(),
    ])

    return local @ _rotation_matrix(e).T + np.asarray(e["x"], dtype=float)