"""Alignment of instantaneous translational velocity and undirected axes."""

import numpy as np
import pandas as pd

from granular_analysis.geometry.ellipsoid import _rotation_matrix


def particle_alignment(particles, mode="prolate", min_speed=0.0):
    """Return one row per input particle; undefined angles remain NaN.

    Prolate: acos(abs(v_hat dot longest_axis)).
    Oblate: acos(abs(v_hat dot c_axis)) in degrees.
    Axis signs are equivalent for an ellipsoid. Particle numbers are input
    row numbers (one-based), not physical particle IDs.
    """
    if mode not in ("prolate", "oblate"):
        raise ValueError("mode must be prolate or oblate")
    if not np.isfinite(min_speed) or min_speed < 0:
        raise ValueError("min_speed must be finite and nonnegative")
    rows = []
    for number, particle in enumerate(particles, 1):
        velocity = np.asarray(particle["u"], dtype=float)
        axes = np.array([particle[k] for k in ("a", "b", "c")], dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError(f"Invalid velocity for particle row {number}")
        if not np.isfinite(axes).all() or np.any(axes <= 0):
            raise ValueError(f"Invalid semiaxes for particle row {number}")
        index = int(np.argmax(axes)) if mode == "prolate" else 2
        oriented = dict(particle)
        if "R" not in oriented:
            q = np.array(particle.get("q", [1, 0, 0, 0]), dtype=float, copy=True)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
                raise ValueError(f"Invalid quaternion for particle row {number}")
            oriented["q"] = q
        rotation = _rotation_matrix(oriented)
        if not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError(f"Invalid rotation for particle row {number}")
        direction = rotation[:, index]
        speed = float(np.linalg.norm(velocity))
        status = "valid"
        if speed <= min_speed:
            status = "stationary"
        elif mode == "prolate" and np.count_nonzero(np.isclose(axes, axes[index], rtol=1e-10, atol=0)) != 1:
            status = "ambiguous_longest_axis"
        angle = np.nan
        if status == "valid":
            cosine = np.clip(abs(np.dot(velocity / speed, direction)), 0, 1)
            angle = float(np.degrees(np.arccos(cosine)))
        rows.append(dict(particle_row=number, time=particle.get("t", np.nan),
                         mode=mode, axis="abc"[index], speed=speed,
                         axis_x=direction[0], axis_y=direction[1], axis_z=direction[2],
                         angle_deg=angle, status=status))
    return pd.DataFrame(rows, columns=["particle_row", "time", "mode", "axis", "speed",
                                      "axis_x", "axis_y", "axis_z", "angle_deg", "status"])


def enrich_alignment(table, particles, masses=None, eps=0.3, min_samples=20):
    """Cluster standardized (angle in degrees, linear speed); -2 excluded, -1 noise."""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    if not np.isfinite(eps) or eps <= 0 or min_samples < 2:
        raise ValueError("Cluster eps must be positive and min_samples >= 2")
    table = table.copy()
    table["abs_cos_angle"] = np.cos(np.radians(table.angle_deg)).clip(0, 1)
    table["v_axis"] = table.speed * table.abs_cos_angle
    table["v_plane"] = table.speed * np.sqrt((1 - table.abs_cos_angle**2).clip(0, 1))
    for index, name in enumerate(("x", "y", "z")):
        table[name] = [p.get("x", [np.nan]*3)[index] for p in particles]
    table["kinetic_energy"] = np.nan
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        if masses.shape != (len(table),) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("Particle masses must be positive and match TRN input rows")
        table["kinetic_energy"] = 0.5 * masses * table.speed**2
    table["cluster"] = -2
    valid = table.status == "valid"
    metadata = dict(method="DBSCAN", features=["angle_deg", "speed"],
                    eps=eps, min_samples=min_samples, excluded_label=-2, noise_label=-1)
    if valid.any():
        features = table.loc[valid, ["angle_deg", "speed"]].to_numpy()
        scaler = StandardScaler()
        normalized = scaler.fit_transform(features)
        table.loc[valid, "cluster"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(normalized)
        metadata.update(feature_mean=scaler.mean_.tolist(), feature_scale=scaler.scale_.tolist())
    return table, metadata
