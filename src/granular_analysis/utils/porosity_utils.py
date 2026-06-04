"""Shared porosity formula used by 2D and 3D pipelines."""
import numpy as np


def compute_porosity(V_outer: float, V_inner: float) -> float:
    """Return (V_outer - V_inner) / V_outer, or NaN on invalid input."""
    if V_outer <= 0 or not np.isfinite(V_outer):
        return np.nan
    return (V_outer - V_inner) / V_outer
