"""Continuous piecewise-linear models for granular surface profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution


@dataclass(frozen=True)
class SlopeModel:
    """One fitted continuous piecewise-linear model."""

    name: str
    breakpoints: np.ndarray
    slopes: np.ndarray
    intercept: float
    aic: float
    bic: float
    rss: float

    @property
    def n_breakpoints(self) -> int:
        return int(self.breakpoints.size)

    @property
    def angles_deg(self) -> np.ndarray:
        return np.degrees(np.arctan(self.slopes))

    def predict(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        result = self.intercept + self.slopes[0] * values
        for index, breakpoint in enumerate(self.breakpoints):
            slope_change = self.slopes[index + 1] - self.slopes[index]
            result = result + slope_change * np.maximum(0.0, values - breakpoint)
        return result


def _clean_profile(x, z) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float).reshape(-1)
    z_values = np.asarray(z, dtype=float).reshape(-1)
    if x_values.size != z_values.size:
        raise ValueError("x and z must contain the same number of values")
    finite = np.isfinite(x_values) & np.isfinite(z_values)
    x_values, z_values = x_values[finite], z_values[finite]
    order = np.argsort(x_values, kind="stable")
    x_values, z_values = x_values[order], z_values[order]
    if x_values.size < 3:
        raise ValueError("At least three finite profile points are required")
    if np.ptp(x_values) <= 0:
        raise ValueError("Profile x coordinates must span a non-zero interval")
    return x_values, z_values


def _design_matrix(x: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    columns = [np.ones_like(x), x]
    columns.extend(np.maximum(0.0, x - point) for point in breakpoints)
    return np.column_stack(columns)


def _fit_coefficients(x, z, breakpoints):
    matrix = _design_matrix(x, breakpoints)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, z, rcond=None)
    residuals = z - matrix @ coefficients
    return coefficients, float(residuals @ residuals)


def _information_criteria(rss: float, n: int, n_breakpoints: int):
    # Linear coefficients, breakpoint locations, and residual variance.
    n_parameters = 3 + 2 * n_breakpoints
    safe_rss = max(rss, np.finfo(float).tiny)
    minus_two_log_likelihood = n * (
        np.log(2.0 * np.pi) + 1.0 + np.log(safe_rss / n)
    )
    return (
        float(minus_two_log_likelihood + 2 * n_parameters),
        float(minus_two_log_likelihood + np.log(n) * n_parameters),
    )


def fit_slope_model(x, z, n_breakpoints: int = 0) -> SlopeModel:
    """Fit a continuous linear-spline model with fixed breakpoint count."""
    x_values, z_values = _clean_profile(x, z)
    if n_breakpoints < 0:
        raise ValueError("n_breakpoints cannot be negative")
    if x_values.size < 2 * n_breakpoints + 3:
        raise ValueError(
            f"At least {2 * n_breakpoints + 3} points are required for "
            f"{n_breakpoints} breakpoints"
        )

    if n_breakpoints:
        unique_x = np.unique(x_values)
        margin_index = min(max(1, n_breakpoints), unique_x.size - 2)
        lower = float(unique_x[margin_index])
        upper = float(unique_x[-margin_index - 1])
        if lower >= upper:
            raise ValueError("Profile does not contain enough distinct x coordinates")
        min_spacing = (upper - lower) / max(20.0, 4.0 * (n_breakpoints + 1))

        def objective(points):
            ordered = np.sort(points)
            separation = np.diff(ordered)
            penalty = np.maximum(0.0, min_spacing - separation).sum()
            _, rss = _fit_coefficients(x_values, z_values, ordered)
            return rss + penalty * (np.var(z_values) * x_values.size + 1.0) * 1e3

        result = differential_evolution(
            objective,
            bounds=[(lower, upper)] * n_breakpoints,
            seed=0,
            polish=True,
            tol=1e-8,
            updating="immediate",
        )
        breakpoints = np.sort(result.x)
    else:
        breakpoints = np.empty(0, dtype=float)

    coefficients, rss = _fit_coefficients(x_values, z_values, breakpoints)
    slopes = np.r_[coefficients[1], coefficients[1] + np.cumsum(coefficients[2:])]
    aic, bic = _information_criteria(rss, x_values.size, n_breakpoints)
    name = "Linear" if n_breakpoints == 0 else f"{n_breakpoints} breakpoint"
    if n_breakpoints != 1:
        name += "s" if n_breakpoints else ""
    return SlopeModel(
        name=name,
        breakpoints=breakpoints,
        slopes=slopes,
        intercept=float(coefficients[0]),
        aic=aic,
        bic=bic,
        rss=rss,
    )


def fit_slope_models(x, z, max_breakpoints: int = 3) -> list[SlopeModel]:
    """Fit the linear model and every breakpoint count through the maximum."""
    if max_breakpoints < 0:
        raise ValueError("max_breakpoints cannot be negative")
    return [fit_slope_model(x, z, count) for count in range(max_breakpoints + 1)]
