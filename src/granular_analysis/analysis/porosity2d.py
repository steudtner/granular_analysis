import os
import numpy as np
import pandas as pd
from granular_analysis.config import RESULTS_DIR
from granular_analysis.geometry.ellipsoid import ellipsoid_slice, ellipse_points
from granular_analysis.geometry.alpha2d import compute_alpha_shape_2d
from granular_analysis.utils.porosity_utils import compute_porosity

def compute_slice_porosity(particles, z0, alpha, lz):
    """Compute 2D slice porosity from a particle set."""
    pts = []
    A_ell = 0.0
    for shift in [-lz, 0.0, lz]:
        for e in particles:
            res = ellipsoid_slice(e, z0, shift)
            if res is None:
                continue
            x0, y0, ax_e, by_e = res
            pts.append(ellipse_points(x0, y0, ax_e, by_e))
            A_ell += np.pi * ax_e * by_e

    if not pts:
        return None

    pts = np.vstack(pts)
    pts = pts[np.isfinite(pts).all(axis=1)]
    shape, A_alpha = compute_alpha_shape_2d(pts, alpha)
    
    # ONLY extract upper edge for the specific z0 we care about (z=2.5)
    if shape is not None and abs(z0 - 2.5) < 1e-6:
        extract_upper_edge(shape)

    return {
        "shape": shape,
        "A_ell": A_ell,
        "A_alpha": A_alpha,
        "porosity": compute_porosity(A_alpha, A_ell),
        "n": len(particles),
    }

def extract_upper_edge(shape_coords, n_bins=50):
    """Extract the upper edge from a closed 2D alpha-shape polygon.
    
    Bins the x-axis into n_bins intervals and takes the maximum y-value
    in each bin to obtain the upper surface profile. 
    Filters out bins where the y-value deviates by more than 1.0 from 
    the previous valid bin.
    """
    x = shape_coords[:, 0]
    y = shape_coords[:, 1]
    
    x_min, x_max = np.min(x), np.max(x)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    indices = np.digitize(x, bins)
    
    x_raw = []
    y_raw = []
    for i in range(1, len(bins)):
        mask = indices == i
        if np.any(mask):
            x_raw.append(np.mean(x[mask]))
            y_raw.append(np.max(y[mask]))
            
    if not x_raw:
        return
        
    # Filter: Skip bins where deviation from previous valid bin > 1.0
    x_out = [x_raw[0]]
    y_out = [y_raw[0]]
    
    for i in range(1, len(x_raw)):
        if abs(y_raw[i] - y_out[-1]) <= 2.0:
            x_out.append(x_raw[i])
            y_out.append(y_raw[i])
        # else: skip this bin, move to the next
        
    df_out = pd.DataFrame({"x": x_out, "z_max": y_out})
    out_path = os.path.join(RESULTS_DIR, "alphashape_last_timestep.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_out.to_csv(out_path, index=False, header=False)
    print(f"Saved upper edge (z0=2.5) to: {out_path}")
