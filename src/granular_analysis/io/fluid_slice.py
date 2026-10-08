"""Read PRIME velocity or pressure slices without loading the full 3D field."""

from pathlib import Path

import numpy as np
from granular_analysis.config import D_EQ

from granular_analysis.io.loader import _read_bin_time
from granular_analysis.utils.physics import read_fluid_density, read_gravity


def vortex_criterion(gradient, component):
    """Evaluate Q or the middle eigenvalue of S@S + Omega@Omega."""
    transpose = np.swapaxes(gradient, -1, -2)
    strain = (gradient + transpose) * 0.5
    rotation = (gradient - transpose) * 0.5
    if component == "q":
        return 0.5 * np.sum(rotation**2 - strain**2, axis=(-2, -1))
    if component == "lambda2":
        return np.linalg.eigvalsh(strain @ strain + rotation @ rotation)[..., 1]
    raise ValueError(f"Unknown vortex criterion: {component}")


def spatial_derivative(values, coordinates, axis, period=None):
    """Cell-center derivative, with periodic neighbors or one-sided edges."""
    if len(coordinates) < 2:
        raise ValueError("Vortex criteria require at least two cells per direction")
    if period is not None:
        padded = np.concatenate((np.take(values, [-1], axis=axis), values,
                                 np.take(values, [0], axis=axis)), axis=axis)
        coords = np.r_[coordinates[-1] - period, coordinates, coordinates[0] + period]
        result = np.gradient(padded, coords, axis=axis, edge_order=2)
        return np.take(result, np.arange(1, len(coordinates) + 1), axis=axis)
    return np.gradient(values, coordinates, axis=axis,
                       edge_order=2 if len(coordinates) >= 3 else 1)


def restart_records(results):
    results = Path(results)
    history = results / "history" / "rst_info.dat"
    sources = ([history] if history.is_file() else []) + [results / "rst_info.dat"]
    records = {}
    for source in sources:
        with source.open() as stream:
            columns = next(stream).split()
            for line in stream:
                values = line.split()
                if len(values) == len(columns):
                    row = dict(zip(columns, values))
                    records[int(row["counter"])] = row
    return sorted(records.values(), key=lambda row: float(row["t"]))


def matching_record(records, time):
    matches = [row for row in records
               if np.isclose(float(row["t"]), time, rtol=1e-12, atol=1e-10)]
    if not matches:
        raise ValueError(
            f"No fluid checkpoint at particle time {time:.12g}. "
            "Choose a particle counter at an exact fluid checkpoint time."
        )
    return matches[-1]


def latest_common_counter(case, component="magnitude"):
    prefix = "pre" if component == "pressure" else "uvw"
    records = restart_records(case.results)
    for frame in sorted(case.require_trn().glob("ell_trn_*.bin"), reverse=True):
        try:
            row = matching_record(records, _read_bin_time(frame))
        except ValueError:
            continue
        if (case.results / "fluid" / f"{prefix}_{int(row['counter']):06d}.bin").is_file():
            return int(frame.stem.rsplit("_", 1)[1])
    raise ValueError("No particle frame has an exactly matching fluid checkpoint.")


def nearest_frame(case, time, component=None):
    """Select the nearest particle time, requiring an exact fluid pair if requested."""
    if not np.isfinite(time):
        raise ValueError("Requested time must be finite.")
    rows = restart_records(case.results) if component else []
    prefix = "pre" if component == "pressure" else "uvw"
    rows = [row for row in rows if
            (case.results / "fluid" / f"{prefix}_{int(row['counter']):06d}.bin").is_file()]
    fluid_times = np.array([float(row["t"]) for row in rows])
    candidates = []
    for frame in case.require_trn().glob("ell_trn_*.bin"):
        frame_time = _read_bin_time(frame)
        if component and not np.any(np.isclose(fluid_times, frame_time, rtol=1e-12, atol=1e-10)):
            continue
        candidates.append((frame_time, int(frame.stem.rsplit("_", 1)[1])))
    if not candidates:
        raise ValueError("No particle frame has an exactly matching fluid checkpoint."
                         if component else "No particle frames available.")
    return min(candidates, key=lambda item: (abs(item[0] - time), item[0], item[1]))


def load_fluid_slice(case, particle_counter, z_plane, component="magnitude"):
    component = component.lower()
    if component not in {"magnitude", "u", "v", "w", "pressure", "lambda2", "q"}:
        raise ValueError(f"Unknown fluid component: {component}")
    time = _read_bin_time(case.trn / f"ell_trn_{particle_counter:06d}.bin")
    row = matching_record(restart_records(case.results), time)
    counter = int(row["counter"])
    folder = case.results / "fluid"
    grids = [np.loadtxt(folder / f"{axis}.dat", ndmin=2) for axis in "xyz"]
    starts = [int(row[f"s{axis}"]) for axis in "xyz"]
    counts = [int(row[f"n{axis}"]) for axis in "xyz"]
    sizes = [int(row[f"e{axis}"]) - start + 1
             for axis, start in zip("xyz", starts)]
    if any(grid.shape != (size, 2) for grid, size in zip(grids, sizes)):
        raise ValueError("Fluid grid dimensions do not match the selected checkpoint.")

    pressure = component == "pressure"
    prefix = "pre" if pressure else "uvw"
    dof = 1 if pressure else 3
    path = folder / f"{prefix}_{counter:06d}.bin"
    expected = int(np.prod(sizes)) * dof
    # PETSc binary Vec: big-endian integer header followed by real scalars.
    header = np.fromfile(path, dtype=">i4", count=2)
    offset = 8
    if len(header) != 2 or header[0] != 1211214:
        header = np.fromfile(path, dtype=">i8", count=2)
        offset = 16
    if len(header) != 2 or header[0] != 1211214 or header[1] != expected:
        raise ValueError(f"Unsupported or inconsistent PETSc field header: {path}")
    if path.stat().st_size != offset + expected * 8:
        raise ValueError(f"Incomplete or unsupported real64 field file: {path}")
    field = np.memmap(path, mode="r", dtype=">f8", offset=offset,
                      shape=(sizes[2], sizes[1], sizes[0], dof))
    indices = [np.arange(1, n + 1) - start for n, start in zip(counts, starts)]
    x, y, z = [grid[index, 0] for grid, index in zip(grids, indices)]
    edges = {}
    for axis, grid, index in zip('xy', grids[:2], indices[:2]):
        # PRIME column 1 stores centres, column 2 the right cell faces.
        first = int(index[0])
        left = grid[first-1, 1] if first > 0 else 2*grid[first, 0]-grid[first, 1]
        edges[f'{axis}_edges'] = np.r_[left, grid[index, 1]]
    periodic = [row[f"{axis}per"].upper() in {"T", ".T.", ".TRUE."}
                for axis in "xyz"]

    def previous(index, axis):
        prev = index - 1
        if periodic[axis]:
            prev = np.where(prev < 0, counts[axis] - 1, prev)
        return prev

    def plane(k):
        ix, iy = indices[:2]
        xx, yy = np.meshgrid(ix, iy)
        if pressure:
            return np.asarray(field[k, yy, xx, 0], dtype=float)
        u = (field[k, yy, xx, 0] + field[k, yy, previous(xx, 0), 0]) * 0.5
        v = (field[k, yy, xx, 1] + field[k, previous(yy, 1), xx, 1]) * 0.5
        w = (field[k, yy, xx, 2] + field[previous(k, 2), yy, xx, 2]) * 0.5
        return np.stack((u, v, w), axis=-1)

    def gradient_plane(k):
        center = plane(k)
        gradient = np.empty(center.shape + (3,))
        for axis, coordinates in enumerate((x, y)):
            period = grids[axis][indices[axis][-1], 1] if periodic[axis] else None
            gradient[..., axis] = spatial_derivative(center, coordinates, 1-axis, period)
        if len(z) < 2:
            raise ValueError("Vortex criteria require at least two cells per direction")
        position = int(np.flatnonzero(indices[2] == k)[0])
        if periodic[2]:
            length = grids[2][indices[2][-1], 1]
            neighbors = [(position - 1) % len(z), position, (position + 1) % len(z)]
            coords = np.array([z[neighbors[0]], z[position], z[neighbors[2]]])
            if position == 0:
                coords[0] -= length
            if position == len(z)-1:
                coords[2] += length
            at = 1
        else:
            start = max(0, min(position-1, len(z)-3))
            neighbors = list(range(start, min(start+3, len(z))))
            coords = z[neighbors]
            at = position-start
        # Differentiate the interpolation basis to obtain a small z stencil.
        weights = np.gradient(np.eye(len(coords)), coords, axis=0,
                              edge_order=2 if len(coords) >= 3 else 1)[at]
        gradient[..., 2] = 0
        for neighbor, coefficient in zip(neighbors, weights):
            gradient[..., 2] += coefficient * (center if neighbor == position else plane(indices[2][neighbor]))
        return gradient

    zs = z
    ks = indices[2]
    if periodic[2]:
        length = grids[2][indices[2][-1], 1]
        z_plane = z_plane % length
        zs = np.concatenate(([z[-1] - length], z, [z[0] + length]))
        ks = np.concatenate(([ks[-1]], ks, [ks[0]]))
    elif not z[0] <= z_plane <= z[-1]:
        raise ValueError(f"Slice z={z_plane} is outside fluid cell centers [{z[0]}, {z[-1]}].")
    upper = min(max(int(np.searchsorted(zs, z_plane)), 1), len(zs) - 1)
    weight = (z_plane - zs[upper - 1]) / (zs[upper] - zs[upper - 1])
    if component in {"q", "lambda2"}:
        gradient = gradient_plane(ks[upper - 1])
        gradient *= 1 - weight
        gradient += weight * gradient_plane(ks[upper])
        values = vortex_criterion(gradient, component)
        return {"x": x, "y": y, **edges, "values": values, "component": component,
                "time": time, "counter": counter}
    sampled = (1 - weight) * plane(ks[upper - 1]) + weight * plane(ks[upper])
    if pressure:
        input_path = case.case_root / "input.dat"
        rho_w = read_fluid_density(input_path)
        gravity = read_gravity(input_path)
        if not np.isfinite(gravity) or gravity <= 0:
            raise ValueError("Pressure normalization requires finite, positive gravity magnitude.")
        if not np.isfinite(D_EQ) or D_EQ <= 0:
            raise ValueError("Pressure normalization requires finite, positive D_EQ.")
        values = sampled / (rho_w * gravity * D_EQ)
    else:
        values = (np.linalg.norm(sampled, axis=-1) if component == "magnitude"
                  else sampled[..., "uvw".index(component)])
    return {"x": x, "y": y, **edges, "values": values, "component": component,
            "time": time, "counter": counter}
