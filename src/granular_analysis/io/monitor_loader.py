"""Readers for per-ellipsoid text monitor files."""

import re
from pathlib import Path

import numpy as np


_MONITOR_PATTERN = re.compile(r"ell_(\d+)_monitor\.dat$")


def _stitch_restart_segments(data):
    """Replace an older overlapping tail when a monitor file restarts."""
    while True:
        decreases = np.flatnonzero(np.diff(data[:, 1]) <= 0)
        if len(decreases) == 0:
            return data
        restart_index = decreases[0] + 1
        restart_time = data[restart_index, 1]
        previous_times = data[:restart_index, 1]
        keep = np.searchsorted(previous_times, restart_time, side="left")
        data = np.concatenate((data[:keep], data[restart_index:]), axis=0)


def get_monitor_files(folder):
    """Return monitor files ordered by their numeric ellipsoid ID."""
    folder = Path(folder)
    files = []
    for path in folder.glob("ell_*_monitor.dat"):
        match = _MONITOR_PATTERN.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(f"No ellipsoid monitor files found in: {folder}")
    return files


def read_monitor_file(path):
    """Read one monitor file as an array with 11 documented columns."""
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 11:
        raise ValueError(
            f"Expected 11 columns in {path}, found {data.shape[1]}"
        )
    return _stitch_restart_segments(data)