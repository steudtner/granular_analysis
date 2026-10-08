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
    """Return monitor files ordered by their one-based input particle row."""
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


def monitor_source_paths(path):
    """Return an optional immutable history segment followed by the live file."""
    path = Path(path)
    history = path.parent / "history" / path.name
    return [history, path] if history.is_file() else [path]


def read_monitor_file(path):
    """Read the complete history, with current data winning at a restart."""
    segments = []
    for source in monitor_source_paths(path):
        data = np.loadtxt(source, ndmin=2)
        if data.shape[1] != 11:
            raise ValueError(
                f"Expected 11 columns in {source}, found {data.shape[1]}"
            )
        segments.append(data)
    return _stitch_restart_segments(np.concatenate(segments, axis=0))
