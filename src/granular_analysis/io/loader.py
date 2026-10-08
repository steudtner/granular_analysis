"""I/O utilities for loading ellipsoid simulation data."""

from pathlib import Path
from typing import Any, List, Sequence, Union

from granular_analysis.io.read_ellipsoid_data import read_ellipsoid_trn


def _trn_files(folder: Path) -> List[Path]:
    """Return exact TRN frame files sorted by numeric counter."""
    files = []
    for path in folder.glob("ell_trn_*.bin"):
        suffix = path.stem.removeprefix("ell_trn_")
        if suffix.isdigit():
            files.append((int(suffix), path))
    return [path for _, path in sorted(files)]


def load_last_frame(folder: Union[str, Path]) -> Any:
    """Load the last *.bin frame from *folder*, sorted alphabetically.
    
    Args:
        folder: Directory containing one or more ``*.bin`` files.
        
    Returns:
        Parsed ellipsoid data (via :func:`read_ellipsoid_trn`).
        
    Raises:
        NotADirectoryError: If *folder* does not exist or is not a directory.
        FileNotFoundError: If the directory contains no ``*.bin`` files.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder.resolve()}")

    files = _trn_files(folder)
    if not files:
        raise FileNotFoundError(
            f"No .bin files found in: {folder.resolve()}"
        )

    return read_ellipsoid_trn(str(files[-1]))


def get_all_bin_files(folder: Union[str, Path]) -> List[str]:
    """Return sorted basenames of all ``*.bin`` files in *folder*.
    
    This preserves the legacy string-only interface.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder.resolve()}")

    files = [p.name for p in _trn_files(folder)]
    if not files:
        raise FileNotFoundError(
            f"No .bin files found in: {folder.resolve()}"
        )
    return files


def get_all_bin_paths(folder: Union[str, Path]) -> List[Path]:
    """Return sorted :class:`~pathlib.Path` objects of all ``*.bin`` files."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder.resolve()}")

    files = _trn_files(folder)
    if not files:
        raise FileNotFoundError(
            f"No .bin files found in: {folder.resolve()}"
        )
    return files


def load_frame(folder: Union[str, Path], frame: Union[None, int, str] = None) -> Any:
    """Load a specific ``*.bin`` frame from *folder*.

    - If *frame* is ``None``, returns the last frame (legacy behavior).
    - If *frame* is an ``int``, selects the frame by index from the sorted list
      (0-based). Negative indices are supported (like Python lists).
    - If *frame* is a ``str`` and matches a filename in *folder*, that file
      is loaded.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder.resolve()}")

    files = _trn_files(folder)
    if not files:
        raise FileNotFoundError(f"No .bin files found in: {folder.resolve()}")

    # None -> last
    if frame is None:
        target = files[-1]
    elif isinstance(frame, int):
        idx = frame
        # support negative indices
        if idx < 0:
            idx = len(files) + idx
        if idx < 0 or idx >= len(files):
            raise IndexError(f"Frame index {frame} out of range (0..{len(files)-1})")
        target = files[idx]
    else:
        # string: try exact match or basename
        s = str(frame)
        # try absolute path
        p = Path(s)
        if p.is_file():
            target = p
        else:
            # match by name
            matches = [f for f in files if f.name == s]
            if matches:
                target = matches[0]
            else:
                raise FileNotFoundError(f"No matching .bin file for '{s}' in {folder}")

    return read_ellipsoid_trn(str(target))


def _read_bin_time(fname: Union[str, Path]) -> float:
    """Read only the time value from a trn ``.bin`` file without parsing all ellipsoids."""
    from scipy.io import FortranFile

    with FortranFile(str(fname), 'r') as f:
        try:
            _ = f.read_record('i4')  # Nell
            t = f.read_record('f8')[0]
            return float(t)
        except Exception as exc:
            raise RuntimeError(f"Failed to read time from {fname}: {exc}")


def load_frame_by_time(folder: Union[str, Path], time_value: float, mode: str = 'nearest') -> Any:
    """Load the frame whose timestamp best matches *time_value*.

    - *mode* in {'nearest', 'before', 'after'} selects matching strategy.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder.resolve()}")

    files = _trn_files(folder)
    if not files:
        raise FileNotFoundError(f"No .bin files found in: {folder.resolve()}")

    times = []
    for f in files:
        try:
            times.append((_read_bin_time(f), f))
        except Exception:
            # skip unreadable files
            continue

    if not times:
        raise RuntimeError(f"No readable .bin times in: {folder}")

    # sort by time
    times.sort(key=lambda x: x[0])
    ts = [t for t, _ in times]

    import bisect

    pos = bisect.bisect_left(ts, time_value)
    choice = None
    if mode == 'nearest':
        if pos == 0:
            choice = times[0][1]
        elif pos == len(ts):
            choice = times[-1][1]
        else:
            before = times[pos - 1]
            after = times[pos]
            choice = before[1] if abs(before[0] - time_value) <= abs(after[0] - time_value) else after[1]
    elif mode == 'before':
        if pos == 0:
            raise IndexError(f"No frame before time {time_value}")
        choice = times[pos - 1][1]
    elif mode == 'after':
        if pos == len(ts):
            raise IndexError(f"No frame after time {time_value}")
        choice = times[pos][1]
    else:
        raise ValueError("mode must be one of 'nearest','before','after'")

    return read_ellipsoid_trn(str(choice))


def filter_particles(data: Sequence[Any], *args, skip_first: int = 0) -> Sequence[Any]:
    """Return ellipsoid data, optionally skipping the first *n* records.
    
    New signature (recommended)::
    
        filter_particles(data, skip_first=1000)
    
    Legacy signature (still supported)::
    
        filter_particles(data, True, 1000)   # use_skip=True, skip=1000
    
    Args:
        data: Sequence of ellipsoid records.
        skip_first: Number of leading records to discard.
        
    Returns:
        Filtered sequence.
    """
    if len(args) == 2:
        # Legacy compatibility: (data, use_skip: bool, skip: int)
        use_skip, skip = args
        skip_first = skip if use_skip else 0
    elif args:
        raise TypeError(
            f"filter_particles() takes 1 positional argument plus optional "
            f"skip_first keyword, but {len(args)} extra positional args were given"
        )

    return data[skip_first:]