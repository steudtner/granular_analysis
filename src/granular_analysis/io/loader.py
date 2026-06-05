"""I/O utilities for loading ellipsoid simulation data."""

from pathlib import Path
from typing import Any, List, Sequence, Union

from granular_analysis.read_ellipsoid_data import read_ellipsoid_trn


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

    files = sorted(folder.glob("*.bin"))
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

    files = sorted(p.name for p in folder.glob("*.bin"))
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

    files = sorted(folder.glob("*.bin"))
    if not files:
        raise FileNotFoundError(
            f"No .bin files found in: {folder.resolve()}"
        )
    return files


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