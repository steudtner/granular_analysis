"""Canonical filesystem layout for one granular simulation case."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _configured_results(case_root: Path) -> Path:
    """Read the solver's output directory without changing simulation inputs."""
    input_file = case_root / "input.dat"
    if input_file.is_file():
        match = re.search(
            r"^\s*res_dir\s*=\s*('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")",
            input_file.read_text(),
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if match:
            quoted = match.group(1)
            value = quoted[1:-1].replace(quoted[0] * 2, quoted[0]).strip()
            if value:
                return (case_root / value).resolve()
    return case_root / "results"


@dataclass(frozen=True)
class CasePaths:
    """Resolved directories belonging to one simulation case.

    ``from_input`` accepts a case root, ``results`` directory, ellipsoid
    directory, or one of the ellipsoid data subdirectories.
    """

    case_root: Path
    results: Path
    ellipsoid: Path
    trn: Path
    coll: Path
    monitor: Path

    @classmethod
    def from_input(cls, value: Optional[os.PathLike | str] = None) -> "CasePaths":
        if value is None:
            value = os.environ.get("GRANULAR_DATA_FOLDER")
        path = Path(value or Path.cwd()).expanduser().resolve()

        if path.name in {"ellipsoid_trn", "ellipsoid_coll", "ellipsoid_monitor"}:
            ellipsoid = path.parent
            results = ellipsoid.parent
            case_root = results.parent
        elif path.name == "ellipsoid":
            ellipsoid = path
            results = path.parent
            case_root = results.parent
        elif path.name == "results":
            results = path
            case_root = path.parent
            ellipsoid = results / "ellipsoid"
        elif (path / "results").is_dir() or not (path / "ellipsoid").is_dir():
            case_root = path
            results = _configured_results(case_root)
            ellipsoid = results / "ellipsoid"
        else:
            # An unnamed results-like directory containing ``ellipsoid``.
            results = path
            case_root = path.parent
            ellipsoid = results / "ellipsoid"

        return cls(
            case_root=case_root,
            results=results,
            ellipsoid=ellipsoid,
            trn=ellipsoid / "ellipsoid_trn",
            coll=ellipsoid / "ellipsoid_coll",
            monitor=ellipsoid / "ellipsoid_monitor",
        )

    def require_trn(self) -> Path:
        """Return the TRN directory after validating that it has TRN files."""
        if not self.trn.is_dir():
            raise FileNotFoundError(f"TRN directory not found: {self.trn}")
        if not any(self.trn.glob("ell_trn_*.bin")):
            raise FileNotFoundError(f"No ell_trn_*.bin files found in: {self.trn}")
        return self.trn


def output_directory(value: Optional[os.PathLike | str] = None) -> Path:
    """Resolve the analysis output directory at call time."""
    configured = value or os.environ.get("GRANULAR_RESULTS_DIR")
    return Path(
        configured or (Path.cwd() / "results_granular_analysis")
    ).expanduser().resolve()


def counters(paths: Iterable[Path], prefix: str) -> set[int]:
    """Extract numeric counters from files named ``<prefix>XXXXXX*.bin``."""
    result = set()
    for path in paths:
        name = path.name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):].split(".", 1)[0].split("_", 1)[0]
        if suffix.isdigit():
            result.add(int(suffix))
    return result


def latest_trn_counter(case: CasePaths) -> Optional[int]:
    values = counters(case.trn.glob("ell_trn_*.bin"), "ell_trn_")
    return max(values) if values else None


def collision_rk_stages(case: CasePaths) -> tuple[int, ...]:
    """Detect single-stage output; mixed output still requires all RK stages."""
    stages = set()
    for path in case.coll.glob("ell_coll_*.bin"):
        match = re.fullmatch(r"ell_coll_\d+_([123])\.bin", path.name)
        if match:
            stages.add(int(match.group(1)))
    return (1,) if stages == {1} else (1, 2, 3)


def latest_common_collision_counter(case: CasePaths) -> Optional[int]:
    """Return the latest TRN counter with all stages required by this case."""
    trn = counters(case.trn.glob("ell_trn_*.bin"), "ell_trn_")
    stages = {}
    for path in case.coll.glob("ell_coll_*.bin"):
        parts = path.stem.removeprefix("ell_coll_").split("_")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            continue
        counter, stage = map(int, parts)
        stages.setdefault(counter, set()).add(stage)
    required = set(collision_rk_stages(case))
    complete_coll = {
        counter for counter, available in stages.items() if required <= available
    }
    common = trn & complete_coll
    return max(common) if common else None
