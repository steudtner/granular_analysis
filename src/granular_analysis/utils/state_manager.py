import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StateManager:
    """Read/write analysis results from JSON state file."""

    def __init__(self, results_dir: str):
        self.state_file = Path(results_dir) / "analysis_state.json"

    def read(self, pipeline: str) -> Optional[Dict[str, Any]]:
        """Load cached results for a pipeline. Returns None if not cached."""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            return state.get(pipeline)
        except (json.JSONDecodeError, IOError):
            return None

    def write(self, pipeline: str, data: Dict[str, Any]) -> None:
        """Write results for a pipeline to the state file."""
        # Load existing state
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
            except json.JSONDecodeError:
                state = {}
        else:
            state = {}

        # Update pipeline entry
        state[pipeline] = {
            **data,
            "timestamp": datetime.now().isoformat(),
        }

        # Write back
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".json.tmp")
        with open(temporary, "w") as f:
            json.dump(state, f, indent=2)
        temporary.replace(self.state_file)

    def clear(self, pipeline: Optional[str] = None) -> None:
        """Remove cached results. If pipeline is None, clear all."""
        if not self.state_file.exists():
            return
        if pipeline is None:
            self.state_file.unlink()
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            state.pop(pipeline, None)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except (json.JSONDecodeError, IOError):
            pass

    def has_result(self, pipeline: str) -> bool:
        """Check if a pipeline has cached results."""
        return self.read(pipeline) is not None


def build_fingerprint(**values: Any) -> str:
    """Return a stable cache key for input paths and analysis parameters."""
    payload = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_state(
    pipeline: str,
    results_dir: str,
    fingerprint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    state = StateManager(results_dir).read(pipeline)
    if state is None:
        return None
    if fingerprint is not None and state.get("fingerprint") != fingerprint:
        return None
    return state


def save_state(
    pipeline: str,
    data: Dict[str, Any],
    results_dir: str,
    fingerprint: Optional[str] = None,
) -> None:
    payload = dict(data)
    if fingerprint is not None:
        payload["fingerprint"] = fingerprint
    StateManager(results_dir).write(pipeline, payload)


def has_state(pipeline: str, results_dir: str) -> bool:
    return StateManager(results_dir).has_result(pipeline)
