import json
import os
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
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

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


def load_state(pipeline: str, results_dir: str) -> Optional[Dict[str, Any]]:
    return StateManager(results_dir).read(pipeline)


def save_state(pipeline: str, data: Dict[str, Any], results_dir: str) -> None:
    StateManager(results_dir).write(pipeline, data)


def has_state(pipeline: str, results_dir: str) -> bool:
    return StateManager(results_dir).has_result(pipeline)
