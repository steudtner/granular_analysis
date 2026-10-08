"""Aggregate the library's per-simplex singular-matrix warnings."""

from contextlib import contextmanager
from contextvars import ContextVar
import logging
from pathlib import Path
import threading


_active = ContextVar("alpha_warning_summary", default=False)
_MESSAGE = "Singular matrix. Likely caused by all points lying in an N-1 space."


class _SingularWarningCounter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.thread = threading.get_ident()

    def filter(self, record):
        path = Path(record.pathname)
        if (record.thread == self.thread and record.levelno == logging.WARNING
                and path.name == "alphashape.py" and path.parent.name == "alphashape"
                and record.getMessage() == _MESSAGE):
            self.count += 1
            return False
        return True


@contextmanager
def summarize_alpha_warnings():
    """Keep other diagnostics intact; nest to summarize a complete slice batch."""
    if _active.get():
        yield
        return
    token = _active.set(True)
    counter = _SingularWarningCounter()
    root = logging.getLogger()
    root.addFilter(counter)
    try:
        yield
    finally:
        root.removeFilter(counter)
        _active.reset(token)
        if counter.count:
            logging.getLogger(__name__).warning(
                "Alpha-shape: %d singular simplices skipped by the library "
                "(repeated warnings summarized; geometry may be affected).",
                counter.count,
            )
