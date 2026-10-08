import logging

import pytest

from granular_analysis.geometry.alpha_logging import summarize_alpha_warnings


def emit(message, pathname="/site-packages/alphashape/alphashape.py"):
    record = logging.LogRecord("root", logging.WARNING, pathname, 80, message, (), None)
    logging.getLogger().handle(record)


MESSAGE = "Singular matrix. Likely caused by all points lying in an N-1 space."


def test_nested_warnings_are_counted_once(caplog):
    with summarize_alpha_warnings():
        for _ in range(3):
            with summarize_alpha_warnings():
                for _ in range(10):
                    emit(MESSAGE)
        emit("Another warning")
        emit(MESSAGE, "/other/module.py")
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert "Another warning" in messages
    assert MESSAGE in messages
    assert "30 singular simplices" in messages[-1]


def test_filter_restored_after_failure(caplog):
    before = list(logging.getLogger().filters)
    with pytest.raises(RuntimeError):
        with summarize_alpha_warnings():
            emit(MESSAGE)
            raise RuntimeError("failure")
    assert logging.getLogger().filters == before
    emit(MESSAGE)
    assert caplog.records[-1].getMessage() == MESSAGE
