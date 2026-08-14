"""Logging configuration tests."""

from __future__ import annotations

import logging

import pytest

from chargeopt.utils.log import configure_logging


def test_configure_logging_sets_level() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    configure_logging("WARNING")
    assert root.level == logging.WARNING
    assert root.handlers
    configure_logging("ERROR")
    assert root.level == logging.ERROR


def test_configure_logging_rejects_invalid_level() -> None:
    with pytest.raises(ValueError, match="invalid log level"):
        configure_logging("NOPE")
