"""
Tests for tracker/logger.py

We test behavior, not implementation:
- setup_logging() runs without error
- get_logger() returns a callable logger
- Log file is created at the configured path
- Module name appears correctly in output
"""

from __future__ import annotations

from pathlib import Path

from tracker.config import LoggingConfig
from tracker.logger import get_logger, setup_logging


def test_setup_logging_creates_log_file(tmp_path: Path) -> None:
    """setup_logging() must create the log file and its parent directories."""
    log_file = tmp_path / "subdir" / "test.log"
    config = LoggingConfig(level="DEBUG", file=str(log_file))

    setup_logging(config)

    # Write something so the file is flushed
    logger = get_logger("test")
    logger.info("test message")

    assert log_file.exists(), f"Log file was not created at {log_file}"


def test_get_logger_returns_callable(tmp_path: Path) -> None:
    """get_logger() must return something we can call .info() on."""
    config = LoggingConfig(file=str(tmp_path / "test.log"))
    setup_logging(config)

    logger = get_logger("test_module")

    # These must not raise
    logger.info("info message")
    logger.warning("warning message")
    logger.debug("debug message")
    logger.error("error message")


def test_log_file_contains_message(tmp_path: Path) -> None:
    """Messages logged must actually appear in the log file."""
    log_file = tmp_path / "test.log"
    config = LoggingConfig(level="DEBUG", file=str(log_file))
    setup_logging(config)

    logger = get_logger("test_module")
    logger.info("hello from test")

    content = log_file.read_text(encoding="utf-8")
    assert "hello from test" in content


def test_log_file_contains_module_name(tmp_path: Path) -> None:
    """The module name passed to get_logger must appear in the log file."""
    log_file = tmp_path / "test.log"
    config = LoggingConfig(level="DEBUG", file=str(log_file))
    setup_logging(config)

    logger = get_logger("my_special_module")
    logger.info("checking module name")

    content = log_file.read_text(encoding="utf-8")
    assert "my_special_module" in content