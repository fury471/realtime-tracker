"""
tracker/logger.py

Centralised logging setup using Loguru.

Usage anywhere in the project:
    from tracker.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Camera opened")
    logger.debug(f"Frame shape: {frame.shape}")
    logger.warning("Low confidence detection ignored")
    logger.error("Camera device not found")

Design decisions:
- get_logger(__name__) passes the module name so every log line shows
  exactly which file produced it — essential for debugging a pipeline
- setup_logging() is called ONCE at application startup (in pipeline.py)
- All other modules just call get_logger() — they never configure handlers
- encoding='utf-8' on file sink — Windows GBK safety
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from tracker.config import LoggingConfig


def _ensure_module(record: dict) -> bool:
    """
    Filter that guarantees {extra[module]} always exists.
    Falls back to {name} (the Python module path) if not bound.
    This prevents KeyError when logger.info() is called
    outside of get_logger() — e.g. inside setup_logging() itself.
    """
    record["extra"].setdefault("module", record["name"])
    return True


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure logging for the entire application.

    Call this ONCE at startup before anything else runs.
    All modules that call get_logger() will inherit this configuration.

    Args:
        config: LoggingConfig section from the main Config object.
    """
    # Remove Loguru's default handler (it has no file output or custom format)
    logger.remove()

    # Console handler — colored, human-readable
    # Format breakdown:
    #   {time:HH:mm:ss} — timestamp (short, no date clutter)
    #   {level:<8}      — log level padded to 8 chars (INFO    / WARNING )
    #   {name}          — module name (__name__ from get_logger)
    #   {line}          — line number
    #   {message}       — the actual log message
    logger.add(
        sys.stderr,
        level=config.level,
        filter=_ensure_module,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{extra[module]}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — plain text, auto-rotates, auto-deletes old files
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path,
        level=config.level,
        filter=_ensure_module,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {extra[module]}:{line} - {message}",
        rotation=config.rotation,    # new file after 10 MB
        retention=config.retention,  # delete files older than 7 days
        encoding="utf-8",            # explicit — never rely on Windows default
    )

    logger.info(f"Logging initialised — level={config.level}, file={log_path}")


def get_logger(name: str):
    """
    Get a named logger for a module.

    Args:
        name: Module name — always pass __name__ here.

    Returns:
        Loguru logger bound to the module name.

    Example:
        logger = get_logger(__name__)
        logger.info("Starting camera capture")
    """
    return logger.opt(depth=0).bind(module=name)