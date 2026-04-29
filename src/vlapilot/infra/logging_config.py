"""Logging configuration for VLA Agent."""

from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    debug: bool = False
):
    """Configure logging for VLA Agent.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        debug: Enable debug mode (more verbose)
    """
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stderr,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="DEBUG" if debug else level
    )

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            rotation="10 MB",
            retention="7 days",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} - "
                "{message}"
            ),
            level="DEBUG" if debug else level
        )

    logger.info(f"Logging configured: level={level}, debug={debug}, file={log_file}")


def get_logger(name: str):
    """Get a logger instance."""
    return logger.bind(name=name)
