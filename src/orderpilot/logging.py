"""Project-wide Loguru logger.

Import the configured logger with ``from orderpilot.logging import logger``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

from orderpilot.settings import settings

if TYPE_CHECKING:
    from loguru import Logger  # only exists in loguru's type stubs


def setup_logger() -> Logger:
    """Configure the global Loguru logger for console output.

    Removes Loguru's default handler and adds a stdout handler at ``settings.LOG_LEVEL``.

    Returns:
        The configured Loguru logger.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        enqueue=True,
    )
    return logger


setup_logger()

__all__ = ["logger", "setup_logger"]
