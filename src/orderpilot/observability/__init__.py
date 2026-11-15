"""Observability: the project logger and LangSmith tracing setup.

Import from the package, e.g. ``from orderpilot.observability import logger``.
"""

from orderpilot.observability.log import logger, setup_logger
from orderpilot.observability.tracing import configure_tracing

__all__ = ["configure_tracing", "logger", "setup_logger"]
