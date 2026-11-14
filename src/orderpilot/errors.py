"""Project-wide exception classes.

Every expected, user-facing failure in OrderPilot derives from :class:`OrderPilotError`,
so callers can catch the whole family with one ``except`` clause. Add new error classes
here rather than defining them inside individual modules.
"""

from __future__ import annotations


class OrderPilotError(Exception):
    """Base class for all expected OrderPilot failures."""


class DatabaseBuildError(OrderPilotError):
    """An expected failure while building the database, reported without a traceback."""


class LLMConfigError(OrderPilotError):
    """The LLM provider or model is misconfigured (unknown provider, missing key or setting)."""


class RepositoryError(OrderPilotError):
    """The data store is unavailable or returned unexpected data."""


__all__ = ["DatabaseBuildError", "LLMConfigError", "OrderPilotError", "RepositoryError"]
