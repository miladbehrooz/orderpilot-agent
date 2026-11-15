"""LangSmith tracing setup.

LangChain and LangGraph read the ``LANGSMITH_*`` values from the process environment.
:class:`~orderpilot.settings.Settings` reads ``.env`` into its own fields only, so
:func:`configure_tracing` copies them into the environment. Every entry point (terminal
chat, ``langgraph dev``, evals) then traces the same way.
"""

import os
from collections.abc import MutableMapping

from orderpilot.logging import logger
from orderpilot.settings import Settings, settings


def configure_tracing(
    config: Settings = settings, environ: MutableMapping[str, str] = os.environ
) -> bool:
    """Turn LangSmith tracing on in the environment if the settings ask for it.

    Does nothing when tracing is off, so a ``LANGSMITH_TRACING`` already set in the shell
    is left alone. When tracing is on but no API key is set, logs a warning and leaves
    tracing off instead of failing.

    Args:
        config: Settings to read the ``LANGSMITH_*`` values from.
        environ: Environment to write to; the process environment by default.

    Returns:
        ``True`` if tracing was turned on.
    """
    if not config.LANGSMITH_TRACING:
        return False
    if config.LANGSMITH_API_KEY is None or not config.LANGSMITH_API_KEY.get_secret_value():
        logger.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is not set; not tracing")
        return False

    environ["LANGSMITH_TRACING"] = "true"
    environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY.get_secret_value()
    environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT
    if config.LANGSMITH_ENDPOINT:
        environ["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT
    logger.info("LangSmith tracing on (project {})", config.LANGSMITH_PROJECT)
    return True


__all__ = ["configure_tracing"]
