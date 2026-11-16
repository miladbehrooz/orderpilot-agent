"""Exception classes of the eval harness.

They stay out of :mod:`orderpilot.errors` because the agent package knows nothing about its
evals. Errors from the agent package (e.g. ``LLMConfigError``) are raised unchanged.
"""


class EvalError(Exception):
    """Base class for expected eval-harness failures."""


class EvalSetupError(EvalError):
    """The eval setup is invalid.

    For example a bad case file, a missing key, or a database that doesn't match the dataset.
    """


__all__ = ["EvalError", "EvalSetupError"]
