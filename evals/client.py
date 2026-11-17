"""The LangSmith client used by the eval harness."""

from langsmith import Client

from evals.errors import EvalSetupError
from orderpilot.settings import settings


def make_client() -> Client:
    """Build a LangSmith client from the project settings.

    Uses the same key and endpoint as tracing, so evals and traces land in the same
    LangSmith workspace and region.

    Returns:
        The LangSmith client.

    Raises:
        EvalSetupError: If ``LANGSMITH_API_KEY`` is not set.
    """
    key = settings.LANGSMITH_API_KEY
    if key is None or not key.get_secret_value():
        raise EvalSetupError("LANGSMITH_API_KEY is not set; add it to .env")
    return Client(api_key=key.get_secret_value(), api_url=settings.LANGSMITH_ENDPOINT)


__all__ = ["make_client"]
