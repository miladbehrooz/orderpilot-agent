"""configure_tracing: copying LangSmith settings into the environment."""

from pydantic import SecretStr

from orderpilot.observability.tracing import configure_tracing
from orderpilot.settings import Settings


def _settings(**overrides: object) -> Settings:
    """Return settings that ignore .env and the shell, with the given overrides.

    Args:
        **overrides: Field values, e.g. ``LANGSMITH_TRACING=True``.

    Returns:
        The settings.
    """
    return Settings.model_construct(**{**Settings().model_dump(), **overrides})


def test_tracing_off_leaves_environment_untouched() -> None:
    """With tracing off, nothing is written."""
    environ: dict[str, str] = {}
    assert configure_tracing(_settings(LANGSMITH_TRACING=False), environ) is False
    assert environ == {}


def test_tracing_on_without_key_stays_off() -> None:
    """Tracing on without an API key logs a warning and writes nothing."""
    environ: dict[str, str] = {}
    config = _settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY=None)
    assert configure_tracing(config, environ) is False
    assert environ == {}


def test_tracing_on_exports_langsmith_values() -> None:
    """Tracing on exports the flag, key and project for LangChain to read."""
    environ: dict[str, str] = {}
    config = _settings(
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY=SecretStr("lsv2-test"),
        LANGSMITH_PROJECT="orderpilot-test",
        LANGSMITH_ENDPOINT=None,
    )
    assert configure_tracing(config, environ) is True
    assert environ == {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "lsv2-test",
        "LANGSMITH_PROJECT": "orderpilot-test",
    }


def test_tracing_exports_endpoint_when_set() -> None:
    """A non-default region endpoint is exported too."""
    environ: dict[str, str] = {}
    config = _settings(
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY=SecretStr("lsv2-test"),
        LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com",
    )
    configure_tracing(config, environ)
    assert environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"
