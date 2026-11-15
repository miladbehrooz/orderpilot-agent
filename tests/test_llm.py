"""LLMFactory: provider registry, key checks and argument pass-through (no network)."""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from pytest_mock import MockerFixture

from orderpilot.errors import LLMConfigError
from orderpilot.llm import LLMFactory
from orderpilot.settings import settings


@pytest.fixture
def api_keys(mocker: MockerFixture) -> None:
    """Set dummy API keys for both providers.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(settings, "OPENAI_API_KEY", SecretStr("sk-test"))
    mocker.patch.object(settings, "ANTHROPIC_API_KEY", SecretStr("sk-ant-test"))


@pytest.mark.usefixtures("api_keys")
def test_build_openai_passes_model_and_kwargs() -> None:
    """OpenAI builds a ChatOpenAI with the model name and extra arguments."""
    model = LLMFactory.build("openai", model_name="gpt-test", temperature=0.3)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-test"
    assert model.temperature == 0.3


@pytest.mark.usefixtures("api_keys")
def test_build_anthropic_passes_model_and_kwargs() -> None:
    """Anthropic builds a ChatAnthropic with the model name and extra arguments."""
    model = LLMFactory.build("anthropic", model_name="claude-test", temperature=0.3)
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-test"
    assert model.temperature == 0.3


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
)
def test_missing_key_raises(mocker: MockerFixture, provider: str, key_name: str) -> None:
    """A provider without an API key fails with a message naming the variable.

    Args:
        mocker: Pytest-mock fixture.
        provider: Provider under test.
        key_name: Settings field holding its key.
    """
    mocker.patch.object(settings, key_name, None)
    with pytest.raises(LLMConfigError, match=key_name):
        LLMFactory.build(provider, model_name="m")  # type: ignore[arg-type]


def test_unknown_provider_raises() -> None:
    """An unregistered provider fails with the list of valid ones."""
    with pytest.raises(LLMConfigError, match="Valid: openai, anthropic"):
        LLMFactory.build("gemini", model_name="m")  # type: ignore[arg-type]


def test_list_providers() -> None:
    """Both supported providers are registered."""
    assert LLMFactory.list_providers() == ["openai", "anthropic"]
