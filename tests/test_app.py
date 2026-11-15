"""Composition root: settings checks and what build_agent asks the LLM factory for."""

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture, MockType

from orderpilot import app
from orderpilot.app import build_agent
from orderpilot.errors import LLMConfigError
from orderpilot.settings import settings
from tests.stubs import StubRepository


@pytest.fixture
def anthropic_settings(mocker: MockerFixture) -> None:
    """Configure Claude Sonnet 5.5 without a temperature and with a dummy key.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(settings, "LLM_PROVIDER", "anthropic")
    mocker.patch.object(settings, "LLM_MODEL", "claude-sonnet-5-5")
    mocker.patch.object(settings, "LLM_TEMPERATURE", None)
    mocker.patch.object(settings, "ANTHROPIC_API_KEY", SecretStr("sk-ant-test"))


@pytest.fixture
def llm_factory(mocker: MockerFixture) -> MockType:
    """Replace the LLM factory that build_agent uses with an autospec mock.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked ``LLMFactory`` class.
    """
    factory: MockType = mocker.patch.object(app, "LLMFactory", autospec=True)
    return factory


@pytest.mark.parametrize("field", ["LLM_PROVIDER", "LLM_MODEL"])
@pytest.mark.usefixtures("anthropic_settings")
def test_build_agent_requires_provider_and_model(
    mocker: MockerFixture, stub_repo: StubRepository, field: str
) -> None:
    """Building the agent without LLM_PROVIDER or LLM_MODEL fails with a clear message.

    Args:
        mocker: Pytest-mock fixture.
        stub_repo: Stub repository.
        field: The setting left unset.
    """
    mocker.patch.object(settings, field, None)
    with pytest.raises(LLMConfigError, match=field):
        build_agent(repo=stub_repo)


@pytest.mark.usefixtures("anthropic_settings")
def test_build_agent_omits_unset_temperature(
    llm_factory: MockType, stub_repo: StubRepository
) -> None:
    """Without LLM_TEMPERATURE no temperature reaches the model (current Claude models reject it).

    Args:
        llm_factory: The mocked LLM factory.
        stub_repo: Stub repository.
    """
    build_agent(repo=stub_repo)
    llm_factory.build.assert_called_once_with("anthropic", model_name="claude-sonnet-5-5")


@pytest.mark.usefixtures("anthropic_settings")
def test_build_agent_passes_configured_temperature(
    mocker: MockerFixture, llm_factory: MockType, stub_repo: StubRepository
) -> None:
    """A configured LLM_TEMPERATURE is passed on to the model.

    Args:
        mocker: Pytest-mock fixture.
        llm_factory: The mocked LLM factory.
        stub_repo: Stub repository.
    """
    mocker.patch.object(settings, "LLM_TEMPERATURE", 0.2)
    build_agent(repo=stub_repo)
    llm_factory.build.assert_called_once_with(
        "anthropic", model_name="claude-sonnet-5-5", temperature=0.2
    )


@pytest.mark.usefixtures("anthropic_settings")
def test_build_agent_compiles_with_real_model(stub_repo: StubRepository) -> None:
    """With a real (never called) Claude model object, the graph compiles with all its nodes.

    Args:
        stub_repo: Stub repository.
    """
    graph = build_agent(repo=stub_repo)
    assert {"agent", "tools", "locked_reply"} <= set(graph.get_graph().nodes)
