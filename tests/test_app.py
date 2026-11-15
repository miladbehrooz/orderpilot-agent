"""Composition root: settings checks and wiring (no network, stub repository)."""

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from orderpilot.app import build_agent
from orderpilot.errors import LLMConfigError
from orderpilot.settings import settings
from tests.stubs import StubRepository


def test_build_agent_requires_provider_and_model(
    mocker: MockerFixture, stub_repo: StubRepository
) -> None:
    """Building the agent without LLM_PROVIDER fails with a clear message.

    Args:
        mocker: Pytest-mock fixture.
        stub_repo: Stub repository.
    """
    mocker.patch.object(settings, "LLM_PROVIDER", None)
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER"):
        build_agent(repo=stub_repo)


def test_build_agent_with_anthropic_and_no_temperature(
    mocker: MockerFixture, stub_repo: StubRepository
) -> None:
    """The agent builds for Claude Sonnet 5.5 without sending a temperature.

    Args:
        mocker: Pytest-mock fixture.
        stub_repo: Stub repository.
    """
    mocker.patch.object(settings, "LLM_PROVIDER", "anthropic")
    mocker.patch.object(settings, "LLM_MODEL", "claude-sonnet-5-5")
    mocker.patch.object(settings, "LLM_TEMPERATURE", None)
    mocker.patch.object(settings, "ANTHROPIC_API_KEY", SecretStr("sk-ant-test"))
    graph = build_agent(repo=stub_repo)
    assert {"agent", "tools", "locked_reply"} <= set(graph.get_graph().nodes)
