"""The LLM judge: what it asks the LLM factory for, and when it warns."""

import pytest
from pytest_mock import MockerFixture, MockType

from evals import judge
from evals.settings import eval_settings
from orderpilot.errors import LLMConfigError
from orderpilot.settings import settings


@pytest.fixture
def judge_settings(mocker: MockerFixture) -> None:
    """Configure an OpenAI judge that differs from an Anthropic agent, with no temperature.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(eval_settings, "JUDGE_PROVIDER", "openai")
    mocker.patch.object(eval_settings, "JUDGE_MODEL", "gpt-judge")
    mocker.patch.object(eval_settings, "JUDGE_TEMPERATURE", None)
    mocker.patch.object(settings, "LLM_PROVIDER", "anthropic")
    mocker.patch.object(settings, "LLM_MODEL", "claude-agent")


@pytest.fixture
def llm_factory(mocker: MockerFixture) -> MockType:
    """Replace the LLM factory that ``build_judge`` uses with an autospec mock.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked ``LLMFactory`` class.
    """
    factory: MockType = mocker.patch.object(judge, "LLMFactory", autospec=True)
    return factory


@pytest.fixture
def judge_logger(mocker: MockerFixture) -> MockType:
    """Replace the logger used by the judge module.

    Args:
        mocker: Pytest-mock fixture.

    Returns:
        The mocked logger.
    """
    patched: MockType = mocker.patch.object(judge, "logger", autospec=True)
    return patched


@pytest.mark.usefixtures("judge_settings")
def test_build_judge_requires_model(mocker: MockerFixture) -> None:
    """Without EVAL_JUDGE_MODEL the judge isn't built, and the error names the variable.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(eval_settings, "JUDGE_MODEL", None)
    with pytest.raises(LLMConfigError, match="EVAL_JUDGE_MODEL"):
        judge.build_judge()


@pytest.mark.usefixtures("judge_settings")
def test_build_judge_uses_llm_factory(llm_factory: MockType) -> None:
    """The judge comes from LLMFactory, with no temperature when none is configured.

    Args:
        llm_factory: The mocked LLM factory.
    """
    model = judge.build_judge()
    llm_factory.build.assert_called_once_with("openai", model_name="gpt-judge")
    assert model is llm_factory.build.return_value


@pytest.mark.usefixtures("judge_settings")
def test_build_judge_passes_configured_temperature(
    mocker: MockerFixture, llm_factory: MockType
) -> None:
    """A configured EVAL_JUDGE_TEMPERATURE reaches the model.

    Args:
        mocker: Pytest-mock fixture.
        llm_factory: The mocked LLM factory.
    """
    mocker.patch.object(eval_settings, "JUDGE_TEMPERATURE", 0.0)
    judge.build_judge()
    llm_factory.build.assert_called_once_with("openai", model_name="gpt-judge", temperature=0.0)


@pytest.mark.usefixtures("judge_settings", "llm_factory")
def test_build_judge_warns_when_same_model_as_agent(
    mocker: MockerFixture, judge_logger: MockType
) -> None:
    """A judge that is the agent's own model is allowed, but logged as a warning.

    Args:
        mocker: Pytest-mock fixture.
        judge_logger: The mocked logger.
    """
    mocker.patch.object(settings, "LLM_PROVIDER", "openai")
    mocker.patch.object(settings, "LLM_MODEL", "gpt-judge")
    judge.build_judge()
    judge_logger.warning.assert_called_once()


@pytest.mark.usefixtures("judge_settings", "llm_factory")
def test_build_judge_quiet_when_models_differ(judge_logger: MockType) -> None:
    """No warning when judge and agent are different models.

    Args:
        judge_logger: The mocked logger.
    """
    judge.build_judge()
    judge_logger.warning.assert_not_called()
