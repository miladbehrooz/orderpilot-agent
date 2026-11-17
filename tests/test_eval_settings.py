"""Eval settings: defaults, the EVAL_ prefix, and the LangSmith client built from settings."""

import os

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture

from evals import client as client_module
from evals.errors import EvalSetupError
from evals.settings import EvalSettings
from orderpilot.settings import settings


@pytest.fixture
def clean_eval_env(mocker: MockerFixture) -> None:
    """Hide ``.env`` and every ``EVAL_*`` environment variable from ``EvalSettings`` for one test.

    Args:
        mocker: Pytest-mock fixture.
    """
    kept = {k: v for k, v in os.environ.items() if not k.startswith("EVAL_")}
    mocker.patch.dict(os.environ, kept, clear=True)
    mocker.patch.dict(EvalSettings.model_config, {"env_file": None})


@pytest.mark.usefixtures("clean_eval_env")
def test_eval_settings_defaults() -> None:
    """Without environment or .env values, the judge is OpenAI with no model, k=3, concurrency 4."""
    config = EvalSettings()
    assert config.JUDGE_PROVIDER == "openai"
    assert config.JUDGE_MODEL is None
    assert config.JUDGE_TEMPERATURE is None
    assert config.REPEATS == 3
    assert config.MAX_CONCURRENCY == 4


@pytest.mark.usefixtures("clean_eval_env")
def test_eval_settings_read_prefixed_env(mocker: MockerFixture) -> None:
    """EVAL_-prefixed variables set the eval fields; unprefixed names are ignored.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.dict(
        os.environ,
        {
            "EVAL_JUDGE_PROVIDER": "anthropic",
            "EVAL_JUDGE_MODEL": "judge-model",
            "EVAL_REPEATS": "5",
            "REPEATS": "99",
        },
    )
    config = EvalSettings()
    assert config.JUDGE_PROVIDER == "anthropic"
    assert config.JUDGE_MODEL == "judge-model"
    assert config.REPEATS == 5


@pytest.mark.usefixtures("clean_eval_env")
def test_eval_settings_reject_zero_repeats(mocker: MockerFixture) -> None:
    """REPEATS (k) must be at least 1, so pass^k is always defined.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.dict(os.environ, {"EVAL_REPEATS": "0"})
    with pytest.raises(ValueError, match="REPEATS"):
        EvalSettings()


def test_make_client_requires_api_key(mocker: MockerFixture) -> None:
    """Without LANGSMITH_API_KEY, building the client fails with a clear setup error.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(settings, "LANGSMITH_API_KEY", None)
    with pytest.raises(EvalSetupError, match="LANGSMITH_API_KEY"):
        client_module.make_client()


def test_make_client_uses_key_and_endpoint(mocker: MockerFixture) -> None:
    """The client gets the same key and endpoint as tracing.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(settings, "LANGSMITH_API_KEY", SecretStr("ls-test"))
    mocker.patch.object(settings, "LANGSMITH_ENDPOINT", "https://eu.example")
    client_cls = mocker.patch.object(client_module, "Client", autospec=True)
    client_module.make_client()
    client_cls.assert_called_once_with(api_key="ls-test", api_url="https://eu.example")
