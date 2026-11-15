"""The rendered system prompt: TODAY and policy numbers come from data and settings."""

from datetime import datetime

from pytest_mock import MockerFixture

from orderpilot.agent.prompts import render_system_prompt
from orderpilot.settings import settings


def test_prompt_contains_today_and_policy_days(today: datetime) -> None:
    """The prompt states TODAY and the return, stuck and lost thresholds.

    Args:
        today: Reference time.
    """
    prompt = render_system_prompt(today)
    assert "Today is 2026-11-12" in prompt
    assert f"within {settings.RETURN_WINDOW_DAYS} days" in prompt
    assert f"Processing for more than {settings.STUCK_DAYS} days" in prompt
    assert f"shipped more than {settings.LOST_DAYS} days ago" in prompt


def test_prompt_follows_settings(mocker: MockerFixture, today: datetime) -> None:
    """Changing a threshold in settings changes the prompt, so the two never disagree.

    Args:
        mocker: Pytest-mock fixture.
        today: Reference time.
    """
    mocker.patch.object(settings, "STUCK_DAYS", 3)
    assert "Processing for more than 3 days" in render_system_prompt(today)


def test_prompt_requires_eligibility_tools_and_forbids_claiming_actions(today: datetime) -> None:
    """Eligibility questions go to the tools, and no action is claimed as done.

    Milestone 1 can only check, so the prompt forbids saying a cancellation or return
    was carried out.

    Args:
        today: Reference time.
    """
    prompt = render_system_prompt(today)
    assert "check_cancellation_eligibility" in prompt
    assert "check_return_eligibility" in prompt
    assert "Never say it has been done" in prompt
