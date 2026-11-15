"""Auth predicates derived from graph state."""

from typing import cast

import pytest
from langchain_core.messages import HumanMessage

from orderpilot.agent.state import AgentState, is_locked, is_verified
from orderpilot.settings import settings


def _state(**fields: object) -> AgentState:
    """Return a graph state with one message and the given extra fields.

    Args:
        **fields: Extra state fields, e.g. ``user_id=1``.

    Returns:
        The state.
    """
    return cast(AgentState, {"messages": [HumanMessage("hi")], **fields})


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        pytest.param({}, False, id="no-user-id"),
        pytest.param({"user_id": None}, False, id="user-id-none"),
        pytest.param({"user_id": 7}, True, id="user-id-set"),
    ],
)
def test_is_verified(fields: dict[str, object], expected: bool) -> None:
    """A conversation is verified exactly when user_id is set.

    Args:
        fields: Extra state fields.
        expected: Expected result.
    """
    assert is_verified(_state(**fields)) is expected


@pytest.mark.parametrize(
    ("failed", "expected"),
    [
        pytest.param(0, False, id="no-failures"),
        pytest.param(settings.MAX_AUTH_ATTEMPTS - 1, False, id="one-below-limit"),
        pytest.param(settings.MAX_AUTH_ATTEMPTS, True, id="at-limit"),
        pytest.param(settings.MAX_AUTH_ATTEMPTS + 1, True, id="above-limit"),
    ],
)
def test_is_locked_at_max_attempts(failed: int, expected: bool) -> None:
    """Verification locks once failures reach MAX_AUTH_ATTEMPTS.

    Args:
        failed: Failed attempts in state.
        expected: Expected result.
    """
    assert is_locked(_state(failed_auth_attempts=failed)) is expected


def test_is_locked_without_counter_is_false() -> None:
    """A fresh conversation, with no counter in state yet, is not locked."""
    assert is_locked(_state()) is False
