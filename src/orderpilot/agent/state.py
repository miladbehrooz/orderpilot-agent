"""Graph state of the support agent and the auth predicates derived from it."""

import operator
from typing import Annotated, NotRequired

from langgraph.graph import MessagesState

from orderpilot.settings import settings


class AgentState(MessagesState):
    """Graph state: the conversation plus authentication state.

    Attributes:
        user_id: The verified customer. Set only by ``verify_customer`` and never a tool
            argument the model can choose (rule 2).
        failed_auth_attempts: Failed verification attempts. Summed by a reducer, so
            parallel ``verify_customer`` calls in one step are all counted.
    """

    user_id: NotRequired[int | None]
    failed_auth_attempts: NotRequired[Annotated[int, operator.add]]


def is_verified(state: AgentState) -> bool:
    """Return whether the conversation has a verified customer.

    Args:
        state: Current graph state.

    Returns:
        ``True`` once ``verify_customer`` has succeeded.
    """
    return state.get("user_id") is not None


def is_locked(state: AgentState) -> bool:
    """Return whether verification is locked after too many failed attempts (AU3).

    Args:
        state: Current graph state.

    Returns:
        ``True`` once failed attempts reach ``settings.MAX_AUTH_ATTEMPTS``.
    """
    return state.get("failed_auth_attempts", 0) >= settings.MAX_AUTH_ATTEMPTS


__all__ = ["AgentState", "is_locked", "is_verified"]
