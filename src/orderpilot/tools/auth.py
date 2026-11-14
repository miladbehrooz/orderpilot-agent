"""The ``verify_customer`` tool and the auth guard used by every other tool."""

from typing import Any, Final

from langchain.tools import BaseTool, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from orderpilot.agent.state import AgentState, is_verified
from orderpilot.repository import OrderRepository
from orderpilot.services import auth as auth_service
from orderpilot.services.results import ReasonCode, ToolResult
from orderpilot.settings import settings

VERIFY_TOOL_NAME: Final = "verify_customer"


def require_verified(runtime: ToolRuntime[None, AgentState]) -> int | None:
    """Return the verified ``user_id`` from graph state (rule 2, rule 3).

    Every non-auth tool calls this first and returns :func:`not_verified_result` when it
    gets ``None``, even though such tools are not bound before verification.

    Args:
        runtime: Injected tool runtime.

    Returns:
        The verified user's id, or ``None`` if the conversation is not verified.
    """
    return runtime.state.get("user_id")


def not_verified_result() -> str:
    """Return the tool content for a call made before verification.

    Returns:
        JSON with reason code ``NOT_VERIFIED``.
    """
    return ToolResult(reason_code=ReasonCode.NOT_VERIFIED).to_content()


def make_verify_customer(repo: OrderRepository) -> BaseTool:
    """Build the ``verify_customer`` tool bound to a repository.

    Args:
        repo: Data access used to look up the account.

    Returns:
        The LangChain tool.
    """

    @tool(VERIFY_TOOL_NAME, parse_docstring=True)
    def verify_customer(
        first_name: str,
        last_name: str,
        email: str,
        postal_code: str,
        runtime: ToolRuntime[None, AgentState],
    ) -> Command[Any]:
        """Verify the customer's identity. Must succeed before any other tool can be used.

        Args:
            first_name: Customer's first name.
            last_name: Customer's last name.
            email: Email address on the customer's account.
            postal_code: Postal code of the customer's address.
            runtime: Injected by LangGraph (state, tool_call_id); hidden from the model.
        """
        outcome = auth_service.verify_customer(
            repo,
            first_name=first_name,
            last_name=last_name,
            email=email,
            postal_code=postal_code,
            failed_attempts=runtime.state.get("failed_auth_attempts", 0),
            already_verified=is_verified(runtime.state),
            max_attempts=settings.MAX_AUTH_ATTEMPTS,
        )
        message = ToolMessage(outcome.result.to_content(), tool_call_id=runtime.tool_call_id)
        update: dict[str, Any] = {"messages": [message]}
        if outcome.user_id is not None:
            update["user_id"] = outcome.user_id
        if outcome.counts_as_failure:
            update["failed_auth_attempts"] = 1
        return Command(update=update)

    return verify_customer


__all__ = ["VERIFY_TOOL_NAME", "make_verify_customer", "not_verified_result", "require_verified"]
