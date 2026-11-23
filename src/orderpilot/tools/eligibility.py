"""Eligibility tools: ``check_cancellation_eligibility`` and ``check_return_eligibility``.

Thin wrappers around :mod:`orderpilot.services.eligibility`. They only check and never
change an order. The customer is always the verified ``user_id`` from graph state
(rule 2), and invalid arguments are rejected by the tool schema (rule 9).
"""

from typing import Annotated

from langchain.tools import BaseTool, ToolRuntime, tool
from pydantic import Field

from orderpilot.agent.state import AgentState
from orderpilot.repository import OrderRepository
from orderpilot.services import eligibility as eligibility_service
from orderpilot.tools.auth import not_verified_result, require_verified


def make_check_cancellation_eligibility(repo: OrderRepository) -> BaseTool:
    """Build the ``check_cancellation_eligibility`` tool bound to a repository.

    Args:
        repo: Data access used to read the order.

    Returns:
        The LangChain tool.
    """

    @tool("check_cancellation_eligibility", parse_docstring=True)
    def check_cancellation_eligibility(
        order_id: int, runtime: ToolRuntime[None, AgentState]
    ) -> str:
        """Check whether one of the verified customer's orders can be cancelled.

        Only checks; it does not cancel anything. Returns eligible and a reason_code
        (ELIGIBLE, ALREADY_SHIPPED, ALREADY_RETURNED, ORDER_CANCELLED or NOT_FOUND),
        with the order's status and dates, days_in_status, and the flags is_stuck /
        is_possibly_lost.

        Args:
            order_id: The order to check.
            runtime: Injected by LangGraph (state, tool_call_id); hidden from the model.
        """
        user_id = require_verified(runtime)
        if user_id is None:
            return not_verified_result()
        result = eligibility_service.check_cancellation_eligibility(repo, user_id, order_id)
        return result.to_content()

    return check_cancellation_eligibility


def make_check_return_eligibility(repo: OrderRepository) -> BaseTool:
    """Build the ``check_return_eligibility`` tool bound to a repository.

    Args:
        repo: Data access used to read the order.

    Returns:
        The LangChain tool.
    """

    @tool("check_return_eligibility", parse_docstring=True)
    def check_return_eligibility(
        order_id: int,
        runtime: ToolRuntime[None, AgentState],
        order_item_ids: Annotated[list[int], Field(min_length=1)] | None = None,
    ) -> str:
        """Check whether items of one of the verified customer's orders can be returned.

        Only checks; it does not record a return. Each item gets eligible, a reason_code
        (ELIGIBLE, NOT_DELIVERED, WINDOW_EXPIRED, ALREADY_RETURNED or ORDER_CANCELLED),
        the refund amount if eligible, and its delivery date, days since delivery and
        return deadline. The top-level reason_code is ELIGIBLE if any item is eligible.
        NOT_FOUND means the order, or one of the given items, is not in this order.

        Args:
            order_id: The order to check.
            runtime: Injected by LangGraph (state, tool_call_id); hidden from the model.
            order_item_ids: Only these items (from get_order_details); all items if omitted.
        """
        user_id = require_verified(runtime)
        if user_id is None:
            return not_verified_result()
        result = eligibility_service.check_return_eligibility(
            repo, user_id, order_id, order_item_ids
        )
        return result.to_content()

    return check_return_eligibility


__all__ = ["make_check_cancellation_eligibility", "make_check_return_eligibility"]
