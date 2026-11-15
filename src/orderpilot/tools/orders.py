"""Order lookup tools: ``list_my_orders`` and ``get_order_details``.

Thin wrappers around :mod:`orderpilot.services.orders`. The customer is always the
verified ``user_id`` from graph state (rule 2); there is no argument to pick another one.
Typed arguments let schema validation reject bad input before the tool runs, and the
graph turns that into ``INVALID_ARGUMENTS`` (rule 9).
"""

from datetime import date
from typing import Annotated

from langchain.tools import BaseTool, ToolRuntime, tool
from pydantic import Field

from orderpilot.agent.state import AgentState
from orderpilot.domain import OrderStatus
from orderpilot.repository import OrderRepository
from orderpilot.services import orders as orders_service
from orderpilot.tools.auth import not_verified_result, require_verified


def make_list_my_orders(repo: OrderRepository) -> BaseTool:
    """Build the ``list_my_orders`` tool bound to a repository.

    Args:
        repo: Data access used to read the orders.

    Returns:
        The LangChain tool.
    """

    @tool("list_my_orders", parse_docstring=True)
    def list_my_orders(
        runtime: ToolRuntime[None, AgentState],
        status: OrderStatus | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> str:
        """List the verified customer's orders, newest first.

        Each order comes with its dates, item count, total, days in its current status
        and the flags is_stuck / is_possibly_lost. has_more is true if more orders match.

        Args:
            runtime: Injected by LangGraph (state, tool_call_id); hidden from the model.
            status: Only orders with this status.
            date_from: Only orders placed on or after this date (YYYY-MM-DD).
            date_to: Only orders placed on or before this date (YYYY-MM-DD).
            limit: Maximum number of orders to return (1-50).
        """
        user_id = require_verified(runtime)
        if user_id is None:
            return not_verified_result()
        result = orders_service.list_my_orders(
            repo, user_id, status=status, date_from=date_from, date_to=date_to, limit=limit
        )
        return result.to_content()

    return list_my_orders


def make_get_order_details(repo: OrderRepository) -> BaseTool:
    """Build the ``get_order_details`` tool bound to a repository.

    Args:
        repo: Data access used to read the order.

    Returns:
        The LangChain tool.
    """

    @tool("get_order_details", parse_docstring=True)
    def get_order_details(order_id: int, runtime: ToolRuntime[None, AgentState]) -> str:
        """Get one of the verified customer's orders with all its items.

        Items include order_item_id, product name, brand, category, sale price, status
        and dates. A missing product name or brand is null.

        Args:
            order_id: The order to look up.
            runtime: Injected by LangGraph (state, tool_call_id); hidden from the model.
        """
        user_id = require_verified(runtime)
        if user_id is None:
            return not_verified_result()
        return orders_service.get_order_details(repo, user_id, order_id).to_content()

    return get_order_details


__all__ = ["make_get_order_details", "make_list_my_orders"]
