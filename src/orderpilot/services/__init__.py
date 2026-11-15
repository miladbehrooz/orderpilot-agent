"""Business services: framework-free functions that tools (and later an MCP server) call.

The result models are re-exported here, e.g. ``from orderpilot.services import ToolResult``.
The service modules themselves are imported as namespaces, e.g.
``from orderpilot.services import auth as auth_service``.
"""

from orderpilot.services.results import (
    ItemView,
    OrderDetails,
    OrderDetailsResult,
    OrderListResult,
    OrderSummary,
    ReasonCode,
    ToolResult,
    VerificationOutcome,
    VerifyResult,
)

__all__ = [
    "ItemView",
    "OrderDetails",
    "OrderDetailsResult",
    "OrderListResult",
    "OrderSummary",
    "ReasonCode",
    "ToolResult",
    "VerificationOutcome",
    "VerifyResult",
]
