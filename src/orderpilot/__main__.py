"""Interactive chat with the agent for manual testing: ``uv run python -m orderpilot``."""

import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from orderpilot.app import build_agent
from orderpilot.errors import OrderPilotError


def main() -> int:
    """Run a terminal chat session until EOF or ``exit``.

    Returns:
        Process exit code: 0 on a normal exit, 1 on a configuration or data error.
    """
    try:
        graph = build_agent(checkpointer=InMemorySaver())
    except OrderPilotError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("OrderPilot chat. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in {"exit", "quit"}:
            return 0
        if not text:
            continue
        try:
            result = graph.invoke({"messages": [HumanMessage(text)]}, config)
        except OrderPilotError as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        reply = result["messages"][-1]
        if isinstance(reply, AIMessage):
            print(f"agent> {reply.text}")


if __name__ == "__main__":
    sys.exit(main())
