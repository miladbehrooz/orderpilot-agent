"""The agent's contract given to the trajectory judges, built from the real prompt and tools."""

from evals.evaluators.judges.contract import agent_contract
from orderpilot.tools import build_tools
from tests.stubs import StubRepository


def test_contract_holds_the_system_prompt_and_every_tool(stub_repo: StubRepository) -> None:
    """The contract has the rendered system prompt and each tool with its parameters.

    Args:
        stub_repo: Repository on the test shop; its TODAY is 2026-11-12.
    """
    contract = agent_contract(stub_repo)
    assert "Today is 2026-11-12 (UTC)." in contract
    assert "Never override a tool's decision." in contract
    for tool in build_tools(stub_repo):
        assert f"## {tool.name}\n" in contract
    assert '"order_item_ids"' in contract


def test_contract_leaves_out_injected_arguments(stub_repo: StubRepository) -> None:
    """Arguments the model can't set, such as the injected ``runtime``, never appear.

    Args:
        stub_repo: Repository on the test shop.
    """
    assert '"runtime"' not in agent_contract(stub_repo)
