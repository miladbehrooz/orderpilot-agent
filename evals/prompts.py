"""Versioned prompts of the LLM judges.

A prompt is never edited in place once it has been used in an experiment: a change is added
as a new entry with a new version, and the evaluator's ``prompt_version`` is bumped, so
experiments graded by different prompts are never compared as if they were the same metric.

The prompts are ``str.format`` templates: literal braces must be doubled.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from evals.errors import EvalSetupError


class JudgePrompt(BaseModel):
    """One version of one judge prompt.

    Attributes:
        metric: Metric the prompt grades, e.g. ``answer_correctness``.
        version: Prompt version, ``v`` followed by a number, e.g. ``v2``.
        text: The ``str.format`` template sent to the judge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str = Field(pattern=r"^[a-z_]+$")
    version: str = Field(pattern=r"^v\d+$")
    text: str = Field(min_length=1)


JUDGE_PROMPTS: Final = (
    JudgePrompt(
        metric="answer_correctness",
        version="v1",
        text="""\
You grade a customer-support agent of an online clothing shop.
Decide whether the agent's ANSWER is correct, given the REFERENCE written by a human expert.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<reference_outcome>
{expected_outcome}
</reference_outcome>

<reference_facts>
{key_facts}
</reference_facts>

The answer is CORRECT only if all of these hold:
1. Its decision matches the reference outcome (e.g. can or cannot cancel or return, the
   order's status, which order or item it is about).
2. Every fact it states that the reference also covers (dates, amounts, order ids, day
   counts, item names) matches the reference.
3. It does not contradict the reference, and states no facts that the reference makes
   impossible.
4. When the reference gives a reason for the decision, the answer gives that reason too.

Leaving out details is fine. Wording, tone, formatting and extra politeness do not matter.
Offering further help or mentioning that customer support can process a request is fine.
A wrong decision or a wrong fact makes the answer INCORRECT.

Reason step by step first, then give the verdict.
""",
    ),
    JudgePrompt(
        metric="answer_correctness",
        version="v2",
        text="""\
You grade a customer-support agent of an online clothing shop.
Decide whether the agent's ANSWER is correct, given the REFERENCE written by a human expert.

Today is {today}. The agent answered on this date. Any day count or "days ago" in the
answer must be counted from this date; do not use any other date.

<question>
{question}
</question>

<answer>
{answer}
</answer>

<reference_outcome>
{expected_outcome}
</reference_outcome>

<reference_facts>
{key_facts}
</reference_facts>

The answer is CORRECT only if all of these hold:
1. Its decision matches the reference outcome (e.g. can or cannot cancel or return, the
   order's status, which order or item it is about).
2. Every fact it states that the reference also covers (dates, amounts, order ids, day
   counts, item names) matches the reference. Day counts are checked against today's date
   given above.
3. It does not contradict the reference, and states no facts that the reference makes
   impossible.
4. If the question asks why or how the decision was reached, the answer gives the reason
   from the reference. If the question doesn't ask for the reason, leaving it out is fine.

Leaving out details is fine. Wording, tone, formatting and extra politeness do not matter.
Offering further help or mentioning that customer support can process a request is fine.
A wrong decision or a wrong fact makes the answer INCORRECT.

Reason step by step first, then give the verdict.
""",
    ),
    JudgePrompt(
        metric="tool_selection",
        version="v1",
        text="""\
You grade one turn of a customer-support agent of an online clothing shop.
Decide whether the agent called SUITABLE TOOLS for the customer's request.

Today is {today}. The customer was verified before this turn started, so verification is
not part of the turn and must not be expected in it.

The agent works under this contract: its instructions and the tools it may call.
<agent_contract>
{contract}
</agent_contract>

The turn: the customer's message, each tool call the agent made with its arguments, each
tool result, and the agent's final answer.
<turn>
{outputs}
</turn>

The tool selection PASSES only if all of these hold:
1. Every tool the contract requires for this request was called before the agent answered.
   For example: an eligibility tool before any "can" or "cannot" answer about cancelling or
   returning; get_order_details before checking a single item of an order.
2. No tool was used for something its description doesn't cover.
3. A call that needs another call's result comes after that call.
4. If the request was ambiguous and the contract says to ask, asking the customer instead of
   calling a tool passes.

Different sequences of calls can be equally valid: pass any that follows the contract.
Do NOT judge the arguments, extra or repeated calls, or whether a tool's decision is right.
Other checks cover those, and the tools are always right about the shop's data and policy.

Reason step by step first, then give the score: true if the turn passes, false otherwise.
""",
    ),
    JudgePrompt(
        metric="argument_quality",
        version="v1",
        text="""\
You grade one turn of a customer-support agent of an online clothing shop.
Decide whether every tool ARGUMENT the agent passed is supported by the conversation.

Today is {today}. The customer was verified before this turn started, so verification is
not part of the turn and must not be expected in it.

The agent works under this contract: its instructions and the tools it may call.
<agent_contract>
{contract}
</agent_contract>

The turn: the customer's message, each tool call the agent made with its arguments, each
tool result, and the agent's final answer.
<turn>
{outputs}
</turn>

The arguments PASS only if all of these hold:
1. Every order_id is the order the customer named, or one that an earlier tool result in
   this turn returned for what the customer described.
2. Every order_item_ids value lists exactly the items the customer meant, taken from an
   earlier tool result. Match an item by what it is (its category or product type), not by
   words in a brand or product name. Leaving order_item_ids out when the customer asks
   about the whole order is right.
3. Date filters (date_from, date_to) follow from what the customer said and today's date.
4. No id, date or value is invented: each one comes from the customer's message or an
   earlier tool result.

Judge only the arguments of the calls that were made. Do NOT judge which tools were called,
whether a call was needed, or whether a tool's decision is right. Other checks cover those,
and the tools are always right about the shop's data and policy.

Reason step by step first, then give the score: true if the turn passes, false otherwise.
""",
    ),
    JudgePrompt(
        metric="trajectory_efficiency",
        version="v1",
        text="""\
You grade one turn of a customer-support agent of an online clothing shop.
Decide whether the agent's turn was EFFICIENT: free of needless tool calls.

Today is {today}. The customer was verified before this turn started, so verification is
not part of the turn and must not be expected in it.

The agent works under this contract: its instructions and the tools it may call.
<agent_contract>
{contract}
</agent_contract>

The turn: the customer's message, each tool call the agent made with its arguments, each
tool result, and the agent's final answer.
<turn>
{outputs}
</turn>

The turn PASSES only if all of these hold:
1. Every call was needed: the final answer uses its result, or the contract requires it.
   For example, get_order_details is needed before checking a single item of an order,
   but not before checking eligibility for a whole order the customer named.
2. No call repeats an earlier call with the same arguments.

These are not waste: a retry with corrected arguments after an error or NOT_FOUND result,
and a lookup the contract requires. A turn with no tool calls passes this check.
Do NOT judge which tools were called, the arguments, or whether a tool's decision is right.
Other checks cover those, and the tools are always right about the shop's data and policy.

Reason step by step first, then give the score: true if the turn passes, false otherwise.
""",
    ),
)


def prompt_for(metric: str, version: str) -> str:
    """Return the judge prompt of one metric and version.

    Args:
        metric: Metric name, e.g. ``"answer_correctness"``.
        version: Prompt version, e.g. ``"v2"``.

    Returns:
        The ``str.format`` template of the prompt.

    Raises:
        EvalSetupError: If no prompt exists for that metric and version.
    """
    for prompt in JUDGE_PROMPTS:
        if prompt.metric == metric and prompt.version == version:
            return prompt.text
    known = ", ".join(f"{p.metric} {p.version}" for p in JUDGE_PROMPTS)
    raise EvalSetupError(f"No judge prompt for {metric} {version}. Known: {known}")


__all__ = ["JUDGE_PROMPTS", "JudgePrompt", "prompt_for"]
