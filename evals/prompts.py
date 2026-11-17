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
