"""Versioned prompts of the LLM judges.

A prompt is never edited in place once it has been used in an experiment: a change gets a
new constant (``..._V2``) and a new ``prompt_version`` on its evaluator, so experiments
graded by different prompts are never compared as if they were the same metric.

The prompts are ``str.format`` templates: literal braces must be doubled.
"""

from typing import Final

ANSWER_CORRECTNESS_V1: Final = """\
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
"""
"""Answer correctness against a hand-written reference. Tool outputs are not shown."""

ANSWER_CORRECTNESS_V2: Final = """\
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
"""
"""Answer correctness, v2: adds today's date and only requires the reason when it is asked for."""

__all__ = ["ANSWER_CORRECTNESS_V1", "ANSWER_CORRECTNESS_V2"]
