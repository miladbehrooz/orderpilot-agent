# Eval Journal

OrderPilot is an AI customer-support agent for an online clothing shop: it checks orders,
and tells customers whether an order can be cancelled or returned. It is built around an
evaluation suite that runs the agent on realistic customer questions, many times each,
and grades every answer.

This journal collects real findings from that suite: what the evals caught, why it
happened, what we changed, and what we learned. Each story is written for readers who
don't know the project; the evidence line at the end points to the experiments behind it.

---

## The grader approved every extra step the agent could explain afterwards

_2026-11-23_

The suite has an automated grader for wasted effort: did the agent make lookups it didn't
need? It is itself an AI model reading the agent's work, so before trusting it we tested
it. We took conversations where we already knew the answer and checked whether the grader
agreed.

On made-up mistakes it was perfect. We copied good conversations and added an obviously
pointless step, the same lookup twice in a row, and it caught every one. On real mistakes
it failed. An earlier experiment with a bigger model had made an extra order lookup on
four questions, every time it was asked them. The grader let 16 of those 20 conversations
pass.

Its reasoning was always the same: "the agent used the result in its answer, so the
lookup was needed". That was exactly the rule we had written. But an agent can always
mention something it looked up. Here it repeated item names and prices the first lookup
had already returned, or added a shipping date the customer never asked about. The rule
let any extra step pass once the agent had found a use for it.

We rewrote the rule: a lookup counts only if it brought information the question needed
and nothing earlier had already given. On the same conversations the grader then caught
15 of the 20 extra lookups. It still passed the conversations it had always passed, and
still caught every planted repeat.

The last 5 were all the same question, "Can I still cancel this order?". Here the grader
was right and our expected answer was wrong. The agent's instructions say to warn the
customer when an order is stuck, but the tool that checks whether an order can be
cancelled doesn't say whether it is stuck. A second lookup was the only way to follow the
instructions. The fix belongs in that tool, not in the grader.

**Lessons:**
- Test an AI grader on real mistakes, not only on made-up ones. Obvious planted
  mistakes are easy to catch, so they tell you little about how good it is.
- When a grader is wrong in the same way every time, check the rule you gave it first. It
  may be following that rule exactly.
- When the grader and the expected answer disagree, either one can be wrong. Here the
  disagreement pointed at a gap in one of the agent's tools.

_Evidence: `trajectory_efficiency` calibrated on `claude-sonnet-5-5-full-a2b65ba9`
(`evals.calibrate`, 1 verdict per item; extra calls labelled from the call count): rubric
v1 passed 16 of 20 runs with more calls than expected (TNR 0.20, κ 0.23), all on P1-01,
P4-01, P5-01 and P6-01; rubric v2 failed 15 of 20 (TNR 0.75, κ 0.78), and the remaining
5 are P1-01, where `check_cancellation_eligibility` returns no `is_stuck`. Planted repeats
caught 6 of 6 with both versions. On `claude-haiku-5-5-full-baa6c0ac` v2 kept all 42 real
runs passing and caught 14 of 14 planted repeats. Small samples; one judge verdict per
item._

---

## The customer said "the jeans". The agent asked "which jeans?"

_2026-11-18_

A customer asked to return "just the jeans" from an order with two items: a pair of
jeans, and a pair of shorts from a brand called James Jeans. About one time in four, the
agent didn't answer. It asked the customer which item they meant.

We would probably not have caught this by testing by hand. The agent got it right most
of the time, so a few manual tries would likely all have passed. The eval suite asks
every question many times, so the failure showed up as a pattern rather than a fluke.
It also records every run: what the agent answered, which tools it called, and why the
grader failed it. That made the cause visible right away: in every failed run, the
agent looked up the order and then stopped to ask.

The cause wasn't the model. Our instructions said "if several items could match, ask
which one", and the brand name made the shorts look like a match. When we tried a bigger
model, it got this wrong *more* often in our runs, not less: it followed the unclear rule
more faithfully.

We changed one sentence: match an item by what it is, not by words in its name. After
that, both models got this question right every time.

A fix in one place can change behaviour somewhere else, so we reran the whole suite, not
only this question. With the model we use, everything still passed. With the bigger
model, the new sentence had a side effect: it started looking up order details for
questions that didn't need them, and on one question it then mixed up two numbers in
most of its answers. Without the full run, that side effect would have gone unnoticed.
We haven't fixed it yet.

**Lessons:**
- Some failures happen only sometimes. You catch them by running each question many
  times and recording every run, not by trying a few questions by hand.
- When an agent misbehaves, check the instructions before blaming the model. A stronger
  model doesn't fix an unclear rule; it may follow it more strictly.
- After every fix, rerun everything. An eval suite shows what a fix does to the rest of
  the agent's behaviour, not only to the case you fixed.

_Evidence: case P8-01. Before the change: Claude Haiku 5.5 failed 4 of 15 runs across
three experiments (3 of 10 in `claude-haiku-5-5-full-06cc1f19`); Claude Sonnet 5.5 failed
3 of 5 (`claude-sonnet-5-5-full-480bb2ec`). After: Haiku 10 of 10
(`claude-haiku-5-5-full-9cfe14b2`), Sonnet 5 of 5 (`claude-sonnet-5-5-full-a2b65ba9`).
Full suite after the change on Haiku: 14 cases × 3 runs, all passed
(`claude-haiku-5-5-full-b3a24fe1`). Side effect on Sonnet: case P4-01 added an
unneeded order lookup in 5 of 5 runs and gave a wrong day count in 4 of 5
(`claude-sonnet-5-5-full-a2b65ba9`). Small samples; the direction was the same in every
run._

---

## The agent asked for five orders. The new grader called it a mistake

_2026-11-22_

Until now, every check on the agent's tool calls compared them with calls we had written
down by hand for each question. That works, but every new question needs someone to write
its expected calls first. So we added a different kind of grader: an AI judge that reads
the whole conversation, including the agent's instructions and what each tool returned,
and decides whether the agent's tool calls made sense. It needs no hand-written answer.

Before trusting the new judge, we ran it next to the old hand-written check on the same
runs. Where both agree, the judge is probably sound. Where they disagree, one of them is
wrong, and that is where to look.

They disagreed in one place. A customer asked "Where's my order?", and the agent looked up
their orders, asking for at most five. The old check passed it. The judge failed it in two
of three runs, because the customer never mentioned the number five.

The agent was right. Asking for up to five orders is a sensible page size, the tool allows
any number from 1 to 50, and the customer only had one order anyway. The judge was
following its instructions too literally. We had told it that no value in a tool call may
be invented, meaning order numbers and dates, but "value" also covered settings like the
page size.

We changed the judge's instructions so that the rule covers ids and dates, and settings
pass with any allowed value unless they hide something the answer needed. Then we graded
the same stored runs again, without running the agent again. Only those two verdicts
changed, both to a pass; every other verdict stayed the same. The old version of the
instructions is kept, so the first results can still be reproduced.

**Lessons:**
- A new grader needs testing like the agent does. Running it next to a check you already
  trust shows where it goes wrong before you rely on it.
- A disagreement between two graders isn't always the agent's fault. Here, the first one
  found a flaw in the grader.
- As with the agent, the fix was in the instructions, not the model: one rule was written
  more broadly than we meant it.

_Evidence: case L1-01 in experiment `claude-haiku-5-5-full-3897d8a5` (14 cases × 3 runs).
`argument_quality` v1 failed 2 of 3 runs, both calling `list_my_orders(limit=5)`;
`argument_correctness` passed all 3. `argument_quality` v2, graded locally on the same 42
stored runs, changed only those 2 verdicts, both to pass. Small sample._
