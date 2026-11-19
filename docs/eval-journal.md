# Eval Journal

OrderPilot is an AI customer-support agent for an online clothing shop: it checks orders,
and tells customers whether an order can be cancelled or returned. It is built around an
evaluation suite that runs the agent on realistic customer questions, many times each,
and grades every answer.

This journal collects real findings from that suite: what the evals caught, why it
happened, what we changed, and what we learned. Each story is written for readers who
don't know the project; the evidence line at the end points to the experiments behind it.

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
