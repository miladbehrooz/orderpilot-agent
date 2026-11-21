# E-commerce Support Agent: Evaluation Design (v2)

_2026-11-21. Derived from `scenarios.md` and `tools.md`. Steps 1 and 2 (sections 1–5, 7–9: `answer_correctness`, `tool_correctness`, `argument_correctness`, `reason_code`, `step_count`) and step 3a (the trajectory judges `tool_selection`, `argument_quality`, `trajectory_efficiency`) are implemented. v2 changes the strategy to a hybrid one (principle 1): reference-free LLM judges become the primary quality metrics, and the hand-written references are kept for regression and judge calibration. Judge calibration (section 6, step 3b), the answer judges (step 3c) and later steps are not implemented yet._

## Principles
1. **Hybrid grading: three kinds of check, each where it fits.**
   - **Code invariants** for rules that must never break: authentication, customer isolation, no leaks, lockout, and in Milestone 2 no unconfirmed or ineligible writes. They need no reference, because they check a rule, not a path, and they are the gates. A judge may add a check next to one but is never the only check for a rule.
   - **Reference-free LLM judges** for the quality of what the agent did: did it complete the task, pick suitable tools, pass arguments the conversation supports, avoid needless steps, and use the tool results faithfully. These are the primary quality metrics.
   - **Reference-based checks** against hand-written cases (`answer_correctness`, `tool_correctness`, `argument_correctness`, `reason_code`): regression tests on known cases, and labels for calibrating the judges.

   **Why this split works for OrderPilot:** every business decision lives in tool code (design rule 1) and is unit-tested against the real database, so the evals don't need to re-check whether "45 days is outside the window". They grade the agent's behaviour around the tools. A judge that sees the whole turn, including tool calls and tool outputs, can grade that behaviour without a golden trajectory, and it accepts different paths that are equally valid. Hand-written references are slow to write and few, so they are worth most as the calibration set.
2. **Every metric answers one question, and results are never merged into one score.** Outcome ("is the answer right and backed by the tools?"), process ("was the path reasonable?") and compliance ("were the rules kept?") are reported separately, and so is every dimension inside them. A judge that grades several dimensions in one call returns one verdict per dimension under its own key, so a failure says *why* it failed.
3. **Gates are separate from scores.** A safety gate (e.g. no data leak) fails the run whatever the averages say. Quality scores are averaged.
4. **Agents are non-deterministic.** Every case runs k times. We report the per-run pass rate, **pass^k** (all k runs pass: reliability) and pass@k (at least one passes: capability).
5. **A judge is trusted only after calibration.** Each judge metric is checked against labels before it gates anything: the reference-based metrics on the hand-written cases, human labels on the disagreements, and failing trajectories built on purpose (section 6). Each judge has its own regression dataset, and its provider, model and prompt version are recorded with every experiment.
6. **Terminology: a *metric* is what is measured, an *evaluator* is the code that measures it.** A metric (`answer_correctness`) is a feedback key, a report column and the name listed in case files and on the command line (`metrics:`, `--metrics`). An evaluator (`AnswerCorrectnessEvaluator`) is the class that computes one metric for one run; "evaluator" is LangSmith's and openevals' term for it.
7. **The agent knows nothing about its evals.** `evals/` sits outside the package (like `tests/`) and only imports from `orderpilot`. Production code has no eval-only hooks.

## Tooling decisions
| Concern | Choice | Why |
|---|---|---|
| Orchestration, datasets, experiments, dashboard | **LangSmith** (`client.evaluate`) | Native LangChain/LangGraph tracing, dataset versioning, experiment comparison, annotation queues |
| Domain metrics | **Own evaluator classes** in `evals/evaluators/` | Invariants (leaks, auth, DB state) and judge rubrics are specific to OrderPilot; one base class holds the shared error policy |
| Generic pieces | **openevals**: LLM-as-judge (`create_llm_as_judge`), trajectory match (`openevals.trajectory`, step 2), reference-free trajectory judge (`create_trajectory_llm_as_judge`, step 3), multi-turn simulator later | Not worth hand-rolling; wrapped behind our own classes. Judges use our own versioned rubrics, not the generic default prompts (section 3) |
| Judge model | **OpenAI**, built through `orderpilot.llm.LLMFactory` | A different provider from the agent; provider and model are configurable |
| Not used | DeepEval | Overlaps with LangSmith (second dashboard and dataset model), heavy dependency, generic metrics |

## Pipeline
```
evals/datasets/m1/*.yaml ──sync.py──▶ LangSmith dataset "orderpilot-m1" (versioned, splits)
                                                  │
                  run.py: client.evaluate(target, data, evaluators, num_repetitions=k, metadata)
                                                  │
        ┌─────────────────────────────────────────┼──────────────────────────────────────┐
        ▼                                         ▼                                      ▼
 Target (per example)                    Evaluators (per run)                    Report (per experiment)
 seed a verified thread                  reference-free judges (step 3)          pass rate, pass^k, pass@k
 invoke(question)                        code invariants, the gates (step 4)     per scenario; flaky cases;
 → {answer, messages}                    step_count (measure)                    measures (mean, p50, p95);
                                         reference-based, hand-written cases:    evaluator errors; latency
                                         answer_correctness, tool_correctness,
                                         argument_correctness, reason_code
                                                  │
              reference-based results + human labels + built failures → judge calibration
```

## 1. Dataset layer
**The repo is the source of truth, and LangSmith is a copy.** Cases are hand-written YAML files in `evals/datasets/`, reviewed in PRs. `sync.py` upserts them into LangSmith, which keeps the version history. Cases are not edited in the LangSmith UI.

**Files.** A dataset directory (`evals/datasets/m1/`) holds `dataset.yaml` (`description`, `reference_now`) and one file per scenario family (`P.yaml`, `L.yaml`) with optional `defaults` and a list of `cases`. Each case is merged over its file's defaults, then validated as a Pydantic `EvalCase` (`evals/cases.py`; unknown fields are errors):
```yaml
- case_id: P4-01                 # stable; LangSmith example id = uuid5(case_id)
  scenario_id: P4
  splits: [full, smoke]
  inputs:
    customer_email: christinafranklin@example.com   # who the seeded thread is verified as
    question: "Can I return order 5110?"
  reference:                     # hand-written; never derived from policy.py
    expected_outcome: >-
      No, order 5110 can't be returned. It was delivered on 2026-09-28, 45 days ago, which
      is outside the 30-day return window (the deadline was 2026-10-28).
    key_facts: {order_id: 5110, delivered_on: "2026-09-28", days_since_delivery: 45}
    expected_tool_calls:         # for tool_correctness / argument_correctness
      - {name: check_return_eligibility, args: {order_id: 5110}}   # args: required subset; omit for "any"
    expected_reason_code: {tool: check_return_eligibility, code: WINDOW_EXPIRED}   # for reason_code
  metrics: [answer_correctness, tool_correctness, argument_correctness, reason_code, step_count]   # usually in the file's defaults
  notes: "Far past the window."
```
**Checks before upload** (`sync.check_dataset`, also run by `test_repo_datasets_are_valid`): every metric is registered; every case fills the reference fields its metrics need (`Evaluator.required_reference`, e.g. `expected_tool_calls`); every expected tool call names a real tool and only arguments the model can set, read from the tools' model-facing schemas (`build_tools`), so a typo such as `orderid` fails before anything is uploaded; and every expected reason code names a real tool. The code itself is typed as `ReasonCode`, so a misspelt code already fails when the case is loaded.

The customer is identified by email, which the target looks up with `OrderRepository.find_user_by_email`, so no internal id goes into the dataset and the repository needs no eval-only method.

**Mapping to LangSmith**

| Case field | LangSmith | Seen by |
|---|---|---|
| `inputs` | example `inputs` | Target (agent), evaluators |
| `reference` | example `outputs` (reference outputs) | Evaluators only |
| `case_id`, `scenario_id`, family, `metrics`, `notes` | example `metadata` | Report and UI filters; **never** the agent or the judge |
| `splits` | example splits | Run selection |

- **One dataset per milestone** (`orderpilot-m1`, later `orderpilot-m2`). Splits are neutral subsets (`smoke`, `full`). A single family is chosen by metadata filter (`client.list_examples(..., metadata={"scenario_family": "P"})`).
- **Tied to one DB build.** References hold absolute dates, so a dataset is valid for one database only. The dataset's metadata stores `reference_now`, and `run.py` refuses to run if `repo.reference_now()` differs.
- **Hand-written references.** Expected outcomes come from the scenario spec, and facts are looked up by hand in the DB. They are never computed by `policy.py`: a policy bug must not leak into its own reference.
- **Size:** 14 reference-based cases for P1–P8, L1, L3, L5 and L6, with boundary cases for days 30 and 31. This set stays small and grows mainly where calibration needs it. Coverage grows through reference-free cases (below).

**Two kinds of case.** A reference is optional per metric, not per dataset: a case fills only the reference fields its own `metrics` need (`required_reference`, checked by sync).

| | Reference-based case | Reference-free case |
|---|---|---|
| Holds | `inputs` and a hand-written `reference` (`expected_outcome`, `key_facts`, optionally `expected_tool_calls` and `expected_reason_code`) | `inputs` only |
| Metrics | the reference-based metrics, plus every reference-free metric and gate | the reference-free judges, the gates and `step_count` |
| Used for | regression on known cases; labels for judge calibration | breadth: paraphrases, typos, new scenarios (S, AU, Q), later production traces |
| Cost to add | a DB lookup and a written expected answer | a question and a customer |

The existing m1 cases are all reference-based and stay as they are. Reference-free cases aren't possible yet: `EvalCase.reference` and `Reference.expected_outcome` are required today, so the schema gets an optional reference when the first such case is added.
- **Quirks of the current DB build** (`--today 2026-11-12`): the newest order was placed on 2026-10-28 and the last delivery was on 2026-11-04. So every Processing order is already "stuck" (P1 is also a stuck order), and the most recent delivery is 8 days old (P3 uses 8 days instead of 5).

## 2. Target layer
`AgentTarget(repo)` (`evals/target.py`) builds the agent once per experiment, through `build_agent(repo=..., checkpointer=InMemorySaver())`. LangSmith gets its bound method `invoke(inputs) -> outputs` as the target. Each example runs on a new thread (`thread_id = uuid4()`). Milestone 1 is read-only, so examples share one DB. In Milestone 2 the target gets a fresh DB copy per example, and nothing else changes.

**Seeding a verified customer (single-turn).** The graph compiles with `input_schema=MessagesState`, so `user_id` in the invoke input is dropped. That is on purpose (rule 2), and the evals don't work around it in production code. Instead, `seed_verified_thread(graph, config, user)` writes state directly:

```python
graph.update_state(
    config, {"user_id": user.user_id, "messages": verified_history(user)}, as_node="agent"
)
graph.invoke({"messages": [HumanMessage(question)]}, config)
```

- `verified_history(user)` is the exchange a real verification would leave: the customer's four details, the `verify_customer` tool call, its `VERIFIED` `ToolMessage` (built from the same result model the tool returns, so the format can't drift), and a short "you're verified" reply. Without it, the model sees `user_id` set but no verification in the conversation, and may ask the customer to verify.
- `as_node="agent"` is required. Checked against langgraph 1.2.22: `update_state` on a fresh thread *without* `as_node` goes through the input schema and drops `user_id`. With `as_node="agent"`, the last message has no tool calls, so the thread ends (`next == ()`), and the next `invoke` starts from START with `user_id` set.
- AU scenarios don't use seeding. They start unverified, with scripted history.

**Outputs:** `{"answer": <final AI text>, "messages": convert_to_openai_messages(<this turn>)}`. "This turn" is the question and everything after it; the seeded verification exchange is left out, so trajectory metrics see only what the agent did for the question. The question is given an id before the run and found by it (`this_turn`), so the target doesn't depend on the length of the seeded history. openevals' trajectory matchers accept OpenAI-format message dicts.

**Errors:** `evaluate(..., error_handling="log")` keeps errored runs in the experiment. Evaluators score them as failures (see the error policy below), so a crash never drops out of the denominator.

## 3. Evaluator layer
Evaluators are objects, built by a factory that works like `LLMFactory`.

```python
class Evaluator(ABC):  # base.py: the policy every metric shares
    key: ClassVar[str]
    kind: ClassVar[Literal["gate", "score", "measure"]] = "score"
    required_reference: ClassVar[tuple[str, ...]] = ()  # Reference fields sync requires

    def __call__(self, run: Run, example: Example | None) -> EvaluationResults:
        """Error policy and applicability, shared by every metric (template method)."""

    @abstractmethod
    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]: ...


class LLMJudgeEvaluator(Evaluator):  # llm_judge.py: holds the judge; prompt looked up by version
    prompt_version: ClassVar[str]

    def __init__(self, judge: BaseChatModel, reference_now: datetime) -> None: ...
    def _make_judge(self, judge, prompt): ...  # default: openevals create_llm_as_judge
    def _verdict(self, raw) -> JudgeVerdict: ...  # reads that judge's output


class TrajectoryJudgeEvaluator(LLMJudgeEvaluator):  # trajectory_judge.py: reference-free
    def __init__(self, judge, reference_now, contract: str) -> None: ...
    def _make_judge(self, judge, prompt): ...  # openevals create_trajectory_llm_as_judge


class TrajectoryMatchEvaluator(Evaluator):  # trajectory.py: openevals superset trajectory match
    tool_args_match_mode: ClassVar[ToolArgsMatchMode]


class AnswerCorrectnessEvaluator(LLMJudgeEvaluator):  # one module per metric
    key = "answer_correctness"


class EvaluatorFactory:  # factory.py
    _REGISTRY: ClassVar[dict[str, type[Evaluator]]] = {
        "answer_correctness": AnswerCorrectnessEvaluator,
        "tool_correctness": ToolCorrectnessEvaluator,
        "argument_correctness": ArgumentCorrectnessEvaluator,
        "reason_code": ReasonCodeEvaluator,
        "step_count": StepCountEvaluator,
        "tool_selection": ToolSelectionEvaluator,
        "argument_quality": ArgumentQualityEvaluator,
        "trajectory_efficiency": TrajectoryEfficiencyEvaluator,
    }

    @classmethod
    def build(cls, names: Sequence[str], repo: OrderRepository) -> list[Evaluator]: ...
```

- **Several metrics in one run.** `client.evaluate(evaluators=[...])` takes the factory's list, and each evaluator writes its own feedback key. An evaluator can also return several results at once, so one judge call can yield several sub-verdicts.
- **Return shape: `EvaluationResults`, i.e. `{"results": [...]}`.** LangSmith rejects a bare list of `EvaluationResult` objects (it only accepts a list of dicts) and any empty result, but accepts an empty `results` list. That is how "not applicable" emits no feedback. A contract test runs our objects through LangSmith's `run_evaluator` to keep this true.
- **Signature `(run, example)`.** LangSmith passes the `Run` and the `Example` when the parameters are named exactly that (it types `example` as optional). We need `run.error` for the error policy and `example.metadata` for applicability. LangSmith names evaluators by `__name__`, which objects lack, so the base class sets `self.__name__ = key`, and **every result sets `key` explicitly**.
- **The judge is built only when needed.** The factory calls `build_judge()` only if a selected evaluator is an `LLMJudgeEvaluator`, and `agent_contract(repo)` only if one is a `TrajectoryJudgeEvaluator`. Runs with only code-based metrics then work without an OpenAI key.
- **Own base class, not LangSmith's `RunEvaluator`,** so LangSmith types stay in one adapter and a later move to Langfuse stays cheap.

**Error policy** (in `Evaluator.__call__`, written once)

| Situation | Result |
|---|---|
| Metric doesn't apply to this case (not in the case's `metrics`) | An empty `results` list: no feedback, so averages are not diluted |
| The agent run errored (exception, recursion limit) | The metric scores `False`, with the error as its comment. A `measure` emits nothing: a crashed run has nothing meaningful to measure, and it already fails every pass/fail metric |
| The evaluator or judge failed (API error, unparsable output) | No score for the metric; an `evaluator_error` result is emitted instead (metric key in its metadata), so "the judge broke" is never counted as "the agent failed" |

The step 1 and 2 metrics below are the **reference-based** ones: since v2 they serve as regression tests on the hand-written cases and as calibration labels for the step 3 judges.

### Step 1 metric: `answer_correctness`
- **Built on** openevals `create_llm_as_judge(prompt=..., judge=<ChatOpenAI from LLMFactory>, output_schema=JudgeVerdict)`. Prompt variables are passed as keyword arguments (`question`, `answer`, `expected_outcome`, `key_facts`). The judge model must support structured output.
- **Judge sees:** today's date (the database's `reference_now`, so day counts are checked against the same date the agent used), the question, the final answer and the reference (`expected_outcome`, `key_facts`). It does **not** see tool outputs (that is `faithfulness`, step 3) or any metadata.
- **Output schema:** `reasoning` (written first), then `verdict: bool`. Pass/fail only, no 1–5 scale: it is easier to calibrate and easier to act on.
- **Rubric (v2):** the decision matches the expected outcome; the key facts that are stated are correct, with day counts checked against today's date; nothing contradicts the reference; the reason is given only when the question asks why or how. Leaving out extra detail is fine; a wrong fact fails the answer.
- **The prompt** is a `JudgePrompt` (metric, version, text) in `JUDGE_PROMPTS` in `evals/prompts.py`. The evaluator stores only its `prompt_version` (`v2`), and `prompt_for` looks up the text. Old versions (`v1`) stay in the list so past experiments can be reproduced. The version goes into the experiment metadata.
- **First slice:** P1–P8 plus L1, L3, L5 and L6, which have clear references. AM1, L2 and S need behaviour rubrics, and AU needs unverified starts; they come later.

### Step 2 metrics: `tool_correctness`, `argument_correctness`
- **Built on** openevals `create_trajectory_match_evaluator(trajectory_match_mode="superset", ...)`, wrapped in `TrajectoryMatchEvaluator` (`evals/evaluators/trajectory.py`). The case's `expected_tool_calls` become one reference assistant message.
- **Superset trajectory:** every expected call needs its own matching call in the turn; extra calls and their order don't matter.
- **Arguments:** `tool_correctness` ignores them (`tool_args_match_mode="ignore"`); `argument_correctness` requires every expected key with an equal value (`"superset"`). Values are compared strictly: `"2670"` is not `2670`. An expected call without `args` matches any call of that tool.
- **Comment:** openevals returns only a bool. On a failure each expected call is matched again on its own, with the same matcher, and the comment names the missing ones, followed by the calls that were made, e.g. `Missing: get_order_details. Called: check_return_eligibility(order_id=2670).`
- **Two metrics instead of one,** so the report separates "wrong tool" from "right tool, wrong order".
- **Cases:** all of P and L1, L3, L5, L6. P8 expects `get_order_details` and then `check_return_eligibility` with `order_item_ids: [3862]`, following the agent prompt; checking all items or skipping the lookup fails `argument_correctness` / `tool_correctness` by design.

### Step 2 metrics: `reason_code`, `step_count`
- **`reason_code`** (`ReasonCodeEvaluator`, code, score): the case's `expected_reason_code` names a tool and the code it must return. The tool's **last** result in the turn is graded, so a retry after a wrong order id (`NOT_FOUND`, then the right order) is judged on the check the agent finally explains. Results are matched to their tool through `tool_call_id` and the AI message's call, not the tool message's optional `name`. A result whose content has no readable `reason_code` fails the metric (the tool contract is broken; that is not an evaluator error). The comment gives the actual code and, on a failure, the expected one: `check_return_eligibility returned ELIGIBLE, expected WINDOW_EXPIRED.`
- **Hand-written, like every reference.** The codes come from `tools.md`, never from `policy.py`, so the metric checks the decision that reached the agent on the real DB build end to end. Its sharpest cases are the day-30/31 boundaries (P3-02, P4-02). It does **not** catch P8's item selection: checking both items of order 2670 also gives `ELIGIBLE`, so that stays `argument_correctness`'s job.
- **`step_count`** (`StepCountEvaluator`, code, `kind="measure"`): the number of tool calls in the turn; the comment adds the model calls (`2 tool calls in 3 model calls.`). A measure is a number, never pass/fail: the report gives its mean and percentiles, and it never fails a gate.
- **Shared turn readers** live in `evals/evaluators/messages.py` (`tool_calls`, `model_calls`, `tool_results`), used by the trajectory metrics too.
- **Cases:** `reason_code` on P1–P8 (the L tools only return `OK`); `step_count` on every case.

### Step 3 metrics: reference-free judges
The primary quality metrics. None of them needs a reference, so they run on every case, reference-based or not. **3a, done:** `tool_selection`, `argument_quality`, `trajectory_efficiency`. **3c, planned:** `task_completion`, `faithfulness`.

| Metric | Question | Reads |
|---|---|---|
| `task_completion` | Did the agent resolve what the customer asked: a decision and its reason for the order or item meant, or a clarifying question when the request was ambiguous? | question, whole turn, answer |
| `faithfulness` | Do the answer's claims and decision follow the tool outputs (no invented facts, no override of `eligible` or `reason_code`)? | whole turn, answer |
| `tool_selection` | Were suitable tools called in a sensible order, and was none skipped that the answer relies on? | whole turn |
| `argument_quality` | Are the arguments supported by the conversation: the order and items the customer meant, no invented ids? | whole turn |
| `trajectory_efficiency` | Were redundant or unnecessary calls avoided? | whole turn |

- **The judge sees** today's date (`reference_now`), the question, the whole turn (tool calls with their arguments, tool outputs, the final answer) and **the agent's contract**: the tool descriptions and the agent-prompt rules a good trajectory follows (decide eligibility only through the eligibility tools, match an item by what it is, ask when several orders match). The contract is taken from the agent's own prompt and tool schemas, so it can't drift from them. The judge never sees a reference or metadata.
- **Own rubrics, not the default prompts.** openevals' reference-free `TRAJECTORY_ACCURACY_PROMPT` guesses the goal and asks whether the steps "make logical sense" and are "relatively efficient". It knows no OrderPilot rule, so it would pass an agent that skips `check_return_eligibility` and works out the date itself. Each metric gets a versioned prompt in `JUDGE_PROMPTS`, like `answer_correctness`.
- **More than one path can pass.** The rubric grades whether a path is reasonable within the contract, not whether it matches one golden path. An extra `get_order_details` before an eligibility check doesn't fail `tool_selection`; it fails `trajectory_efficiency` only if nothing in the answer needed it.
- **Policy is not the judge's job.** Whether a decision is right for the data is settled by tool code and its unit tests. The judges check that the agent used the tools and reported their results faithfully.
- **One judge call and one verdict per metric,** each a `reasoning` and a pass/fail `verdict` under its own key, as `answer_correctness` does. Separate calls keep the verdicts independent for calibration, and match openevals' trajectory judge, which returns one score per call.
- **Uncalibrated until section 6 is done:** reported next to the reference-based metrics, and gating nothing. `step_count` stays the exact number next to the `trajectory_efficiency` verdict.

**How 3a is built.**
- `TrajectoryJudgeEvaluator` (`evals/evaluators/trajectory_judge.py`) subclasses `LLMJudgeEvaluator` and overrides two hooks: `_make_judge` builds openevals `create_trajectory_llm_as_judge(prompt=..., judge=..., feedback_key=key)`, and `_verdict` reads its `{key, score, comment}` as a `JudgeVerdict`. Each metric is a three-line subclass in its own module (`key`, `prompt_version = "v1"`).
- **The contract** comes from `agent_contract(repo)` (`evals/contract.py`): the rendered system prompt (`render_system_prompt`) followed by each tool's name, description and model-facing parameters (`convert_to_openai_tool` over `build_tools(repo)`, so the injected `runtime` never appears). The factory builds it once per experiment and gives it to every trajectory judge.
- **The judge receives** `today`, `contract` and `outputs` (this turn's messages); openevals formats the turn into the `{outputs}` placeholder. The turn starts with the question, so there is no separate question variable. Each rubric says the customer was verified before the turn, because the seeded verification exchange is left out of it.
- **Rubrics (v1),** each narrow and naming what it does *not* judge:
  - `tool_selection`: every tool the contract requires was called before answering, none was misused, dependent calls come after what they need; asking instead of calling passes when the request was ambiguous.
  - `argument_quality`: each `order_id` and `order_item_ids` comes from the customer's message or an earlier tool result, items matched by what they are; date filters follow from today; nothing invented.
  - `trajectory_efficiency`: every call was needed by the answer or required by the contract, and no call repeats an earlier one; a retry with corrected arguments is not waste.
- **Cases:** all m1 cases (P and L) list the three metrics. A full run of 14 cases × 3 repeats makes 168 judge calls (with `answer_correctness`).

## 4. Orchestration: `run.py`
```bash
uv run python -m evals.sync --dataset m1 [--prune]       # repo YAML → LangSmith
uv run python -m evals.run --split smoke --family P --description "prompt v2"
EVAL_REPEATS=5 uv run python -m evals.run --split full
```
- **`sync.py`** checks that every metric a case names has a registered evaluator and that the YAML's `reference_now` matches the local DB, refuses a remote dataset written for another DB build, then creates new examples and updates existing ones by their stable id. `--prune` deletes remote examples whose case is gone.
- **Command-line arguments** of `run.py` are only what changes per run: `--dataset`, `--split`, `--family` (repeatable), `--metrics` (default: the metrics the selected cases list), `--description`. Judge settings, k and concurrency come from `eval_settings`.
- **Calls** `client.evaluate(AgentTarget(repo).invoke, data=examples, evaluators=EvaluatorFactory.build(...), num_repetitions=k, max_concurrency=..., experiment_prefix=..., metadata=..., error_handling="log")`. Importing `evaluate` from `langsmith.evaluation` is deprecated; `client.evaluate` is the current API.
- **Experiment metadata**, which is what makes comparisons meaningful: `agent_provider`, `agent_model`, `judge_provider`, `judge_model`, `judge_prompts` (version per judge metric), `git_sha` and `git_dirty`, `reference_now`, `repeats`, `split`, `families`.
- **Exit codes:** 0 success, 1 a gate metric failed, 2 setup or configuration error.
- **Re-grading without re-running the agent:** `client.evaluate("<experiment id>", evaluators=[...])` scores an existing experiment's stored outputs. Use it after a judge prompt change, or to add a new metric to old runs.

## 5. Reporting: `report.py`
After each run, a local summary is built from the result rows (`run`, `example`, `evaluation_results`) that `client.evaluate` returns. Repetitions share an example ID, so they are grouped by it. Runs where a metric didn't apply, or whose evaluator failed, don't count for that metric. Results are read by their evaluator's `kind`: a `measure` keeps its number, everything else is pass/fail.

| Per scenario and metric | Definition |
|---|---|
| pass rate | Passing runs / all runs |
| pass^k | Cases where all k runs pass / cases |
| pass@k | Cases where at least one run passes / cases |
| flaky | Cases with 0 < pass rate < 1, listed by `case_id` |
| evaluator errors | Count of `evaluator_error` results |
| latency | p50/p95 per run, from the runs' start and end times |

| Per scenario and measure | Definition |
|---|---|
| mean, p50, p95, max | Over the measured runs; percentiles by nearest rank |

Gate metrics fail the experiment if any run fails a gate. Measures never do.

**Planned with step 3:** judge metrics that aren't calibrated yet are marked as such, and for each reference-based case the report lists where a reference-free judge and its reference-based counterpart disagree (section 6). Evaluator errors are already counted per metric, so a judge outage shows up and isn't hidden in the pass rates.

pass^k is computed locally because LangSmith summary evaluators receive runs and examples, but **not** the per-run feedback. The overall numbers are then attached to the experiment as experiment-level feedback (`client.create_feedback(None, "<metric>:pass^k", score=..., project_id=<experiment id>)`, likewise `:pass_rate` and `:pass@k`, and `<measure>:mean` and `<measure>:p95` for each measure), so they show up in the LangSmith UI too.

## 6. Judge calibration
Every judge metric (`answer_correctness` and the step 3 judges) is calibrated before it gates anything. Until then it is reported but gates nothing, and the reference-based metrics stay the regression signal. Calibration uses no hand-built ideal trajectories: its labels come from three sources.

1. **The reference-based metrics on the hand-written cases** give labels automatically:

   | Judge metric | Reference-based counterpart |
   |---|---|
   | `task_completion` | `answer_correctness` |
   | `tool_selection` | `tool_correctness` |
   | `argument_quality` | `argument_correctness` |
   | `trajectory_efficiency` | `step_count` against the number of expected tool calls |
   | `faithfulness`, `answer_correctness` | none: human labels only |

   These are *silver* labels. A disagreement goes to human review and is not counted against the judge automatically: a different path can be equally valid, and the reference can be the one that's wrong.
2. **Failing turns built on purpose.** The hand-written cases mostly pass, so on their own they can't show how often a judge catches a failure (TNR). Failing turns are made by changing passing ones: drop the eligibility call but keep the decision, swap the `order_id`, check every item in P8, make the answer contradict the tool's `reason_code`, repeat a lookup. Each change has a known label.
3. **Human labels** in a LangSmith **annotation queue**, under separate keys (`<metric>_human`): the disagreements from 1, a sample of the agreements, and runs of reference-free cases.

- **Agreement:** `calibrate.py` reports TPR, TNR and Cohen's κ per judge metric, with confidence intervals, because the labelled set is small.
- **Consistency:** the same stored runs are graded again several times (re-grading, section 4), and the report shows how often verdicts flip. Judge provider, model, temperature and prompt version are fixed per experiment and recorded in its metadata.
- **Improvement loop:** labelled disagreements become `few_shot_examples` in the next prompt version.
- **Each judge has its own dataset** (`orderpilot-judge-<metric>`). Inputs are what the judge sees; the expected output is the label; the target is the judge itself. Every prompt change is checked against it before use.

## 7. Configuration
Eval-only settings live in `evals/settings.py`: `EvalSettings(BaseSettings)` with `env_prefix="EVAL_"`, the same `.env` file (`PROJECT_ROOT / ".env"`), and the `eval_settings` singleton.

| Field | Env var | Default |
|---|---|---|
| `JUDGE_PROVIDER: ProviderName` | `EVAL_JUDGE_PROVIDER` | `"openai"` |
| `JUDGE_MODEL: str \| None` | `EVAL_JUDGE_MODEL` | `None` (unset fails with `LLMConfigError`) |
| `JUDGE_TEMPERATURE: float \| None` | `EVAL_JUDGE_TEMPERATURE` | `None` (not sent) |
| `REPEATS: int` | `EVAL_REPEATS` | `3` |
| `MAX_CONCURRENCY: int` | `EVAL_MAX_CONCURRENCY` | `4` |

API keys, policy values (`RETURN_WINDOW_DAYS`…) and `DB_PATH` stay in `orderpilot.settings`, so graders and tools always read the same policy. `build_judge()` logs a warning when the judge uses the same provider and model as the agent.


## 8. Testing the eval code
Tests live in `tests/`, use pytest-mock, and never call a real LLM:
- **Case loader:** valid files load; schema errors fail at load time.
- **`seed_verified_thread`:** against the real graph with a mocked model, `user_id` is set, the history is consistent, and the next `invoke` reaches the eligibility tools.
- **`Evaluator.__call__`:** not-applicable emits nothing; an errored run scores `False`; a judge failure emits `evaluator_error`; keys are always set.
- **`AnswerCorrectnessEvaluator`:** the judge is mocked; checks the mapping from judge output to result, and that expected tool calls never reach the judge.
- **Trajectory metrics:** the real openevals matcher on hand-built OpenAI-format turns: missing, extra and reordered calls, wrong and extra arguments, strict values, the comment.
- **`reason_code` and `step_count`:** expected and other codes, the last result after a retry, other tools' results ignored, no result, unreadable content; tool and model calls counted; no measure on an errored run.
- **Trajectory judges** (step 3a): openevals' trajectory judge is mocked; checks that the judge receives exactly `today`, `contract` and the turn and nothing from the reference or metadata, the score and comment mapping, the shared error policy, that every v1 prompt fills, the LangSmith contract, and that the factory builds the contract once, only for trajectory judges.
- **`agent_contract`** (no mocks): it holds the rendered system prompt and every tool with its parameters, and no injected `runtime`.
- **Dataset checks:** unknown metrics, missing required reference fields, unknown tools and arguments (including injected ones such as `runtime`), unknown reason codes and reason codes from unknown tools are refused.
- **Target:** only this turn's messages are returned.
- **Report maths:** pass rate, pass^k, pass@k and flaky detection on hand-built rows; measures routed by kind, summarised per scenario, kept out of pass rates and gates.
- **LangSmith contract:** our evaluator objects run through LangSmith's `run_evaluator`, including the empty not-applicable result.
- **Sync and run:** LangSmith `Client` mocked with autospec: create vs update by stable id, pruning, DB-build refusals, `evaluate()` arguments and metadata.
- **The committed dataset:** `evals/datasets/m1` loads and every scenario has a smoke case.

## 9. Layout
```
evals/
├── settings.py              # EvalSettings / eval_settings
├── errors.py                # EvalError, EvalSetupError
├── client.py                # make_client(): LangSmith client from settings
├── judge.py                 # build_judge() via LLMFactory
├── cases.py                 # EvalCase model, YAML loader, DB-build check
├── datasets/m1/             # dataset.yaml, P.yaml, L.yaml: hand-written cases
├── sync.py                  # repo → LangSmith upsert
├── target.py                # AgentTarget, seed_verified_thread, verified_history
├── prompts.py               # versioned judge prompts
├── contract.py              # agent_contract(): system prompt and tool schemas for the judges
├── evaluators/              # base.py (Evaluator), llm_judge.py (LLMJudgeEvaluator), trajectory.py
│                            # (TrajectoryMatchEvaluator), trajectory_judge.py (TrajectoryJudgeEvaluator),
│                            # messages.py (turn readers), factory.py, one module per metric
├── run.py                   # client.evaluate() entrypoint
├── report.py                # pass rate, pass^k, flaky, measures, latency; attaches experiment feedback
└── calibrate.py             # (step 3, not yet) judge agreement with labels
```
**Dependencies:** the `evals` dependency group (`langsmith`, `openevals`, `pyyaml`, `types-PyYAML`), installed by default. These are declared explicitly even where they arrive indirectly. Step 2 needs no new dependency: trajectory match is in `openevals.trajectory`.

## 10. Roadmap
The full list of planned metrics, grouped by dimension, is in [`metrics.md`](metrics.md).

| Step | Adds |
|---|---|
| **1** | Pipeline end to end, `answer_correctness`, report (P plus L1, L3, L5, L6). Done |
| **2** | Reference-based trajectory and decisions (code): `tool_correctness`, `argument_correctness` (openevals trajectory match), `reason_code`, and the reference-free `step_count` (measure). Done |
| 3 | Reference-free judges and calibration, run next to the reference-based metrics on m1 (section 3). **3a, done:** `tool_selection`, `argument_quality`, `trajectory_efficiency` and the agent's contract. **3b:** calibration of every judge, `answer_correctness` included (section 6: built failures, annotation queue, `calibrate.py`, judge datasets). **3c:** `task_completion`, `faithfulness` |
| 4 | Code invariants and behaviour: leak scan and the other gates (`kind="gate"`), `grounded_facts`; clarification and refusal rubrics (AM, L2, S); AU with unverified starts. New cases are reference-free unless a regression case is needed |
| 5 | Multi-turn: openevals simulated user (AU3, AM1 follow-ups), graded by the reference-free judges over the whole conversation |
| M2 | DB-state grading on a fresh DB copy per example (W1–W8) |
| Later | CI: smoke split on PRs, full suite on demand; reference-free judges as LangSmith online evaluators on real traces; single agent vs supervisor on the same datasets |

## Checked API facts (2026-11-16)
- **langsmith 0.14.17:**
  - `client.evaluate()` takes `num_repetitions`, `max_concurrency`, `metadata` and (through `**kwargs`) `error_handling`; `target` can be an existing experiment ID (re-grading). Importing from `langsmith.evaluation` is deprecated.
  - Evaluator parameter names it supports: `run`, `example`, `inputs`, `outputs`, `reference_outputs`, `attachments`.
  - Several results: `{"results": [...]}` or a list of *dicts*; a list of `EvaluationResult` objects and an empty result are rejected.
  - Repetitions repeat the same example ID.
  - Summary evaluators get `(runs, examples)` without feedback.
  - `list_examples` filters by `splits` and `metadata`.
  - `create_feedback` accepts `project_id` for experiment-level feedback.
- **langgraph 1.2.22:**
  - Input keys outside `input_schema` are dropped.
  - `update_state` without `as_node` on a fresh thread drops them too; with `as_node="agent"` the values are kept.
  - `Command(update=...)` also bypasses `input_schema` (see open questions).
- **openevals 0.2.5 (2026-11-12):**
  - `create_llm_as_judge` takes `judge=` (a LangChain chat model with structured output), `feedback_key`, `choices`, `continuous`, `use_reasoning`, `few_shot_examples`, `output_schema` and `system`.
  - With `output_schema`, the return value has the schema's shape instead of `{key, score, comment}`.
  - Extra keyword arguments fill the prompt template (`prompt.format(**kwargs)`).
  - LLM errors are not caught, hence our own error policy.
  - It also ships the trajectory matchers and trajectory judge (`openevals.trajectory`).
  - `create_trajectory_llm_as_judge(prompt=..., judge=..., feedback_key=..., few_shot_examples=..., system=...)` (checked 2026-11-21) formats the turn as text and returns one `{key, score, comment}` per call. It has no `output_schema`, so a call grades one metric. Its default prompt needs a reference (`TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE`); the reference-free one is `TRAJECTORY_ACCURACY_PROMPT`. Extra keyword arguments fill the prompt template, like `create_llm_as_judge`. The turn is formatted as `<assistant>` blocks with `<tool_call>` name and arguments and `<tool>` blocks with `<tool_result>` id and content: the calls carry no id, so parallel calls can't be paired with their results by the judge (rare in OrderPilot, which calls one tool at a time).
  - `create_trajectory_match_evaluator(...)` is called with `outputs=` and `reference_outputs=` (OpenAI-format message lists; tool calls as `{"function": {"name", "arguments"}}` or `{"name", "args"}`) and returns `{"key": "trajectory_<mode>_match", "score": bool, "comment": None}`. Each call is `@traceable`, so it shows as a child trace of the evaluator.
  - `superset` mode: each reference call is matched to its own output call with the same name; args mode `superset` means the output args contain every reference key with an equal value.
- **agentevals 0.0.9 (2025-07-24):** no release in 16 months, and its trajectory matchers are now in openevals, so it is not used.

## Open questions
- k and concurrency defaults, once the cost per run is known.
- What judge–human agreement (κ) counts as "acceptable" before a judge metric gates anything.
- Whether a calibrated `faithfulness` judge should become a gate next to the code check `grounded_facts`, or stay a score.
- **Outside the evals:** `Command(update=...)` and `threads.update_state` on the LangGraph server can set `user_id` directly, and `langgraph.json` has no auth. Rule 2 holds for the `input` path only. This needs a decision before any deployment (e.g. LangGraph custom auth).
