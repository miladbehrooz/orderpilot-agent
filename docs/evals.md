# E-commerce Support Agent: Evaluation Design (v1)

_Draft, 2026-11-16. Derived from `scenarios.md` and `tools.md`. Nothing here is implemented yet._

## Principles
1. **Grade with code first, then state, then an LLM judge.** Tools return a `reason_code` and the supporting numbers, so most behaviour can be checked by code. An LLM judge is used only where meaning matters: whether the answer is correct, whether it asked a clarifying question, whether it refused.
2. **Every metric answers one question.** "Is the answer right?" (correctness against a reference) and "Is every claim backed by a tool result?" (groundedness against the trajectory) are separate metrics, so a failure says *why* it failed.
3. **Gates are separate from scores.** A safety gate (e.g. no data leak) fails the run whatever the averages say. Quality scores are averaged.
4. **Agents are non-deterministic.** Every case runs k times. We report the per-run pass rate, **pass^k** (all k runs pass: reliability) and pass@k (at least one passes: capability).
5. **A judge is trusted only after calibration.** Judge verdicts are checked against human labels before they gate anything, and the judge has its own regression dataset.
6. **The agent knows nothing about its evals.** `evals/` sits outside the package (like `tests/`) and only imports from `orderpilot`. Production code has no eval-only hooks.

## Tooling decisions
| Concern | Choice | Why |
|---|---|---|
| Orchestration, datasets, experiments, dashboard | **LangSmith** (`langsmith.evaluate`) | Native LangChain/LangGraph tracing, dataset versioning, experiment comparison, annotation queues |
| Domain metrics | **Own evaluator classes** in `evals/evaluators/` | The valuable checks (reason codes, leaks, DB state, facts against a reference) are specific to OrderPilot |
| Generic pieces | **openevals** (LLM-as-judge setup, later the multi-turn simulator); **agentevals** (trajectory match), from step 2 | Not worth hand-rolling; wrapped behind our own classes |
| Judge model | **OpenAI**, built through `orderpilot.llm.LLMFactory` | A different provider from the agent; provider and model are configurable |
| Not used | DeepEval | Overlaps with LangSmith (second dashboard and dataset model), heavy dependency, generic metrics |

## Pipeline
```
evals/datasets/m1/*.yaml ──sync.py──▶ LangSmith dataset "orderpilot-m1" (versioned, splits)
                                                  │
                         run.py: evaluate(target, data, evaluators, num_repetitions=k, metadata)
                                                  │
        ┌─────────────────────────────────────────┼──────────────────────────────────────┐
        ▼                                         ▼                                      ▼
 Target (per example)                    Evaluators (per run)                    Report (per experiment)
 seed a verified thread                  answer_correctness (judge)              pass rate, pass^k, pass@k
 invoke(question)                        later: trajectory, reason codes,        per scenario; flaky cases;
 → {answer, messages}                    leak gate, groundedness, DB state       evaluator errors; latency, cost
                                                  │
                                annotation queue → human labels → judge calibration
```

## 1. Dataset layer
**The repo is the source of truth, and LangSmith is a copy.** Cases are hand-written YAML files in `evals/datasets/`, reviewed in PRs. `sync.py` upserts them into LangSmith, which keeps the version history. Cases are not edited in the LangSmith UI.

**Case schema**, a Pydantic `EvalCase` model validated on load:
```yaml
- case_id: P4-01                 # stable; LangSmith example id = uuid5(case_id)
  scenario_id: P4
  splits: [full, smoke]
  inputs:
    question: "Can I return order 1234?"
    user_id: 87                  # used only to seed the verified thread
  reference:                     # hand-written; never derived from policy.py
    expected_outcome: >-
      Not returnable: delivered on 2026-09-28, 45 days ago, outside the 30-day window.
    key_facts: {order_id: 1234, delivered_on: "2026-09-28", days_since_delivery: 45}
  evaluators: [answer_correctness]   # optional; defaults per scenario family
  notes: "Clear case, far past the boundary"
```

**Mapping to LangSmith**

| Case field | LangSmith | Seen by |
|---|---|---|
| `inputs` | example `inputs` | Target (agent), evaluators |
| `reference` | example `outputs` (reference outputs) | Evaluators only |
| `case_id`, `scenario_id`, family, `evaluators`, `notes` | example `metadata` | Report and UI filters; **never** the agent or the judge |
| `splits` | example splits | Run selection |

- **One dataset per milestone** (`orderpilot-m1`, later `orderpilot-m2`). Splits are neutral subsets (`smoke`, `full`). A single family is chosen by metadata filter (`client.list_examples(..., metadata={"scenario_family": "P"})`).
- **Tied to one DB build.** References hold absolute dates, so a dataset is valid for one database only. The dataset's metadata stores `reference_now`, and `run.py` refuses to run if `repo.reference_now()` differs.
- **Hand-written references.** Expected outcomes come from the scenario spec, and facts are looked up by hand in the DB. They are never computed by `policy.py`: a policy bug must not leak into its own reference.
- **Size:** 1–3 cases per scenario, to start with. Later, a helper can suggest DB rows that match a scenario's data condition, with the references still written by hand.

## 2. Target layer
`make_target(repo)` builds the agent once per experiment, through `build_agent(repo=..., checkpointer=InMemorySaver())`, and returns `target(inputs) -> outputs`. Each example runs on a new thread (`thread_id = uuid4()`). Milestone 1 is read-only, so examples share one DB. In Milestone 2 the factory gets a fresh DB copy per example, and nothing else changes.

**Seeding a verified customer (single-turn).** The graph compiles with `input_schema=MessagesState`, so `user_id` in the invoke input is dropped. That is on purpose (rule 2), and the evals don't work around it in production code. Instead, `seed_verified_thread(graph, config, user)` writes state directly:

```python
graph.update_state(config, {"user_id": user.id, "messages": verified_history(user)}, as_node="agent")
graph.invoke({"messages": [HumanMessage(question)]}, config)
```

- `verified_history(user)` is the exchange a real verification would leave: the customer's four details, the `verify_customer` tool call, its `VERIFIED` `ToolMessage` (built from the same result model the tool returns, so the format can't drift), and a short "you're verified" reply. Without it, the model sees `user_id` set but no verification in the conversation, and may ask the customer to verify.
- `as_node="agent"` is required. Checked against langgraph 1.2.22: `update_state` on a fresh thread *without* `as_node` goes through the input schema and drops `user_id`. With `as_node="agent"`, the last message has no tool calls, so the thread ends (`next == ()`), and the next `invoke` starts from START with `user_id` set.
- AU scenarios don't use seeding. They start unverified, with scripted history.

**Outputs:** `{"answer": <final AI text>, "messages": convert_to_openai_messages(<full trajectory>)}`. The trajectory is returned from step 1 on, so trajectory evaluators can be added later without changing the target. agentevals accepts OpenAI-format message dicts.

**Errors:** `evaluate(..., error_handling="log")` keeps errored runs in the experiment. Evaluators score them as failures (see the error policy below), so a crash never drops out of the denominator.

## 3. Evaluator layer
Evaluators are objects, built by a factory that works like `LLMFactory`.

```python
class Evaluator(ABC):
    key: ClassVar[str]
    kind: ClassVar[Literal["gate", "score"]] = "score"

    def __call__(self, run: Run, example: Example) -> list[EvaluationResult]:
        """Error policy and applicability, shared by every metric (template method)."""

    @abstractmethod
    def evaluate(self, run: Run, example: Example) -> list[EvaluationResult]: ...

class LLMJudgeEvaluator(Evaluator):        # holds the judge and a versioned prompt
    prompt: ClassVar[str]
    prompt_version: ClassVar[str]
    def __init__(self, judge: BaseChatModel) -> None: ...

class AnswerCorrectnessEvaluator(LLMJudgeEvaluator):
    key = "answer_correctness"

class EvaluatorFactory:
    _REGISTRY: ClassVar[dict[str, type[Evaluator]]] = {"answer_correctness": AnswerCorrectnessEvaluator}
    @classmethod
    def build(cls, names: Sequence[str]) -> list[Evaluator]: ...
```

- **Several metrics in one run.** `evaluate(evaluators=[...])` takes the factory's list, and each evaluator writes its own feedback key. An evaluator can also return several results at once (LangSmith accepts a list of results), so one judge call can yield several sub-verdicts.
- **Signature `(run, example)`.** LangSmith passes the `Run` and the `Example` when the parameters are named exactly that. We need `run.error` for the error policy and `example.metadata` for applicability. A callable object has no `__name__`, so **every result sets `key` explicitly**.
- **The judge is built only when needed.** The factory calls `build_judge()` only if a selected evaluator is an `LLMJudgeEvaluator`. Runs with only code-based metrics then work without an OpenAI key.
- **Own base class, not LangSmith's `RunEvaluator`,** so LangSmith types stay in one adapter and a later move to Langfuse stays cheap.

**Error policy** (in `Evaluator.__call__`, written once)

| Situation | Result |
|---|---|
| Metric doesn't apply to this case (not in the case's `evaluators`) | No result emitted, so averages are not diluted |
| The agent run errored (exception, recursion limit) | The metric scores `False`, with the error as its comment |
| The evaluator or judge failed (API error, unparsable output) | No score for the metric; an `evaluator_error` result is emitted instead (metric key in its metadata), so "the judge broke" is never counted as "the agent failed" |

### Step 1 metric: `answer_correctness`
- **Built on** openevals `create_llm_as_judge(prompt=..., judge=<ChatOpenAI from LLMFactory>, output_schema=...)`. The judge model must support structured output.
- **Judge sees:** the question, the final answer and the reference (`expected_outcome`, `key_facts`). It does **not** see tool outputs (that is groundedness, a later metric) or any metadata.
- **Output schema:** `reasoning` (written first), then `verdict: bool`. Pass/fail only, no 1–5 scale: it is easier to calibrate and easier to act on.
- **Rubric:** the decision matches the expected outcome; the key facts that are stated are correct; nothing contradicts the reference; the reason is explained. Leaving out extra detail is fine; a wrong fact fails the answer.
- **The prompt** is a versioned constant in `evals/prompts.py` (`ANSWER_CORRECTNESS_V1`), and its version goes into the experiment metadata.
- **First slice:** P1–P8 plus L1, L3, L5 and L6, which have clear references. AM1, L2 and S need behaviour rubrics, and AU needs unverified starts; they come later.

## 4. Orchestration: `run.py`
```bash
uv run python -m evals.run --split smoke --evaluators answer_correctness --description "prompt v2"
EVAL_REPEATS=5 uv run python -m evals.run --split full
```
- **Command-line arguments** are only what changes per run: `--split`, `--family`, `--evaluators`, `--description`. Judge settings, k and concurrency come from `eval_settings`.
- **Calls** `evaluate(target, data=examples, evaluators=EvaluatorFactory.build(...), num_repetitions=k, max_concurrency=..., experiment_prefix=..., metadata=...)`.
- **Experiment metadata**, which is what makes comparisons meaningful: `agent_provider`, `agent_model`, `judge_provider`, `judge_model`, the prompt version of each judge, `git_sha`, `reference_now`.
- **Re-grading without re-running the agent:** `evaluate("<experiment id>", evaluators=[...])` scores an existing experiment's stored outputs. Use it after a judge prompt change, or to add a new metric to old runs.

## 5. Reporting: `report.py`
After each run, a local summary is built from the `ExperimentResults` rows (`run`, `example`, `evaluation_results`). Repetitions share an example ID, so they are grouped by it.

| Per scenario and metric | Definition |
|---|---|
| pass rate | Passing runs / all runs |
| pass^k | Cases where all k runs pass / cases |
| pass@k | Cases where at least one run passes / cases |
| flaky | Cases with 0 < pass rate < 1, listed by `case_id` |
| evaluator errors | Count of `evaluator_error` results |
| latency, tokens, cost | p50/p95 per run, from the runs |

Gate metrics fail the experiment if any run fails a gate.

pass^k is computed locally because LangSmith summary evaluators receive runs and examples, but **not** the per-run feedback. The summary numbers are then attached to the experiment as experiment-level feedback (`client.create_feedback(project_id=<experiment id>, key=..., score=...)`), so they show up in the LangSmith UI too.

## 6. Judge calibration
1. Send the first experiments to a LangSmith **annotation queue** and label 30–50 runs by hand, under the separate key `answer_correctness_human`.
2. `calibrate.py` measures agreement between judge and human: TPR, TNR and Cohen's κ. The judge gates nothing until agreement is acceptable.
3. Labelled disagreements become `few_shot_examples` in the next prompt version.
4. **The labelled set becomes the judge's own dataset** (`orderpilot-judge-answer-correctness`). Inputs are question, answer and reference; the expected output is the human label; the target is the judge itself. Every judge prompt change is checked against it before use.

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

_Pending:_ CLAUDE.md still says `evals/` reads only `settings`. It needs a line allowing `eval_settings` as a singleton.

## 8. Testing the eval code
Tests live in `tests/`, use pytest-mock, and never call a real LLM:
- **Case loader:** valid files load; schema errors fail at load time.
- **`seed_verified_thread`:** against the real graph with a mocked model, `user_id` is set, the history is consistent, and the next `invoke` reaches the eligibility tools.
- **`Evaluator.__call__`:** not-applicable emits nothing; an errored run scores `False`; a judge failure emits `evaluator_error`; keys are always set.
- **`AnswerCorrectnessEvaluator`:** the judge is mocked; checks the mapping from judge output to result.
- **Report maths:** pass rate, pass^k, pass@k and flaky detection on hand-built rows.

## 9. Layout
```
evals/
├── settings.py              # EvalSettings / eval_settings
├── judge.py                 # build_judge() via LLMFactory
├── cases.py                 # EvalCase model + YAML loader
├── datasets/m1/{P,L}.yaml   # hand-written cases
├── sync.py                  # repo → LangSmith upsert
├── target.py                # make_target(repo), seed_verified_thread, verified_history
├── prompts.py               # versioned judge prompts
├── evaluators/              # base.py (Evaluator, LLMJudgeEvaluator), factory.py, answer_correctness.py
├── run.py                   # evaluate() entrypoint
├── report.py                # pass rate, pass^k, flaky, cost; attaches experiment feedback
└── calibrate.py             # judge vs human agreement
```
**Dependencies:** a new `evals` dependency group with `langsmith`, `openevals` and `pyyaml`. These are declared explicitly even where they arrive indirectly today. `agentevals` is added in step 2.

## 10. Roadmap
| Step | Adds |
|---|---|
| **1** | Pipeline end to end, `answer_correctness`, report, calibration (P plus L1, L3, L5, L6) |
| 2 | Trajectory: expected tool calls and `reason_code` (code), agentevals trajectory match |
| 3 | Gates and behaviour: leak scan (`kind="gate"`), clarification and refusal rubrics (AM, L2, S), AU with unverified starts |
| 4 | Groundedness: claims in the answer against tool outputs in the trajectory |
| 5 | Multi-turn: openevals simulated user (AU3, AM1 follow-ups) |
| M2 | DB-state grading on a fresh DB copy per example (W1–W8) |
| Later | CI: smoke split on PRs, full suite on demand; single agent vs supervisor on the same datasets |

## Checked API facts (2026-11-16)
- **langsmith 0.14.17:**
  - `evaluate()` takes `num_repetitions`, `max_concurrency`, `metadata` and `error_handling`; `target` can be an existing experiment ID (re-grading).
  - Evaluator parameter names it supports: `run`, `example`, `inputs`, `outputs`, `reference_outputs`, `attachments`.
  - A list of results is accepted.
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
  - Its behaviour on LLM errors is undocumented, hence our own error policy.
- **agentevals 0.0.9 (2025-07-24):**
  - The trajectory match modes are `strict`, `unordered`, `subset` and `superset`, plus `tool_args_match_mode`.
  - It accepts OpenAI-format dicts or LangChain messages.
  - **Risk:** no release in 16 months. It is wrapped behind our `Evaluator`, so it can be replaced by our own trajectory matcher if needed.

## Open questions
- k and concurrency defaults, once the cost per run is known.
- What judge–human agreement (κ) counts as "acceptable" before `answer_correctness` gates anything.
- **Outside the evals:** `Command(update=...)` and `threads.update_state` on the LangGraph server can set `user_id` directly, and `langgraph.json` has no auth. Rule 2 holds for the `input` path only. This needs a decision before any deployment (e.g. LangGraph custom auth).
