# OrderPilot

**OrderPilot: an evaluated e-commerce support agent.** A LangGraph customer-support agent for an online clothing shop (order tracking, return/cancellation eligibility, and later returns, cancellations and support tickets), built around a strong evaluation suite. Portfolio project: code quality, clarity and the evaluation story matter as much as features.

## Read these first
- `docs/project-context.md` — goal, name, stack
- `docs/scenarios.md` — business rules and all scenarios (source of truth for behaviour and tests)
- `docs/tools.md` — tool specs, reason codes, design principles
- `docs/data.md` — database schema, cleaning, time handling, known quirks
- `docs/evals.md` — evaluation design: pipeline, datasets, evaluators, metrics, roadmap

If code and docs disagree, ask before changing either.

## Stack
- Python 3.11+, LangChain + LangGraph, SQLite (stdlib `sqlite3`)
- LangChain/LangGraph APIs change often: check the current docs before using an API, don't rely on memory.
- Keep dependencies minimal.
- Evals: LangSmith runs them (datasets, experiments); metrics are our own evaluator classes, using openevals for the generic parts (LLM-as-judge; trajectory match in `openevals.trajectory`). Judge model on OpenAI via `LLMFactory`. See `docs/evals.md`.

## Non-negotiable design rules
1. **Business rules and date maths live in tool code**, never only in prompts. Tools return a decision, a `reason_code` and the supporting numbers; the LLM only explains them.
2. **`user_id` lives in graph state** (tools read it via LangChain `ToolRuntime`, i.e. `runtime.state`). It is never a tool argument the LLM can set.
3. **Auth gate:** before `verify_customer` succeeds, only that tool is bound to the model, and every other tool also refuses with `NOT_VERIFIED`.
4. **Another user's order returns `NOT_FOUND`**, the same as a non-existent order.
5. **`TODAY` comes from `meta.reference_now`** in the database. Never use `datetime.now()` in tools or evals.
6. **Policy values in one place, `Settings` (`src/orderpilot/settings.py`):** `RETURN_WINDOW_DAYS=30`, `STUCK_DAYS=7`, `LOST_DAYS=14`, `MAX_AUTH_ATTEMPTS=3`.
7. **Write tools re-check eligibility themselves** before writing.
8. **Every scenario ID** (e.g. `P4`, `S1`) maps to eval cases in `evals/`, and the IDs live there only. Unit tests in `tests/` are named after the rule or behaviour they check (e.g. `test_return_window_boundary`), never after a scenario ID.
9. **Tool-call errors are returned as structured results**, not raised to the LLM as stack traces.

## Milestones
- **Milestone 1 (read-only):** `verify_customer`, `list_my_orders`, `get_order_details`, `check_cancellation_eligibility`, `check_return_eligibility`; single-agent LangGraph loop; eval harness.
- **Milestone 2 (writes):** `cancel_order`, `return_items`, `create_support_ticket`; state-based grading on a fresh DB copy per test.
- **Later:** supervisor/worker multi-agent version, compared with the single agent on the same evals.

## Open decisions (ask, don't decide silently)
- How write confirmation (scenario W8) is enforced: LangGraph `interrupt()`, a confirm step in the tool contract, or prompt-only.
- Which LLM provider/model to use for the agent.

## Suggested layout
```
orderpilot-agent/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── docs/                 # design docs (above)
├── scripts/build_db.py   # CSV -> ecommerce.sqlite
├── data/                 # CSVs, git-ignored
├── src/orderpilot/
│   ├── settings.py       # Settings: policy values, DB path, LLM settings, log level
│   ├── app.py            # composition root: repo -> tools, model, prompt -> graph
│   ├── server.py         # `graph` for `langgraph dev` (langgraph.json)
│   ├── llm.py            # LLMFactory (provider registry: openai, anthropic)
│   ├── domain.py         # plain dataclasses (User, later Order, OrderItem)
│   ├── errors.py         # OrderPilotError and its subclasses
│   ├── observability/    # log.py (Loguru `logger`), tracing.py (LangSmith setup)
│   ├── repository/       # OrderRepository Protocol, SqliteOrderRepository, db.py (SQLite helpers)
│   ├── services/         # framework-free business functions + result models
│   ├── tools/            # thin LangChain @tool wrappers around services
│   ├── agent/            # graph, state, prompts (knows only model + tools)
│   └── policy.py         # eligibility logic shared by read and write tools
├── evals/                # eval harness outside the package: cases, target, evaluators, runner (docs/evals.md)
└── tests/                # pytest + pytest-mock unit tests (no real LLM calls)
```

## Commands
```bash
uv sync                                   # install deps (dev + data groups by default)
cp sample.env .env                        # then fill in keys
uv run pre-commit install                 # git hook: ruff, ruff format, mypy, file checks
uv run python scripts/build_db.py --data-dir data --out ecommerce.sqlite --today 2026-11-12
uv run ruff check . && uv run ruff format .   # lint + format
uv run mypy                               # type check (strict)
uv run pytest                             # tests
uv run langgraph dev                      # local agent server + Studio (see README)
uv run python -m orderpilot               # terminal chat with the agent
uv run python -m evals.sync --dataset m1  # eval cases (evals/datasets/m1) -> LangSmith dataset
uv run python -m evals.run --split smoke  # run an eval experiment (EVAL_REPEATS=k runs per case)
```

## Coding style
- Google-style docstrings (`Args:`, `Returns:`, `Yields:`, `Raises:`) on **every** function and class, including private `_helpers` and tests. Ruff (`D` rules, `convention = "google"`) enforces this for public ones only, so write the private ones by hand.
- Type hints everywhere; `mypy --strict` and `ruff check` / `ruff format` must pass.
- **No unnecessary module-level constants.** Before adding an `UPPER_CASE = ...` global, use the first option that fits:
  1. **A local variable** if only one function uses the value (e.g. `week = pd.Timedelta(weeks=1)` inside `shift_dates`).
  2. **Derive it** from an existing source of truth instead of repeating it (e.g. read column lists from the SQLite schema with `PRAGMA table_info`, not a hand-written list).
  3. **A field on `Settings`** for runtime configuration or business policy (e.g. `RETURN_WINDOW_DAYS`, `LOG_LEVEL`), read from the `settings` singleton (see below).
  4. **A frozen dataclass config** created by a `default_config()` function and passed in explicitly, for fixed structural settings of a script in `scripts/` (see `BuildConfig` in `scripts/build_db.py`).
  5. **A pytest fixture** for test data (e.g. `today`), not a module-level `TODAY = ...`.
- **Inside `src/orderpilot/` (and `evals/`), runtime configuration and policy values are read directly from the `settings` singleton** (`from orderpilot.settings import settings`) where they are needed, e.g. `settings.MAX_AUTH_ATTEMPTS` in `make_verify_customer` or `is_locked`. Don't thread `Settings` or individual policy values through function arguments. In tests, override a value with `mocker.patch.object(settings, "RETURN_WINDOW_DAYS", 10)`.
- **Eval-only settings live in `evals/settings.py`** (`EvalSettings`, `env_prefix="EVAL_"`, singleton `eval_settings`): judge provider/model, repeats, concurrency. `evals/` reads those from `eval_settings` and everything else (API keys, policy values, `DB_PATH`) from `settings`, so graders and tools share one policy. Don't add eval-only fields to `Settings`.
- **Scripts in `scripts/` are offline tooling outside the core and don't read `settings`**: their inputs come from CLI arguments, and their fixed structural settings from a `default_config()` dataclass passed in explicitly (option 4, e.g. `BuildConfig`). This keeps a build reproducible from its command line alone, independent of `.env`. The `logger` singleton may still be used.
- Exceptions: `settings.py` (e.g. `PROJECT_ROOT`), `__all__`, `__version__`, type aliases, and the `logger`/`settings`/`eval_settings` singletons.

## Testing
- pytest with **pytest-mock**: use the `mocker` fixture (typed as `pytest_mock.MockerFixture`). Don't import `unittest.mock` directly or use `monkeypatch` for patching.
- **Mock the edges, not the logic under test.** Mock things that are slow, random, external or hard to trigger: LLM calls, the clock (`datetime.now`), SQLite/IO failures, `KeyboardInterrupt`, and a collaborator when testing a caller on its own (e.g. `main()` with `build()` mocked). Don't mock the database in tool and policy tests: they run against a real SQLite file.
- **Patch where it is looked up:** `mocker.patch.object(build_db, "write", ...)` or `mocker.patch("build_db.sqlite3.connect", ...)`, not the defining module.
- Use `autospec=True` when patching functions and classes, so a call with the wrong signature fails the test.
- Use `side_effect` to raise errors or simulate behaviour, `return_value` for canned results, and `mocker.spy` to check calls to real code without replacing it.
- Assert on calls (`assert_called_once_with`, `call_args`) only when the call is the behaviour being tested; otherwise assert on outcomes (return values, files, DB rows).
- Build test data in fixtures (`tmp_path` CSVs, a fresh DB copy); never depend on `data/`.
- Scripts in `scripts/` are importable in tests (`pythonpath = ["scripts"]`), e.g. `import build_db`.

## Working style
- Build in small steps; tools and policy first, with unit tests against the real database, then the graph, then evals.
- Unit tests must not call a real LLM; mock it with `mocker` (see Testing).
- Don't commit `data/`, `*.sqlite` or API keys.
