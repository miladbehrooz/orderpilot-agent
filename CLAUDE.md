# OrderPilot

**OrderPilot: an evaluated e-commerce support agent.** A LangGraph customer-support agent for an online clothing shop (order tracking, return/cancellation eligibility, and later returns, cancellations and support tickets), built around a strong evaluation suite. Portfolio project: code quality, clarity and the evaluation story matter as much as features.

## Read these first
- `docs/project-context.md` — goal, name, stack
- `docs/scenarios.md` — business rules and all scenarios (source of truth for behaviour and tests)
- `docs/tools.md` — tool specs, reason codes, design principles
- `docs/data.md` — database schema, cleaning, time handling, known quirks

If code and docs disagree, ask before changing either.

## Stack
- Python 3.11+, LangChain + LangGraph, SQLite (stdlib `sqlite3`)
- LangChain/LangGraph APIs change often: check the current docs before using an API, don't rely on memory.
- Keep dependencies minimal.

## Non-negotiable design rules
1. **Business rules and date maths live in tool code**, never only in prompts. Tools return a decision, a `reason_code` and the supporting numbers; the LLM only explains them.
2. **`user_id` lives in graph state** (injected into tools, e.g. via `InjectedState`). It is never a tool argument the LLM can set.
3. **Auth gate:** before `verify_customer` succeeds, only that tool is bound to the model, and every other tool also refuses with `NOT_VERIFIED`.
4. **Another user's order returns `NOT_FOUND`**, the same as a non-existent order.
5. **`TODAY` comes from `meta.reference_now`** in the database. Never use `datetime.now()` in tools or evals.
6. **Policy constants in one config module:** `RETURN_WINDOW_DAYS=30`, `STUCK_DAYS=7`, `LOST_DAYS=14`, `MAX_AUTH_ATTEMPTS=3`.
7. **Write tools re-check eligibility themselves** before writing.
8. **Every scenario ID** (e.g. `P4`, `S1`) maps to eval test cases; keep the IDs in test names.
9. **Tool-call errors are returned as structured results**, not raised to the LLM as stack traces.

## Milestones
- **Milestone 1 (read-only):** `verify_customer`, `list_my_orders`, `get_order_details`, `check_cancellation_eligibility`, `check_return_eligibility`; single-agent LangGraph loop; eval harness.
- **Milestone 2 (writes):** `cancel_order`, `return_items`, `create_support_ticket`; state-based grading on a fresh DB copy per test.
- **Later:** supervisor/worker multi-agent version, compared with the single agent on the same evals.

## Open decisions (ask, don't decide silently)
- How write confirmation (scenario W8) is enforced: LangGraph `interrupt()`, a confirm step in the tool contract, or prompt-only.
- Which LLM provider/model to use.
- Eval tooling (LangSmith or open-source).

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
│   ├── config.py         # policy constants, DB path
│   ├── db.py             # connection, reference_now
│   ├── tools/            # one module per tool group
│   ├── agent/            # graph, state, prompts
│   └── policy.py         # eligibility logic shared by read and write tools
├── evals/                # scenario generators, user simulator, graders
└── tests/                # unit tests for tools and policy (no LLM)
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
```

## Coding style
- Google-style docstrings (`Args:`, `Returns:`, `Yields:`, `Raises:`) on **every** function and class, including private `_helpers` and tests. Ruff (`D` rules, `convention = "google"`) enforces this for public ones only, so write the private ones by hand.
- Type hints everywhere; `mypy --strict` and `ruff check` / `ruff format` must pass.

## Working style
- Build in small steps; tools and policy first, with unit tests against the real database, then the graph, then evals.
- Unit tests for tools must not call an LLM.
- Don't commit `data/`, `*.sqlite` or API keys.
