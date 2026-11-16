# OrderPilot: an evaluated e-commerce support agent

## Run the agent locally

1. Set up the project and the database (see [CLAUDE.md](CLAUDE.md) for all commands):
   ```bash
   uv sync
   cp sample.env .env    # then set ANTHROPIC_API_KEY (LLM_PROVIDER / LLM_MODEL are preset)
   uv run python scripts/build_db.py --data-dir data --out ecommerce.sqlite --today 2026-11-12
   ```
2. Start the LangGraph dev server:
   ```bash
   uv run langgraph dev
   ```
   It serves the `orderpilot` graph at `http://127.0.0.1:2024` and keeps conversations
   (threads) in memory while it runs. Code changes reload automatically.
3. Open a UI. Both talk to the same server:

   | UI | Use it for | How |
   |---|---|---|
   | **LangGraph Studio** | Debugging: graph view, state after each step (`user_id`, `failed_auth_attempts`), tool calls, time travel | Opens automatically, or go to `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`. Needs a free LangSmith account; set `LANGSMITH_API_KEY` in `.env`. |
   | **Agent Chat UI** | Chatting like a customer, demos | Go to `https://agentchat.vercel.app`, enter Deployment URL `http://127.0.0.1:2024` and Graph ID `orderpilot`. No account needed. |

   Safari blocks pages on `https://` from reaching `localhost`. Use Chrome or Firefox, or run
   `uv run langgraph dev --tunnel` and use the tunnel URL it prints instead.

A terminal chat is also available: `uv run python -m orderpilot`.

**Tracing:** set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` to send every run
(model calls, tool calls, tokens, latency) to the LangSmith project `LANGSMITH_PROJECT`. This
works for every entry point, since building the agent copies these values into the environment.
Tests never trace.

The graph's input accepts only `messages`. Other keys sent by a client (for example `user_id`)
are dropped, so a customer can only be identified through `verify_customer`.

## Evaluation

The agent is evaluated with LangSmith experiments; the design is in
[docs/evals.md](docs/evals.md). Hand-written cases live in `evals/datasets/` (the source of
truth) and are graded by evaluator classes in `evals/evaluators/`. The first metric,
`answer_correctness`, is an LLM judge (OpenAI) that checks each answer against a hand-written
reference.

1. In `.env`, set `LANGSMITH_API_KEY`, `OPENAI_API_KEY` and `EVAL_JUDGE_MODEL` (see `sample.env`).
2. Copy the cases to LangSmith, then run an experiment:
   ```bash
   uv run python -m evals.sync --dataset m1
   EVAL_REPEATS=1 uv run python -m evals.run --split smoke    # quick check
   uv run python -m evals.run --split full                    # k = EVAL_REPEATS (default 3)
   ```
Each case runs as a single question from an already-verified customer, k times. The report
prints pass rate, pass^k (every run passes) and pass@k (at least one passes) per scenario, and
attaches the overall numbers to the experiment in LangSmith. The dataset is tied to the
database build: `sync` and `run` refuse to run if `reference_now` differs.
