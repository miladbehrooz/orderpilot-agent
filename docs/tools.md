# E-commerce Support Agent: Tool Design (v1)

_Draft, 2026-11-12. Derived from `scenarios.md`._

## Design principles
1. **Rules and date math live in tools, not in the LLM.** Tools return the decision, a reason code and the supporting numbers (days since delivery, deadline, refund). The LLM only explains them.
2. **`user_id` comes from graph state and is never a tool argument.** Tools read it via LangGraph `InjectedState`. The model has no way to pass another user's ID (S1).
3. **Defense in depth for auth.** Before verification, only `verify_customer` is bound to the model. Every other tool also refuses with `NOT_VERIFIED` if called (AU5).
4. **"Not yours" looks like "doesn't exist".** Both return `NOT_FOUND`, so order IDs can't be probed (S1, AM2).
5. **Structured outputs with reason codes:** `ELIGIBLE`, `NOT_FOUND`, `NOT_VERIFIED`, `ALREADY_SHIPPED`, `NOT_DELIVERED`, `WINDOW_EXPIRED`, `ALREADY_RETURNED`, `ORDER_CANCELLED`. Evals check reason codes, not only the reply text.
6. **Policy constants in one config file:** `RETURN_WINDOW_DAYS=30`, `STUCK_DAYS=7`, `LOST_DAYS=14`, `MAX_AUTH_ATTEMPTS=3`. `TODAY` is read from the database's `meta.reference_now`, set when the database is built.

## Milestone 1: Read tools

### 1. `verify_customer(first_name, last_name, email, postal_code)`
- Looks the user up by email (unique in the database), then checks name and postal code against that one row. Case-insensitive, trimmed matching. The attempt counter lives in state.
- Returns `VERIFIED`, `NOT_MATCHED` (with attempts remaining, never which field was wrong), or `LOCKED`.
- On success, writes `user_id` to state via LangGraph `Command`.
- Scenarios: AU1–AU6

### 2. `list_my_orders(status=None, date_from=None, date_to=None, limit=10)`
- Returns order summaries: `order_id`, status, created/shipped/delivered dates, item count, total, and flags `is_stuck` / `is_possibly_lost`.
- Scenarios: L1, L2, L4, L6, Q1, Q2

### 3. `get_order_details(order_id)`
- Returns the order plus items: `order_item_id`, product name, brand, category, `sale_price`, item status and timestamps. A missing name/brand comes back as `null`.
- Scenarios: L3, L5, Q3, AM1, AM2, S1

### 4. `check_cancellation_eligibility(order_id)`
- Returns `{eligible, reason_code, explanation}`.
- Scenarios: P1, P2

### 5. `check_return_eligibility(order_id, order_item_ids=None)`
- Checks each item (all items if none are given): `{eligible, reason_code, refund_amount, delivered_at, days_since_delivery, return_deadline}`.
- Also takes Milestone 2 `return_requests` into account.
- Scenarios: P3–P8

Finding a specific item ("the blue thing", "the shirt") is done by combining `list_my_orders` and `get_order_details`. There is no dedicated search tool, so AM1, P8 and W4 also test multi-step reasoning.

## Milestone 2: Write tools

### 6. `cancel_order(order_id)`
- Re-runs the eligibility check inside the tool, then sets the order and its items to Cancelled.
- Scenarios: W1, W2

### 7. `return_items(order_id, order_item_ids, reason)`
- Re-runs the eligibility check, then inserts rows into a new `return_requests` table (`return_id`, `order_item_id`, `refund_amount`, `reason`, `created_at`). Original rows stay untouched.
- Scenarios: W3, W4, W5, W7

### 8. `create_support_ticket(category, description, order_id=None)`
- `category`: `stuck_order` | `lost_package` | `other`.
- At most one open ticket per order; a duplicate call returns the existing ticket.
- Scenarios: W6

## Not handled by a tool
- **Policy text:** in the system prompt for v1. A `get_store_policy` tool or RAG can come later.
- **Off-topic requests and prompt injection (S3, S4):** prompt behavior, backed by the code guards. Tools enforce the rules regardless of what the model is told.
- **Other users' data (S2):** no tool can return it.

## Open decision
- **Write confirmation (W8):** how to enforce "no action without explicit customer confirmation". This depends on the LangGraph loop design. Options:
  - LangGraph `interrupt()` before write tools (enforced by the graph)
  - A confirmation step in the tool contract (e.g. propose, then confirm)
  - Prompt-only (weakest, but measurable in evals)

## Coverage
| Tool | Scenarios |
|---|---|
| `verify_customer` | AU1–AU6 |
| `list_my_orders` | L1, L2, L4, L6, Q1, Q2, AM1, P8, W4 |
| `get_order_details` | L3, L5, Q3, AM1, AM2, S1, P8, W4 |
| `check_cancellation_eligibility` | P1, P2 |
| `check_return_eligibility` | P3–P8 |
| `cancel_order` | W1, W2 |
| `return_items` | W3, W4, W5, W7 |
| `create_support_ticket` | W6 |
| _confirmation mechanism (open)_ | W8 |
| prompt + code guards | S2, S3, S4 |
