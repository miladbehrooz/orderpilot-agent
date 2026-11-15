# E-commerce Support Agent: Tool Design (v1)

_Draft, 2026-11-12. Derived from `scenarios.md`._

## Design principles
1. **Rules and date math live in tools, not in the LLM.** Tools return the decision, a reason code and the supporting numbers (days since delivery, deadline, refund). The LLM only explains them.
2. **`user_id` comes from graph state and is never a tool argument.** Tools read it via LangChain `ToolRuntime` (`runtime.state`), which replaces the older `InjectedState`. The model has no way to pass another user's ID (S1).
3. **Defense in depth for auth.** Before verification, only `verify_customer` is bound to the model. Every other tool also refuses with `NOT_VERIFIED` if called (AU5).
4. **"Not yours" looks like "doesn't exist".** Both return `NOT_FOUND`, so order IDs can't be probed (S1, AM2).
5. **Structured outputs with reason codes:** `OK` (a lookup succeeded), `ELIGIBLE`, `NOT_FOUND`, `NOT_VERIFIED`, `ALREADY_SHIPPED`, `NOT_DELIVERED`, `WINDOW_EXPIRED`, `ALREADY_RETURNED`, `ORDER_CANCELLED`. Auth adds `VERIFIED`, `NOT_MATCHED`, `MISSING_DETAILS`, `LOCKED`, `ALREADY_VERIFIED`. Any tool can return `INVALID_ARGUMENTS` (bad tool-call arguments) or `INTERNAL_ERROR` (unexpected failure, logged, details hidden). Evals check reason codes, not only the reply text.
6. **Policy constants in one config file:** `RETURN_WINDOW_DAYS=30`, `STUCK_DAYS=7`, `LOST_DAYS=14`, `MAX_AUTH_ATTEMPTS=3`. `TODAY` is read from the database's `meta.reference_now`, set when the database is built.

## Milestone 1: Read tools

### 1. `verify_customer(first_name, last_name, email, postal_code)`
- Looks the user up by email (unique in the database), then checks name and postal code against that one row. Case-insensitive, trimmed matching. The attempt counter lives in state.
- Returns `VERIFIED`, `NOT_MATCHED` (with attempts remaining, never which field was wrong), or `LOCKED`. The failure that reaches `MAX_AUTH_ATTEMPTS` returns `LOCKED`; after that the graph answers with a fixed reply and never calls the model.
- A blank detail returns `MISSING_DETAILS` and does not use up an attempt (AU6).
- On success, writes `user_id` to state via LangGraph `Command`. `user_id` is never part of the tool's output.
- After success the tool is no longer bound to the model, and a call anyway returns `ALREADY_VERIFIED` (no switching customers mid-conversation).
- Scenarios: AU1–AU6

### 2. `list_my_orders(status=None, date_from=None, date_to=None, limit=10)`
- Arguments are validated by the tool schema: `status` is one of the five order statuses, `date_from`/`date_to` are `YYYY-MM-DD` dates (inclusive, on the order date), `limit` is 1–50. Invalid values return `INVALID_ARGUMENTS`, and so does `date_from` after `date_to`.
- Returns `OK` with `orders` (newest first) and `has_more` (more orders match than `limit`). No orders is `OK` with `orders: []` (L6).
- Each order summary: `order_id`, `status`, `created_on` / `shipped_on` / `delivered_on` / `returned_on` (dates, omitted when unset), `item_count`, `total`, `days_in_status` (calendar days since the current status began), flags `is_stuck` / `is_possibly_lost`, and `refund_total` when items were returned.
- Scenarios: L1, L2, L4, L6, Q1, Q2

### 3. `get_order_details(order_id)`
- Returns `OK` with `order`: the same summary as above plus `items`, each with `order_item_id`, `product_name`, `brand`, `category`, `sale_price`, `status` and its dates. A missing name/brand is an explicit `null` (Q3), so the model can see it is missing.
- An order that doesn't exist or belongs to another user returns `NOT_FOUND` with nothing else; the two cases are identical (AM2, S1).
- Never returns `user_id`, `product_id` or `inventory_item_id`.
- Scenarios: L3, L5, Q3, AM1, AM2, S1

The stuck/lost flags and day counts come from pure functions in `src/orderpilot/policy.py` (counted in calendar days against `meta.reference_now`), which the eligibility and write tools will share.

Both eligibility tools only check; they never change an order. Their decisions come from `cancel_decision` / `return_decision` in `src/orderpilot/policy.py`, which the Milestone 2 write tools re-run before writing (rule 7). There is no `explanation` text: the tools return codes and facts, and the LLM explains them.

### 4. `check_cancellation_eligibility(order_id)`
- Returns `reason_code`, `eligible`, `order_id`, `status`, `created_on`, `shipped_on`, `delivered_on`.
- Only Processing orders can be cancelled (a stuck one too). Codes by status:

  | Status | `reason_code` |
  |---|---|
  | Processing | `ELIGIBLE` |
  | Shipped, Complete | `ALREADY_SHIPPED` (it can be returned once delivered) |
  | Returned | `ALREADY_RETURNED` |
  | Cancelled | `ORDER_CANCELLED` |
- An order that doesn't exist or belongs to another user returns only `NOT_FOUND`.
- Scenarios: P1, P2

### 5. `check_return_eligibility(order_id, order_item_ids=None)`
- Checks each requested item (all items if `order_item_ids` is omitted; duplicates are ignored). Each item: `order_item_id`, `product_name` (explicit `null` if missing), `category`, `sale_price`, `eligible`, `reason_code`, `refund_amount` (only when eligible; equals `sale_price`), `delivered_on`, `days_since_delivery`, `return_deadline`.
- Item codes, first match wins:
  1. Returned, or a row in `return_requests` → `ALREADY_RETURNED`
  2. Cancelled → `ORDER_CANCELLED`
  3. Not delivered (Processing, Shipped, or no delivery date) → `NOT_DELIVERED`
  4. More than `RETURN_WINDOW_DAYS` calendar days since delivery → `WINDOW_EXPIRED`. Day 30 is still eligible, day 31 is not; `return_deadline = delivered_on + 30 days`.
  5. Otherwise → `ELIGIBLE`
- Top level: `order_id`, `eligible` (any item eligible), `refund_total` (sum over eligible items) and a `reason_code` that is `ELIGIBLE` if any item is eligible, otherwise the code the items share (or the first item's, if they differ).
- `NOT_FOUND` with nothing checked if the order isn't the customer's, or if any requested item is not in that order (`detail` lists those ids). This matches the all-or-nothing behaviour of `return_items`. An empty `order_item_ids` list is `INVALID_ARGUMENTS`.
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
