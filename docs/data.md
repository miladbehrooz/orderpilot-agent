# OrderPilot: Data and Database

## Source
Looker **theLook E-commerce** dataset (BigQuery public dataset `bigquery-public-data.thelook_ecommerce`, also on Kaggle). Synthetic clothing shop data.

Used: `users`, `orders`, `order_items`, `products`, `distribution_centers`.
Not used: `events` (web clickstream), `inventory_items` (optional, not needed by any tool).

## Building the database
```bash
python scripts/build_db.py --data-dir data --out ecommerce.sqlite --today 2026-11-12
```
- CSVs go in `data/` (git-ignored). The SQLite file is git-ignored too; always rebuild it from the script.
- Omitting `--today` uses the current UTC date.
- The build validates itself and exits non-zero if any check fails.

## Tables
| Table | Rows | Notes |
|---|---|---|
| `users` | 100,000 | `email` is UNIQUE (case-insensitive); `postal_code` is TEXT |
| `orders` | 125,226 | 1–4 items per order; ~20% of users have no orders |
| `order_items` | 181,759 | Same status and timestamps as its order (until M2 partial returns) |
| `products` | 29,120 | 26 clothing categories, Men/Women; 2 NULL names, 24 NULL brands |
| `distribution_centers` | 10 | US warehouses |
| `return_requests` | 0 | Milestone 2; one row per returned item (`order_item_id` UNIQUE) |
| `support_tickets` | 0 | Milestone 2; at most one open ticket per order |
| `meta` | 5 | `reference_now`, `date_shift_days`, `source_latest_event`, `email_seed`, `built_at` |

Relationships: `users → orders → order_items → products → distribution_centers`.

## Order lifecycle
| Status | Share | Timestamps set |
|---|---|---|
| Processing | 20% | created |
| Shipped | 30% | created, shipped |
| Complete | 25% | + delivered |
| Returned | 10% | + returned |
| Cancelled | 15% | created |

Typical timing: ~1.4 days to ship, ~2.5 days in transit, returns 0–3 days after delivery.

## Time
- All timestamps are UTC text `YYYY-MM-DD HH:MM:SS`.
- Every timestamp was shifted by one constant offset (whole weeks) so the latest event lands just before `reference_now`.
- **`TODAY` = `meta.reference_now`.** Tools and evals must read it from the database and never use the system clock, so results are reproducible.

## Cleaning applied
1. `order_items.created_at` set to its order's `created_at` (the source value was 0–4 days later than the order).
2. Timestamps normalised to UTC, truncated to seconds.
3. Prices rounded to 2 decimals.
4. `Deutschland`/`España` mapped to `Germany`/`Spain`.
5. Empty product name/brand kept as `NULL`.
6. Duplicate emails (15,989 users) made unique: the earliest sign-up keeps the original, later users get a realistic name-based variant (`first.last`, `flast84`, `first.last1984`, …) on the same reserved `example.*` domain. Birth year = reference year − age. Seeded, so builds are reproducible.

## Known quirks (kept on purpose; scenarios rely on them)
- **Stuck orders:** many Processing/Shipped orders are months or years old and never closed (scenarios Q1, Q2, W6).
- **Products with NULL name/brand** (Q3).
- **Same name/brand/price under different product IDs** (300 groups): natural ambiguity (AM1).
- **No partial returns in the source:** all items in an order share one status (P8, W4 create the first ones).
- **No refund amounts, carriers or tracking numbers:** refund = item `sale_price`; tracking is status + dates.
- **~10k users aged 12–17:** synthetic, kept.
- **Two near-free products** ($0.02, $0.49): kept.
