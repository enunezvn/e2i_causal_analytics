# Copilot pill follow-ups (#1901, mechanical slice) — design

Date: 2026-09-06. Issue: https://github.com/enunezvn/e2i_causal_analytics/issues/1901.
Parent: PR #1900 (capability-catalog prompt + pill validator + page-summary readable), spec
`docs/superpowers/specs/2026-09-05-copilot-pill-capability-catalog-design.md`.

## 1. Scope (user decision 2026-09-06)

Items 1, 2, 3, 4d, 5 and 6 of #1901 in one PR. Items 4a–4c (validator recall on the three
kept-NO categories), 4e (bare "CATE" title) and 4f (`chang*`/`month*` in the extends-list)
stay open on the issue: each trades recall for a real chance of over-blocking and needs a
fresh 92-pill grading run as its own gate.

## 2. Measured premises (cheapest disproof first, real module, 2026-09-06)

| item | assumption | measurement |
|---|---|---|
| 1 | the loaders are fast on the normal path, so a small budget never trips | client creation 130 ms; coverage 30–70 ms; outcomes ~45 ms (two fresh processes). One unreproduced 18 s cold coverage read. First prod probe call after the deploy: 3.2 s (normal). |
| 1 | the client factory is safe to call concurrently | it is NOT: `get_async_supabase_client` checks a module global and awaits `acreate_client` without a lock, so two first-callers would create two clients. Loaders stay sequential. |
| 3 | pure on-screen territory reads drop today and the page publishes what they read | `match_unsupported_rule` returns `territory_detail` for "Which of the territories shown has the largest recommended reallocation?"; the /resource-optimization summary publishes allocation count, ROI, total outcome, largest increase and largest decrease. |
| 4d | the no-preposition form slips | "Is Fabhalta outperforming the competition?" and "… outperforming competitors on market share" both return `None`. |
| 5 | an eager task factory breaks the identity guard | under `asyncio.eager_task_factory` with non-suspending loaders: published=False and `_inflight` stays set to the finished task, so the first build would be served forever (worse than the issue's "rebuild every call"). |
| 6 | section A over-promises windows | registry: 45 KPIs, 9 windowable (5 `clean`: TRx, NRx, NBRx, TRx Share, Conversion Rate; 4 `needs_care`: Trigger Precision, Acceptance Rate, Override Rate, Trigger Funnel Conversion); 36 `not_applicable`. The KPI tool answers those with `window_status="not_applicable"` (honest, no fabrication). |

## 3. Design

### 3.1 Loader timeout (item 1)
`CATALOG_LOADER_TIMEOUT_SECONDS = 5.0`. Each loader call in `build_capability_catalog` is wrapped
in `asyncio.wait_for`. `asyncio.TimeoutError` is an `Exception`, so the existing handler logs a
warning, marks the field degraded and `_keep_last_good_fields` carries the previous lists
forward; the degraded catalog is retried after `DEGRADED_TTL_SECONDS`. Sequential execution is
kept (see the factory premise). Worst case per refresh drops from ~80 s to 10 s. `wait_for`
cancels the inner coroutine; `KPIHistoryRepository.get_coverage` catches `Exception` only, so the
`CancelledError` propagates as intended.

### 3.2 Cache generation token (item 5)
`_CatalogCache` gains `_generation: Optional[object]`. `get()` assigns a fresh token before
`ensure_future`, and stores the future only if it is not already done (an eager build has
already published and cleared). `_refresh(token, …)` publishes and clears only when
`self._generation is token`. `reset()` clears the token, so an orphaned build still serves its
waiters but neither publishes nor clears a newer build's future — the semantics of the four
existing cache tests are unchanged.

### 3.3 Territory on-screen exemption (item 3)
`territory_detail` joins `_ON_SCREEN_ARTEFACT_RULES`. `_EXTENDS_ON_SCREEN_RE` already contains
`per-territory`, `by … territory`, `by … region`, `trend`, `over time`, so the extension shapes
keep dropping. Two keep fixtures (largest reallocation shown; on-screen table read) and two drop
fixtures (by census region; trend over time).

### 3.4 Competitor no-preposition form (item 4d)
Two alternations on the `competitor_data` rule: `(out|under)?perform\w* <competitor noun>` and
`beat(s|ing|en)? <competitor noun>`. Keep fixture guards the clinical-context form section E
serves ("mechanism of action differ from competitors'").

### 3.5 Window wording from data (item 6)
`CapabilityCatalog.windowable_kpi_ids` (code-derived: registry `windowable != "not_applicable"`,
never degraded). Section A renders "A time window (…) applies ONLY to: <names>; every other KPI
answers its current value and reports that a window does not apply". The window sentence leaves
`AXIS_RULES`, which keeps the axis vocabulary (`test_axis_vocabulary_matches_kpi_calculate_tool`
untouched) and the composition rule.

### 3.6 Backend CI sees the router file (item 2)
`frontend/src/router/routes.tsx` is added to both `push.paths` and the `changes` job `PATTERN` in
`.github/workflows/backend-tests.yml` (the file's comment requires the two to stay in sync).

## 4. Error handling
No new failure modes: a timeout degrades exactly like a loader exception; the token guard only
changes WHICH build may publish; the validator changes are regex alternations and a set member.

## 5. Testing and certification
- Unit: `tests/api/test_chat_capability_catalog.py` (in the CI Unit Tests lane): timeout test,
  eager-task-factory test, keep/drop fixtures, windowable-set and window-clause render tests.
- CI: full backend + frontend gates on the PR; codex audit to ACCEPT.
- Post-deploy: container content markers (the new constant, the token field, the window clause);
  the 46-call probe (`probe_pills_post_merge.py`, new output name) graded with the existing rubric —
  kept-pill NO ≤ 3.3 % and validator precision 100 % on the window are the gate.

## 6. Files
- `src/services/chat_capability_catalog.py` — items 1, 3, 4d, 5, 6.
- `tests/api/test_chat_capability_catalog.py` — tests for all of them.
- `.github/workflows/backend-tests.yml` — item 2.
