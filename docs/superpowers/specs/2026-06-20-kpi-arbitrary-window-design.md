# Arbitrary User-Requested Time Windows for KPIs (Engine + Chatbot)

- **Date:** 2026-06-20
- **Status:** Approved design — pending spec review → implementation plan
- **Owner:** enunezvn
- **Surface (this effort):** KPI engine + CopilotKit chatbot. Dashboard UI is an explicit Phase 2 follow-up.

---

## 1. Problem & motivation

A user asked the CopilotKit chatbot: *"tell me about the NRx for Kisqali in the past 3 months."* The bot returned a Kisqali NRx figure **but over a rolling 30-day window**, buried the period mismatch in a footnote, and then asked *"which brand is this for?"* despite having already computed Kisqali's number.

Two distinct problems surfaced:

1. **The window the user asked for is silently ignored.** The KPI is *defined* as a fixed rolling-30-day metric, and the `kpi_query` allowlist bakes `INTERVAL '30 days'` into the vetted SQL string with **no date-range parameter**. So "past 3 months" is unanswerable today — not because of the data (this is 3 years of longitudinal claims data, 2023-06-18 → 2026-06-10), but because windowing was never exposed.
2. **The chatbot's presentation layer discards context** (covered separately — see §8; the relevant fix is folded into this effort because it blocks a good windowed answer).

**This spec addresses problem 1 as a first-class capability**: users can request *any* time window for the KPIs where a window is meaningful, and the chatbot answers over the requested period (or honestly says the window doesn't apply).

### Verified facts (live data + code, 2026-06-20)

- Prescription data span: **2023-06-18 → 2026-06-10** (longitudinal; arbitrary windows are well-supported by the data).
- NRx (WS3-BI-006) is brand-scoped via optional param `($1::text IS NULL OR brand::text = $1)`; window is hardcoded `event_date >= NOW() - INTERVAL '30 days'`.
- Kisqali NRx by window (first-Rx count, synthetic-included substrate): 30d ≈ 3,183 (engine) / 3,227 (direct REST, ~1% boundary nuance), **90d = 3,394**, 180d = 3,570, 365d = 3,922. All-brand 30d total = 9,337 (= 3,183 Kisqali + 3,084 Fabhalta + 3,070 Remibrutinib). The chatbot's displayed 3,183 was the **correct Kisqali** value — the failure was purely window + presentation.

---

## 2. Decisions (locked during brainstorming)

| # | Decision | Choice |
|---|---|---|
| D1 | Non-temporal KPIs (no claims time-dimension) | **Window the temporal KPIs; be honest about the rest.** Non-temporal KPIs return their value with an explicit "window not applicable" provenance — never a silently faked window. |
| D2 | Window contract | **Both rolling and absolute, normalized to a single internal `[start, end)` date range.** |
| D3 | Surface | **Engine + chatbot first**, engine designed surface-agnostic so the dashboard can adopt it later. |
| D4 | How windowed queries are produced/bound | **Code-generated additive variants** ([start,end) positional params); base vetted queries untouched. |

---

## 3. KPI time-semantics taxonomy (47 KPIs)

Classification grounds the scope. Where the YAML registry and the calculator-file audit disagreed, the **calculator audit (closest to the executing SQL) is authoritative**, and the conflict is flagged. The **authoritative per-KPI `windowable` flag is set during Phase 1 implementation** against the executing SQL.

### CLEAN — single-timestamp volumes & same-window ratios → Phase 1 IN (~18)

| KPI | Name | Window column | Note |
|---|---|---|---|
| WS3-BI-001 | MAU | session_start | view-backed primary (`v_kpi_active_users`); see §6 view caveat |
| WS3-BI-002 | WAU | session_start | view-backed primary; see §6 |
| WS3-BI-005 | TRx | event_date | clean |
| WS3-BI-006 | NRx | event_date | clean (driving example) |
| WS3-BI-007 | NBRx | MIN(event_date) | clean (first-brand Rx) |
| WS3-BI-008 | TRx Share | event_date (both legs) | ratio — bind `$start/$end` to brand_rx **and** category CTEs |
| WS3-BI-009 | Conversion Rate | trigger window | ratio — **fixed 30-day outcome look-forward stays constant**, not user-controlled |
| WS3-BI-010 | ROI | 30d | windowed avg |
| WS2-TR-001 | Trigger Precision | trigger_timestamp | clean |
| WS2-TR-004 | Acceptance Rate | trigger_timestamp | clean |
| WS2-TR-005 | False Alert Rate | trigger_timestamp | clean |
| WS2-TR-006 | Override Rate | trigger_timestamp | clean |
| WS2-TR-007 | Lead Time | trigger_timestamp | clean |
| WS2-TR-008 | Case Fatality / CFR | trigger_timestamp | clean |
| WS1-DQ-001 | Source Coverage — Patients | created_at | ratio — both legs |
| WS1-DQ-005 | Completeness Pass Rate | created_at | clean |
| WS1-MP-007 | SHAP Coverage | created_at | clean (prediction timestamp) |
| BR-005 | Kisqali Reach | trigger_timestamp | currently 90d |

### NEEDS-CARE — windowable only with a disambiguation decision → Phase 2

- **WS2-TR-002 Trigger Recall** — asymmetric (outcome window bounded, trigger look-back unbounded); must define whether the user window also bounds triggers.
- **WS2-TR-003 Action Rate Uplift** — currently unwindowed; adding a window is a new behavior (A/B arms over time).
- **BR-002 Remi Intent Δ** — primary path is a snapshot (latest survey month); only the 90d fallback branch is windowable.
- **CM-001 ATE / CM-002 CATE** — mechanically clean on `prediction_timestamp`, but the window selects **model-run cohorts, not claims** — must be labeled as such.
- **BR-001, BR-003, BR-004** — clean per audit but currently have **no** window in SQL; a window must be *added* to their rows (event_date on treatment_events).
- **WS1-DQ-007 Data Lag, WS1-DQ-009 TTR** — view-backed (`v_kpi_*`); rolling-only unless the view bucketing is reparameterized (see §6).

### NOT-APPLICABLE — no claims time-window → return value + "window not applicable" (D1)

Snapshots: WS3-BI-003 Patient Touch Rate, WS3-BI-004 HCP Coverage, WS1-DQ-002 Source Coverage HCPs, WS1-DQ-003 Cross-source Match, WS1-DQ-004 Stacking Lift, WS1-DQ-006 Geographic Consistency.
ML / quality: WS1-MP-001 ROC-AUC, WS1-MP-002 PR-AUC, WS1-MP-003 F1, WS1-MP-004 Recall@K, WS1-MP-005 Brier, WS1-MP-006 Calibration Slope, WS1-MP-008 Fairness Gap, WS1-MP-009 Feature Drift PSI, WS1-DQ-008 Label Quality (IAA).
Timeless causal: CM-003 Causal Impact, CM-004 Counterfactual, CM-005 Mediation.

---

## 4. Governance constraints (the trust boundary)

These are hard constraints the design must respect (verified in `database/migrations/044_kpi_query_allowlist.sql`, `077`, `066`):

- **G1 — Allowlist forbids client SQL.** `kpi_query` is a SECURITY DEFINER RPC; statements must come from the vetted registry (`CHECK (sql ~* '^\s*(with|select)\s')`). Windows must be expressed as registered, parameterized SQL — not injected by clients.
- **G2 — 4 positional-param ceiling.** `kpi_query` hard-stops at 4 params. Region variants already use 2 (`$1`=brand, `$2`=region). A date range needs 2 (`$start`, `$end`). `brand + region + start + end = 4` = exactly the cap. Synthetic is handled by a **separate query-id suffix** (`_include_synthetic`), not a param, so it does not consume a slot.
- **G3 — Additive, never in-place.** Base vetted queries feed certified KPI gates and are kept byte-for-byte. Windowing adds **parallel** `*_windowed` query-ids the calculator routes to (the proven migration-077 pattern), so `window=None` behaves exactly as today.
- **G4 — Text-typed params + casts.** Params arrive as text and statements cast (`$1::text`). Date params bind as text and need `$N::timestamptz` casts authored into each row; type errors surface at execution, so per-row testing is mandatory.

---

## 5. Architecture & data flow

```
"NRx for Kisqali, past 3 months"
 → chat_node (LLM): brand=Kisqali, window phrase="past 3 months"
 → kpi_calculate_tool(kpi_name, brand, region, window)
      → time_window parser: "past 3 months" → [2026-03-22, 2026-06-20)
      → recognize_kpi → WS3-BI-006
      → calculator.calculate(id, {brand, region, window})
           → resolver composes query_id variant + positional params
           → kpi_query RPC  (brand[, region], start, end)
      → KPIResult{value, status, data_source, window_applied, window_requested}
 → _kpi_result_to_response echoes brand + region + window_applied + data_source
 → synthesize_node (FIXED): {user question + tool-call args + result} → honest prose
```

### Components

1. **Config — per-KPI window spec** (`config/kpi_definitions.yaml`). Each KPI gains:
   ```yaml
   - id: WS3-BI-006
     windowable: clean          # clean | needs_care | not_applicable
     window:
       column: event_date       # timestamp the window filters
       legs: [count]            # for ratios: which CTEs/legs bind the window
       look_forward_days: null  # conversion rate: fixed outcome window (constant, NOT user-controlled)
   ```
   Default (no window requested) → existing rolling-30-day base query runs unchanged.

2. **Codegen — additive windowed variants** (extend the existing twin generator family, e.g. `scripts/gen_kpi_synthetic_exclusion.py` → add `scripts/gen_kpi_windowed_variants.py`). For each `windowable ∈ {clean, needs_care}` KPI emit:
   - `{id}_windowed`, `{id}_windowed_region`, `{id}_windowed_include_synthetic`, `{id}_windowed_region_include_synthetic`
   - filtering `window_column >= $start AND window_column < $end`, binding the **same** `$start`/`$end` across all ratio legs, preserving fixed `look_forward` constants.
   - Output: one new migration `0xx_kpi_windowed_variants.sql`. Base queries untouched (G3). Param budget ≤ 4 (G2).

3. **Calculator + resolver** (`src/kpi/calculator.py`, `src/kpi/calculators/*.py`, `src/kpi/synthetic_mode.py`). Extend the `resolve_kpi_query_id` / `_region_variant` seam with a `windowed` dimension that composes the right query-id suffix(es). `calculate(id, ctx)` reads `ctx["window"]={start,end}`; routes to the windowed variant + appends `[start,end)` params when the KPI is windowable and a window is present; else base. `KPIResult` gains `window_applied: {start,end} | None` and `window_requested`.

4. **Window parser** (`src/services/time_window.py`, new). Normalizes rolling + absolute → `[start, end)`:
   - Rolling: "last N days/weeks/months", "past 3 months" → `[now − N, now)` (anchored to now).
   - Absolute: "Q1 2025", "Jan–Mar 2025", "2024", explicit ISO dates → fixed `[start, end)`.
   - Returns a normalized object + a human label; raises a clear error on unparseable / invalid (start>end, future-only) input.

5. **Chatbot wiring** (`src/api/routes/chatbot_tools.py`, `src/api/routes/copilotkit.py`):
   - `kpi_calculate_tool` gains a `window` arg (phrase or `{start,end}`), parses it, passes to the calculator.
   - `_kpi_result_to_response` echoes `brand`, `region`, `window_requested`, `window_applied`, `data_source` (also fixes the pre-existing provenance-omission bug).
   - **`synthesize_node` fix:** build the synthesis prompt from `{original user question, assistant tool-call args, tool results}` — not the tool results alone — so it stops asking for a brand it already used and can contrast requested-vs-actual window.
   - **System prompt:** list `kpi_calculate_tool`; instruct brand/window extraction + echo; instruct leading with window limitations / "window not applicable" honesty; drop the blanket "Suggest follow-up questions" boilerplate.

---

## 6. Error / empty / honesty handling (D1)

- **Empty window** (no rows in range) → honest **zero** with `window_applied` set — *not* an error (matches existing fail-closed semantics).
- **Unparseable / invalid window** (start > end, future-only) → the tool returns a clear error suggesting accepted formats; it **never** silently falls back to 30 days without saying so.
- **`not_applicable` KPI + window requested** → return the value with `window_applied=None` and an explicit reason ("this is a current snapshot / model-eval metric, not time-windowed").
- **View-backed KPIs** (MAU/WAU primary, `v_kpi_*` snapshots, Data Lag, TTR) cannot honor *absolute* windows without view changes. Phase-1 handling: support **rolling** windows via their direct-SQL twins where available; otherwise mark `needs_care` and defer. Document explicitly which KPIs are rolling-only in Phase 1.

---

## 7. Testing & rollout

### Cheapest-disproof GATE (do this FIRST, before generating ~100 variants)

The load-bearing assumption is **"the window can be a parameter."** It currently cannot (G1/G4). Author **one** windowed row (`business_impact_nrx_windowed` with `$start`/`$end`), apply to a scratch/branch DB, and confirm `kpi_query` binds + casts text date params through its `EXECUTE … USING` ladder and returns the correct count (e.g. 90-day Kisqali NRx = **3,394**). If this fails, stop and reconsider the approach (e.g. fall back to the new-RPC option) before building codegen.

### Tests

- **Unit:** window parser (phrases/ISO/quarters → `[start,end)`, invalid inputs raise); resolver (query-id composition across windowed×region×synthetic); `_kpi_result_to_response` provenance echo.
- **Faithful SQL:** each windowed variant's count matches direct SQL and differs correctly from the 30-day base (e.g. 90d Kisqali NRx = 3,394 ≠ 30d 3,183); ratio variants window both legs.
- **Regression:** base (no-window) query-ids byte-for-byte unchanged; certified KPI gates green (CI `Causal Validation Benchmarks` + KPI coverage).
- **Chatbot:** synthesis prompt includes the question + tool-call args; brand is echoed (no "which brand?" re-ask); "past 3 months" returns the 90-day value, correctly labeled; a `not_applicable` KPI returns value + honest window note.

### Phasing

- **Phase 1 (this effort):** the ~18 CLEAN volume + same-window-ratio KPIs (§3) + the full chatbot wiring (parser, tool window arg, provenance echo, synthesize fix, prompt). View-backed KPIs rolling-only or deferred per §6.
- **Phase 2 (follow-up):** the `needs_care` KPIs (Trigger Recall, Action Rate Uplift, BR-001/002/003/004, CM-001/002 labeled), view reparameterization for absolute windows, and the dashboard date-range UI.

---

## 8. Related (not in scope here, but adjacent)

The same review found generic CopilotKit presentation bugs that degrade *every* tool-using answer: `synthesize_node` reasoning from tool-JSON only, the tool result omitting provenance, system-prompt gaps, and the FE-selected brand/filters being dropped (`context=[]` in `execute()`). The first three are **folded into this effort** (§5) because they block a good windowed answer. The dropped-UI-context item (forwarding CopilotKit readables/instructions to the backend LLM) is tracked as a separate chatbot-quality fix.

---

## 9. Key files

- `database/migrations/044_kpi_query_allowlist.sql` — RPC + seed (trust boundary, 4-param cap).
- `database/migrations/077_kpi_region_variants.sql` — additive-variant precedent (2 param slots used).
- `scripts/gen_kpi_synthetic_exclusion.py` — existing twin codegen to extend.
- `src/kpi/synthetic_mode.py` (`resolve_kpi_query_id`), `src/kpi/calculators/*.py` (`_region_variant`, `_execute_query`) — resolution seam.
- `config/kpi_definitions.yaml` — where the `windowable` flag + `window` block live.
- `src/api/routes/chatbot_tools.py` (`kpi_calculate_tool`, `_kpi_result_to_response`), `src/api/routes/copilotkit.py` (`synthesize_node`, `E2I_COPILOT_SYSTEM_PROMPT`).
- `src/services/time_window.py` — new window parser.
