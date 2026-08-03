# #1407 clarification probe — 2026-08-03

Targeted probe of the `/chat/stream` clarification ask-back shipped in **#1407**
(`23494dd5`, 2026-08-02), run against the four demo questions a deterministic
pre-check had flagged as candidates for the gate: **4.3, 5.5, 5.7, A.5**.

**Question: did #1407 deliver the intended clarification without causing
over-abstention on questions that previously answered fine?**

**Answer: yes on both counts.** A.5 clarifies and resumes to a real answer; no
suite question clarifies in narrative order except A.5.

## Method

7 turns total on `POST /api/copilotkit/chat/stream` (prod, `eznomics.site`).

1. **Cold probes** (`clarify_probe.py`) — 4.3, 5.5, 5.7, A.5, each in its **own
   fresh session** so `has_prior_referent` is false. This is the worst case for
   over-abstention.
2. **Resume probe** (`clarify_probe2.py` B1) — answered A.5's pending ask-back
   with `"Kisqali TRx"` in the same session.
3. **Narrative control** (`clarify_probe2.py` B2) — seeded a fresh session with
   5.3 (names "Kisqali"), then asked 5.7, reproducing real demo order.

Verdicts use **three converging methods**, not prose matching: the SSE
`dispatch_info` frame (`orchestrator_used`, `agents_dispatched`, `intent`), the
answer text, and the persisted `chatbot_conversations.metadata ->
pending_clarification` row queried directly from `supabase-db`.

## Results

| Q | Cold turn | `orchestrator_used` | pending row | Narrative order | Verdict |
|---|---|---|---|---|---|
| **A.5** | **CLARIFIES** 5.6 s | false | ✅ `["brand","metric"]` | always cold by design | ✅ intended win |
| **5.7** | CLARIFIES 7.9 s | false | ✅ `["brand","metric"]` | **ANSWERS** — 55.0 s, dispatched `causal_impact`+`explainer`, real refutation answer, no ask-back | ✅ no regression |
| **5.5** | CLARIFIES 33.7 s | false | ✅ `["brand","metric"]` | suppressed (same mechanism) | ✅ no regression |
| **4.3** | does not clarify, 79.9 s | true (`heterogeneous_optimizer`, `gap_analyzer`) | ✗ none | same | ✅ unrelated to #1407 |

### A.5 — the intended win, round-trip verified

Cold turn returned an honest ask-back naming both missing slots in 5.6 s with
**no orchestrator dispatch** (baseline behaviour was a ~13–17 s dispatch that
failed closed). Answering `"Kisqali TRx"` resumed on the merged query,
dispatched `causal_impact`+`explainer`, returned a grounded **TRx = 12,867**,
and **cleared the pending row**. The clarification is not a dead end.

### No over-abstention in the demo flow

5.5 and 5.7 clarify *only* as isolated cold turns. In their real narrative
session, `_has_analytical_referent` finds a brand/metric token in earlier turns
(4.1/4.2 carry "TRx"; 5.3/5.4 carry "Kisqali") and suppresses the gate. This was
predicted deterministically and then **confirmed live** for 5.7.

### 4.3 never reaches the gate — and a caution about offline prediction

The cold set was first derived from the **rule-based fallback** classifier, which
predicted `kpi_query` for 4.3. Production's **DSPy** classifier returned
`multi_faceted` (0.82), which is outside `CLARIFY_INTENTS`, so 4.3 never touched
the gate. Its 79.9 s runtime and `#1336` fail-closed-bridge preamble are
pre-existing `/chat/stream` behaviour, not a #1407 effect.

**Lesson for future offline analysis:** the deterministic fallback is a
hypothesis generator, not a prediction of live routing.

## Defects found alongside — all filed 2026-08-03

Root-caused from `e2i_api` logs for the probe requests, then filed:

| Issue | Defect | Root cause |
|---|---|---|
| **#1447** | `health_score` answers "System health is critical (Grade: F, 0.0/100)" while `GET /health` shows every component operational | `model_health` correctly returns **UNKNOWN** (`No metrics_store wired`, fail-closed per #883) and `score_composer` then collapses UNKNOWN → `score=0.0, grade=F, 1 critical issue`. An unmeasurable component is narrated as a measured catastrophe. Also scope-blind: the run was `scope: models`, the prose says "System health" |
| **#1448** | 3 Tier-0 agents (`model_selector`, `model_trainer`, `model_deployer`) dropped from the prod registry on every request | `project_root.py` walks for a `pyproject.toml` marker that is absent under `/app` in the image → construction raises → PARTIAL registry (18 of 21 agents). Not chat-routable today, so no demo impact |
| **#1449** | 4.3 misroutes and burns ~80 s reaching an avoidable fail-closed | Routed to `heterogeneous_optimizer`+`gap_analyzer` (gold: `cohort_profiler`); both fail closed on preconditions (*no brand*, *no treatment column*) that were knowable **before** dispatch |
| **#1450** | No chat surface answers model-quality questions (ROC-AUC / calibration / Brier) | 5.3's routing is correct (`system_health` 0.92, gold `health_score`) but no metrics store is wired and the agent emits a composite grade, not the requested metrics |
| **#1451** | The #1336 bridge preamble reads as failure on turns whose bridged answer is correct | Preamble describes the *internal pipeline's* outcome; the good grounded answer (e.g. TRx = 12,867) is buried beneath an apology |
| **#1452** | `health_score` MLflow metrics logging fails every run | `[Errno 30] Read-only file system: '/mlflow'` — should use the running tracking server via `MLFLOW_TRACKING_URI`, or a rw named volume |

Note the fail-closed dispatcher messages themselves are **correct #883 behaviour**,
not defects: a bare chat query genuinely cannot name the treatment/outcome/confounder
columns a causal spec needs. #1449 is about paying ~80 s to discover that.

## Files

| File | Contents |
|---|---|
| `raw_cold_probes.json` | 4 cold-turn records (full SSE events, `dispatch_info`, answers) |
| `raw_resume_and_narrative.json` | A.5 resume + 5.3-seed/5.7 narrative control |
| `pattern_diff_20260803.jsonl` | Deterministic legacy routing over all 337 gold rows, current `main` |
| `clarify_probe.py`, `clarify_probe2.py` | The probes, rerunnable |

Deterministic routing measured the same day: pattern accuracy **0.852** full set
(was 0.843) / **0.907** deterministic subset (was 0.891); TOOL_COMPOSER recall
**0.321, unchanged**. See `COPILOT_CHAT_DEMO_SCENARIOS_V2.md`.
