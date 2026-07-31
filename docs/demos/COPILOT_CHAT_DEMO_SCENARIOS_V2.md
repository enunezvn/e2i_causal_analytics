# Copilot Chat — Demo Scenarios & Performance Suite (v2)

**Supersedes** `COPILOT_CHAT_DEMO_SCENARIOS.md` (v1, retained as a point-in-time
record). **Closes #1345.**

v1 was hand-authored before three things that each invalidated parts of it:

1. the **recorded baseline** (`docs/demos/results/2026-07-29_copilot_chat_perf/`
   — every question actually run, timed, and transcribed),
2. the **two-chat-surfaces finding** (the real UI never calls `/chat/stream`), and
3. the **verified contract registry + 337-row routing gold set** from the #1337
   Step 0 benchmark (`scripts/benchmarks/routing/data/agent_contracts.json`,
   `scripts/benchmarks/routing/data/benchmark_queries_gold.jsonl`).

Every routing annotation below is **regenerated from those artifacts**, not
hand-distilled. The per-question tables carry:

- **Gold routing** — the pattern + owning agent(s) each query SHOULD route to,
  from `benchmark_queries_gold.jsonl` (48 gold rows cover the suite, keyed by
  `demo_meta.question_id`; A.2 is split into A.2a/A.2b, A.9's follow-up is the
  gold row, and A.7/A.8 are not gold rows — UI-only and measured separately.
  Label provenance per row is in `gold_source` — for the demo rows:
  `human-ratified` 19, `llm-judge` 25, `llm-judge+human-confirmed` 2,
  `human-triage` 2 — see `scripts/benchmarks/routing/GOLD_STAGE_RESULTS.md`).
- **Legacy routes to (today)** — what the deterministic legacy chain
  (`IntentClassifierNode._pattern_classify` → `RouterNode.execute` →
  `derive_legacy_pattern`) produces on current `main`, regenerated with
  `scripts/benchmarks/routing/pattern_diff.py` (zero LLM calls). Rows marked
  **(LLM)** score below the 0.8 pattern-trust floor and escalate to the haiku
  fallback in production — their live route is LLM-dependent, so the
  deterministic prediction shown is the *fallback default*, not a guarantee.
- **AG-UI baseline** — the tool(s) the real UI brain actually used and the
  measured total, from `results/2026-07-29_copilot_chat_perf/measurements.csv`
  (answers/transcripts in `transcripts.md` of that dir).

Where **Gold ≠ Legacy today**, that is a *known, measured* routing miss of the
incumbent (aggregates below) — do not "fix" the annotation; fix the router.

---

## The two chat surfaces (read this before demoing anything)

Established by the 2026-07-29 recording (`results/2026-07-29_copilot_chat_perf/SUMMARY.md`)
and verified against `src/api/routes/copilotkit.py`; full endpoint reference in
`docs/api/chat.md`:

| Surface | Endpoint | Brain | Use it for |
|---|---|---|---|
| **Real copilot UI** (what a demo audience sees) | `POST /api/copilotkit/agent/default` (CopilotKit/AG-UI protocol; `threadId` ≡ DB `session_id`; full history resent each turn) | `chat_node` + `E2I_CHATBOT_TOOLS` + `synthesize_node` | **Answers.** Tool-grounded responses; the orchestrator runs only when the model elects `orchestrator_tool` |
| **Scripted chat API** | `POST /api/copilotkit/chat/stream` (SSE) | `classify → orchestrator → generate` (`chatbot_graph.py`) | **Routing measurement.** The deterministic instrument for intent/classifier telemetry (`dispatch_info` frame) |

Consequences:

- **Demo answers are recorded on the AG-UI surface only.** `/chat/stream`
  fails closed on conversational queries *by design* (#883: no structured
  inputs → no fabricated analysis). Since PR #1394 (issue #1336, owner
  decision: BRIDGE), a **complete** orchestrator failure on `/chat/stream`
  falls back to the AG-UI brain behind an honest preamble — partial and full
  successes are untouched, so the routing record stays clean.
- **Routing expectations are asserted on `/chat/stream`** via the
  `dispatch_info` SSE frame (`routing_pattern`, `classification_latency_ms`,
  `used_llm_layer` — real since PR #1330) and `classification_logs` rows.
- On AG-UI, "agents dispatched" is usually **empty**: the model answers from
  bound tools (`kpi_calculate_tool`, `causal_analysis_tool`, …) and only 2 of
  51 baseline turns (2.6, 3.4) elected `orchestrator_tool`. The AG-UI tool
  choice is **LLM-judged from tool docstrings**, not regex-gated — expect
  reasonable variance between runs.

## Routing semantics the annotations rely on

Verified in source (file references are current `main`):

- **Two routing layers coexist** on `/chat/stream`
  (`src/agents/orchestrator/nodes/intent_classifier.py`,
  `src/agents/orchestrator/nodes/router.py`): the **legacy** regex-intent
  router (`INTENT_PATTERNS` → `INTENT_TO_AGENTS`, haiku LLM fallback when
  pattern confidence < 0.8) always runs; the **4-stage ClassificationPipeline**
  runs when `ORCHESTRATOR_CLASSIFIER_MODE` is `shadow` (default; logged, never
  routes) or `active` (routes only at confidence ≥ 0.5 and never on
  CLARIFICATION_NEEDED — otherwise it **abstains to legacy**,
  `RouterNode.MIN_ACTIVE_CONFIDENCE`).
- **TOOL_COMPOSER has a multi-domain gate** (v1's T4 description was wrong):
  a query is TOOL_COMPOSER only when it spans **≥ 2 distinct capability
  domains AND the cross-domain sub-questions are dependency-linked**
  (`pattern_selector.py` Rules 3/5/6; `composition_ruling` in
  `agent_contracts.json`). Single-domain multi-step is SINGLE_AGENT no matter
  how many dependent internal steps it has — the domain agent runs its own
  pipeline. Step-counting is NOT the criterion. Multi-domain *without*
  dependencies is PARALLEL_DELEGATION.
- **health_score vs drift_monitor boundary**: health_score answers
  *point-in-time snapshot* asks; drift_monitor answers *shift-over-baseline*
  asks. In the legacy patterns: `model.*perform` → `system_health` →
  health_score; `model.*degrad` → `drift_check` → drift_monitor
  (`intent_classifier.py` INTENT_PATTERNS).
- **CLARIFICATION is structurally unreachable on the legacy path** (open issue
  #1407): `derive_legacy_pattern` only emits SINGLE_AGENT / PARALLEL_DELEGATION
  / TOOL_COMPOSER, and in active mode a pipeline CLARIFICATION_NEEDED causes
  *abstention to legacy* rather than a clarify flow. **No v2 annotation
  promises a clarification response from the orchestrator path.** Gold rows
  labeled CLARIFICATION_NEEDED (A.1, A.2a, A.5) document what routing *should*
  do, not what any surface delivers today; the AG-UI brain may still clarify
  conversationally (model-elected, not routed).

### Which agents each path can reach

From `agent_contracts.json` (14-agent registry, SSOT) and
`pattern_selector.py DOMAIN_TO_AGENT` / `router.py INTENT_TO_AGENTS`:

| Agent | Legacy router | 4-stage classifier | Notes |
|---|---|---|---|
| causal_impact, heterogeneous_optimizer, gap_analyzer, experiment_designer, prediction_synthesizer, drift_monitor, explainer, cohort_profiler | yes | yes (`DOMAIN_TO_AGENT`, 8 domains) | |
| tool_composer | yes (`multi_faceted`) | yes (TOOL_COMPOSER pattern target) | |
| resource_optimizer, health_score, feedback_learner, experiment_monitor | yes | **no — unmapped in DOMAIN_TO_AGENT** | classifier-active routing can never select these; legacy (or abstention to legacy) is the only chat path |
| cohort_constructor, scope_definer, data_preparer, model_selector, model_trainer, feature_analyzer, model_deployer, observability_connector | no | no | not chat-routable (ML pipeline / deliberately bypassed — chat `cohort_definition` routes to cohort_profiler, not cohort_constructor) |

The AG-UI brain reaches agents only through `orchestrator_tool` (same routing
stack once inside); its ordinary answers come from bound tools, not agents.

### Benchmark context (why "Gold ≠ Legacy" rows exist)

From the #1337 Step 0 benchmark and post-remediation reruns (reproduced on
current `main` with `pattern_diff.py`, 2026-07-31): legacy pattern accuracy
over the 337-row gold is **0.843** (full set) / **0.891** (deterministic
subset, n=193), with TOOL_COMPOSER precision **1.000** but recall 0.321 — the
incumbent under-fires the composer and never abstains. The 2026-07-29 shadow
recording predates the remediation chain (PRs #1347–#1372), so its
`routing_pattern` column shows the *old* 56%-CLARIFICATION pipeline — treat it
as the point-in-time record it is.

---

## Latency / complexity tiers

> **⚠ Budgets pending re-baseline — #1338.** The 2026-07-29 run measured T1 0/2
> and T2 4/12 against these budgets (structurally unachievable: every AG-UI
> turn includes ≥ 1 LLM round-trip). The budget **values below are carried over
> from v1 UNCHANGED** and are owned by #1338 — do not tune them here.

| Tier | Path exercised | What "good" looks like |
|------|---------------|------------------------|
| **T1 — Conversational** | AG-UI: model answers directly, no tools | Near-instant first token; correct capability description |
| **T2 — KPI fast path** | AG-UI: one `kpi_calculate_tool` / `e2i_data_query_tool` call | A real number with period + comparison, chart where applicable |
| **T3 — Single agent** | routing: SINGLE_AGENT; AG-UI: one analytical tool call | Streaming progress visible; structured result |
| **T4 — Multi-domain composite** | routing: TOOL_COMPOSER (**multi-domain + dependency-linked** — see gate above) or PARALLEL_DELEGATION (multi-domain, independent) | Visible decomposition; each facet resolved; coherent synthesis |
| **T5 — Context / memory** | follow-up turns and paraphrase repeats against session memory (AG-UI full-history resend) | Correct referent resolution; paraphrase repeats acknowledge/reuse the earlier analysis with consistent grounded numbers — **measured PASS 5/5, 2026-07-31** (`results/2026-07-31_t5_paraphrase_repeat/RESULTS.md`; #1339 closed measured-not-needed) |
| **T6 — Robustness** | typos, ambiguity | Typos silently corrected; ambiguous queries produce honest scope-limiting, not hallucination (**routed** clarification is not deliverable today — #1407) |

Record per question: time-to-first-token, time-to-first-progress (T3+), total,
and — on `/chat/stream` — the `dispatch_info` fields (`routing_pattern`,
`classification_latency_ms`, `used_llm_layer`, `agents_dispatched`).

---

## Narrative 1 — Kisqali brand pulse: "the Monday morning question"

**Persona:** Kisqali brand lead. **Arc:** what's happening → why → who's
affected → what do we do.

| # | Question | Gold routing | Legacy routes to (today) | AG-UI baseline (tools · total) | Tier |
|---|----------|--------------|--------------------------|-------------------------------|------|
| 1.1 | What is TRx for Kisqali? | SINGLE_AGENT → explainer | explanation → explainer ✓ | kpi_calculate_tool · 7.4 s | T2 |
| 1.2 | What are the TRx trends for Kisqali in Q4? | SINGLE_AGENT → explainer | explanation → explainer ✓ | kpi_calculate_tool · 7.1 s | T2 |
| 1.3 | What is the market share for Kisqali compared to competitors? | SINGLE_AGENT → explainer | explanation → explainer ✓ | kpi_calculate_tool · 8.1 s | T2 |
| 1.4 | Why did Kisqali TRx drop in Q1 in the northeast region? | SINGLE_AGENT → causal_impact | causal_effect → causal_impact ✓ | kpi_calculate_tool · 19.0 s | T3 |
| 1.5 | What is the causal impact of rep visits on TRx for Kisqali? | SINGLE_AGENT → causal_impact | causal_effect → causal_impact ✓ | causal_analysis_tool · 7.9 s | T3 |
| 1.6 | *(follow-up)* Which HCP segments show the strongest effect? | SINGLE_AGENT → heterogeneous_optimizer | segment_analysis → heterogeneous_optimizer ✓ | e2i_data_query_tool · 12.7 s | T5 |
| 1.7 | *(follow-up)* Based on that, how should we reallocate rep effort next quarter? | SINGLE_AGENT → resource_optimizer | resource_allocation → resource_optimizer ✓ | e2i_data_query_tool · 14.7 s | T5 |

Notes: KPI lookups (1.1–1.3) route to **explainer** by contract — the #1337
gold's largest class (`kpi_query`, 111/337 rows) is owned by explainer's
catch-all narration, and the legacy `explanation` pattern gained a KPI-lookup
regex during remediation (PR #1366; `intent_classifier.py`). 1.7's
resource_optimizer is **legacy-only** (classifier-unmapped). The follow-up
mechanics of 1.6/1.7 live on AG-UI's full-history resend; `/chat/stream` has
no cross-turn memory unless `conversation_history` is supplied.

**Demo beat** (unchanged): 1.6 returning *different* effect sizes for
high-volume vs community oncologists is the "causal AI, not dashboards" moment.

---

## Narrative 2 — Remibrutinib CSU launch: finding growth

**Persona:** launch/marketing analytics for Remibrutinib (BTK inhibitor, CSU).

| # | Question | Gold routing | Legacy routes to (today) | AG-UI baseline (tools · total) | Tier |
|---|----------|--------------|--------------------------|-------------------------------|------|
| 2.1 | Build a patient cohort for Remibrutinib CSU with inclusion criteria for adults over 18 diagnosed in 2024 | SINGLE_AGENT → cohort_profiler | cohort_definition → cohort_profiler ✓ | clinical_context_tool · 20.1 s | T3 |
| 2.2 | Give me an NRx breakdown by patient clinical segment for Remibrutinib | SINGLE_AGENT → cohort_profiler | segment_analysis → heterogeneous_optimizer ✗ | kpi_calculate_tool · 14.1 s | T2 |
| 2.3 | What are biologic-naive vs biologic-experienced NRx numbers for Remibrutinib? | SINGLE_AGENT → cohort_profiler | explanation → explainer ✗ | kpi_calculate_tool · 8.9 s | T2 |
| 2.4 | What is driving the drop in Remibrutinib NRx in the northeast region this quarter? | SINGLE_AGENT → causal_impact | causal_effect → causal_impact ✓ | causal_analysis_tool · 15.9 s | T3 |
| 2.5 | Where are the biggest untapped opportunities to grow Remibrutinib market share? | **TOOL_COMPOSER** → tool_composer (human-ratified) | performance_gap → gap_analyzer ✗ | kpi_calculate_tool · 19.4 s | T4 |
| 2.6 | How can I optimize resource allocation for Remibrutinib in the northeast region? | SINGLE_AGENT → resource_optimizer | resource_allocation → resource_optimizer ✓ | **orchestrator_tool** · 15.7 s | T3 |
| 2.7 | What is the intent-to-prescribe delta for Remibrutinib? | SINGLE_AGENT → explainer | general → explainer **(LLM)** | kpi_calculate_tool · 9.0 s | T2 |

Notes: v1 sent 2.1 to **cohort_constructor** — retired: chat `cohort_definition`
deliberately routes to **cohort_profiler** (cohort_constructor materializes
patient rows for the ML pipeline and cannot run from a chat payload;
`router.py` INTENT_TO_AGENTS comment). 2.2/2.3 are known legacy misses
(segment-phrasing pulls them off the cohort_profiler contract). 2.6 is one of
the two baseline turns where the AG-UI model elected the orchestrator.

---

## Narrative 3 — Fabhalta PNH funnel: test before you spend

**Persona:** Fabhalta (Factor B inhibitor, PNH) commercial lead. Digital Twin
showcase.

| # | Question | Gold routing | Legacy routes to (today) | AG-UI baseline (tools · total) | Tier |
|---|----------|--------------|--------------------------|-------------------------------|------|
| 3.1 | What percentage of PNH patients have been tested? | SINGLE_AGENT → explainer (human-triaged) | cohort_definition → cohort_profiler ✗ | clinical_context_tool · 11.6 s | T2 |
| 3.2 | What is the current TRx volume for Fabhalta? | SINGLE_AGENT → explainer | explanation → explainer ✓ | kpi_calculate_tool · 5.0 s | T2 |
| 3.3 | Predict which HCP segments are most likely to increase Fabhalta prescriptions next quarter | SINGLE_AGENT → prediction_synthesizer | segment_analysis → heterogeneous_optimizer ✗ | e2i_data_query_tool · 17.3 s | T3 |
| 3.4 | Design an experiment to measure whether speaker programs increase Fabhalta NRx | SINGLE_AGENT → experiment_designer | experiment_design → experiment_designer ✓ | **orchestrator_tool** · 75.2 s | T3–T4 |
| 3.5 | *(follow-up)* What did the digital twin simulation say about expected lift and sample size? | SINGLE_AGENT → experiment_designer | experiment_design + prediction → PARALLEL (experiment_designer, prediction_synthesizer) ✗ | e2i_data_query_tool · 12.7 s | T5 |
| 3.6 | Design an experiment to test whether increasing rep visits improves Fabhalta adoption | SINGLE_AGENT → experiment_designer | experiment_design → experiment_designer ✓ | causal_analysis_tool · 81.4 s | T3 |

Notes: experiment-design turns are the longest single-agent runs on record
(75–90 s; the `experiment_design` dispatch SLA was raised to 150 s on these
measurements — `router.py`). The 3.4 → 3.5 "pre-screen A/B tests with ML
simulation" beat is unchanged.

---

## Narrative 4 — HCP targeting & trigger quality: field operations

**Persona:** field-force effectiveness / omnichannel ops.

| # | Question | Gold routing | Legacy routes to (today) | AG-UI baseline (tools · total) | Tier |
|---|----------|--------------|--------------------------|-------------------------------|------|
| 4.1 | Create a cohort of HCPs who are oncologists in the northeast | SINGLE_AGENT → cohort_profiler | cohort_definition → cohort_profiler ✓ | e2i_data_query_tool · 21.9 s | T3 |
| 4.2 | Build a cohort of high-value HCPs who prescribed more than 50 TRx last quarter | SINGLE_AGENT → cohort_profiler | cohort_definition → cohort_profiler ✓ | (direct answer) · 10.9 s | T3 |
| 4.3 | Segment HCPs by prescription volume into high, medium, and low tiers | SINGLE_AGENT → cohort_profiler | segment_analysis → heterogeneous_optimizer ✗ | (direct answer) · 9.1 s | T3 |
| 4.4 | Which HCP segments show the strongest treatment effect for Remibrutinib? | SINGLE_AGENT → heterogeneous_optimizer | segment_analysis → heterogeneous_optimizer ✓ | causal_analysis_tool · 15.0 s | T3 |
| 4.5 | What is our trigger precision and acceptance rate this month? | SINGLE_AGENT → explainer (human-triaged) | general → explainer **(LLM)** | e2i_data_query_tool · 17.7 s | T2 |
| 4.6 | What is the false alert rate and override rate for triggers? | SINGLE_AGENT → experiment_monitor | general → explainer **(LLM)** ✗ | e2i_data_query_tool · 52.2 s | T2 |
| 4.7 | Did rep actions driven by triggers actually lift prescriptions? | SINGLE_AGENT → causal_impact (human-ratified) | general → explainer **(LLM)** ✗ | causal_analysis_tool · 15.9 s | T3 |

Notes: v1 labeled 4.3 `multi_faceted`/T4 — retired: single-domain tiering is
SINGLE_AGENT by the composition ruling (no second domain, no dependency link).
4.6's gold owner (experiment_monitor) is **legacy-only** and currently reached
only via the LLM fallback, if at all — a known gap, not a demo promise.

---

## Narrative 5 — Trust & governance: "can I believe this model?"

**Persona:** data-science lead / MLOps reviewer.

| # | Question | Gold routing | Legacy routes to (today) | AG-UI baseline (tools · total) | Tier |
|---|----------|--------------|--------------------------|-------------------------------|------|
| 5.1 | What is the current system health score? | SINGLE_AGENT → health_score | system_health → health_score ✓ | e2i_data_query_tool · 5.9 s | T2–T3 |
| 5.2 | What agents are available in the system? | SINGLE_AGENT → health_score | general → explainer **(LLM)** ✗ | agent_routing_tool · 8.1 s | T1 |
| 5.3 | What is the ROC-AUC and calibration of the current Kisqali model? | SINGLE_AGENT → health_score | general → explainer **(LLM)** ✗ | e2i_data_query_tool · 9.9 s | T2 |
| 5.4 | Is there any feature drift in the Kisqali model? | SINGLE_AGENT → drift_monitor | drift_check → drift_monitor ✓ | e2i_data_query_tool · 10.0 s | T3 |
| 5.5 | Why did the model flag this HCP segment — what features drove the prediction? | SINGLE_AGENT → explainer (human-ratified) | segment_analysis → heterogeneous_optimizer ✗ | e2i_data_query_tool · 12.8 s | T3 |
| 5.6 | Explain what heterogeneous treatment effects mean in our analyses | SINGLE_AGENT → explainer | segment_analysis → heterogeneous_optimizer ✗ | document_retrieval_tool · 14.5 s | T1 |
| 5.7 | How confident are we in the rep-visit effect — did it pass refutation tests? | SINGLE_AGENT → causal_impact (human-ratified) | general → explainer **(LLM)** | causal_analysis_tool · 15.2 s | T3 |

Notes: 5.1–5.3 exercise the **health_score vs drift_monitor boundary** —
snapshot asks (health score, current model quality, `model.*perform`) belong to
health_score/system_health; shift-over-baseline asks (5.4, `model.*degrad`,
drift, distribution change) belong to drift_monitor/drift_check. health_score
is **legacy-only** (classifier-unmapped). v1's D1 caveat is retired: the
simple-intent crash on `/chat/stream` (5.1/5.2 died in
`E2I_CHATBOT_SYSTEM_PROMPT.format`) was fixed in #1332.

---

## Narrative 6 — Executive multi-faceted: one question, whole system

**Persona:** franchise head. **These are the TOOL_COMPOSER gate tests** — each
must span ≥ 2 domains with dependency-linked sub-questions (see gate above).

| # | Question | Gold routing | Legacy routes to (today) | AG-UI baseline (tools · total) | Tier |
|---|----------|--------------|--------------------------|-------------------------------|------|
| 6.1 | Compare TRx market share for Kisqali vs its competitors over the last 6 months, explain what's driving the difference, and recommend where to focus reps next quarter | TOOL_COMPOSER (KPI → causal → resource, dependency-linked) | multi_faceted → tool_composer ✓ | kpi_calculate_tool · 22.9 s | T4 |
| 6.2 | Which regions are underperforming on Remibrutinib conversion rate, and for those regions, what would be the ROI of shifting 20% more rep capacity there? | TOOL_COMPOSER (gap → ROI, entity-transformation dependency) | multi_faceted → tool_composer ✓ | kpi_calculate_tool · 19.5 s | T4 |
| 6.3 | If conversion rate in the west is below 15%, which patient segments should we prioritize? | TOOL_COMPOSER (conditional dependency, human-ratified) | segment_analysis → heterogeneous_optimizer ✗ | (direct answer) · 5.7 s | T4 |
| 6.4 | Give me a launch-readiness snapshot for Fabhalta: % PNH tested, NRx trend, top adoption barriers, and one experiment we should run next | TOOL_COMPOSER (4-facet, ends in experiment_designer, human-ratified) | general → explainer **(LLM)** ✗ | clinical_context_tool · 24.0 s | T4 |
| 6.5 | Forecast Kisqali TRx volume for the next two quarters and tell me the biggest risk to that forecast | TOOL_COMPOSER (prediction → risk, human-ratified) | prediction → prediction_synthesizer ✗ | causal_analysis_tool · 16.9 s | T4 |

Notes: composer recall is the incumbent's weakest cell (0.321 over gold;
precision 1.000 — when legacy says TOOL_COMPOSER it is always right, but it
misses most composer-worthy asks: 6.3–6.5 here). The demo beat (streamed
decomposition in the agent-progress renderer) fires reliably on 6.1/6.2/A.10.

---

## Appendix A — Robustness & UX edge cases

| # | Question / action | Gold routing | Legacy routes to (today) | What it tests |
|---|-------------------|--------------|--------------------------|---------------|
| A.1 | Hello, how are you? | CLARIFICATION_NEEDED | general → explainer **(LLM)** | Greeting — AG-UI answers directly, no tools (4.6 s baseline). Routed clarification not deliverable (#1407) |
| A.2a | What can you do? | CLARIFICATION_NEEDED | general → explainer **(LLM)** | Capability grounding (9.4 s) |
| A.2b | What KPIs are available for Kisqali? | SINGLE_AGENT → explainer | general → explainer **(LLM)** ✓ | Capability grounding with entity (9.6 s) |
| A.3 | Waht is the TRx for Kisqali? | SINGLE_AGENT → explainer | general → explainer **(LLM)** ✓ | Typo handler — same answer as 1.1 (5.7 s; the `explanation` KPI regex tolerates the recurring "teh" typo but not "Waht") |
| A.4 | Show me converson rate for Remibrutnib | SINGLE_AGENT → explainer | general → explainer **(LLM)** ✓ | Typos in metric and brand (5.1 s) |
| A.5 | Why did it drop? *(cold first message)* | CLARIFICATION_NEEDED | causal_effect → causal_impact ✗ | The canonical #1407 case: gold says clarify; legacy structurally cannot; AG-UI asks conversationally |
| A.6 | What is TRx? | SINGLE_AGENT → explainer | explanation → explainer ✓ | Definitional — should define, not dump data (9.2 s) |
| A.7 | Ask 1.1, then immediately ask 1.4 while streaming | — (UI-only, not a gold row) | — | Concurrent/interrupt behavior; baseline: both turns complete, episodic recap on repeat |
| A.8 | Paraphrase 1.1 later in the same session | — (measured separately) | — | **Paraphrase repeat: measured PASS 5/5** — recap acknowledges/reuses the earlier grounded number, often faster than cold because the recap skips the tool round-trip (`results/2026-07-31_t5_paraphrase_repeat/RESULTS.md`; #1339 closed measured-not-needed; verbatim caching out of scope) |
| A.9 | Ask 1.4, wait, then ask: Why is this happening? | SINGLE_AGENT → causal_impact | general → explainer **(LLM)** | T5 memory: referent resolution via AG-UI history (14.0 s) |
| A.10 | A 60+ word compound question mixing 4 domains | TOOL_COMPOSER (human-ratified) | multi_faceted → tool_composer ✓ | Complexity handling — baseline completed a real composite in 45.8 s |

Retired v1 caveats (all fixed and verified post-merge):

- **#1332 (D1)**: greeting/help/agent_status no longer crash `/chat/stream`.
- **#1333 (D2)**: the haiku intent-fallback JSON now parses (fence-tolerant) —
  **(LLM)** rows above get a real LLM classification, not a parse-fail default.
- **#1334 (D3)**: chat RAG retrieval returns results (embedding-space fix).
- **#1335 (D4)**: `/chat/stream` turns persist (session bookkeeping fixed).
- **#1336 (D5)**: complete orchestrator failures now bridge to the AG-UI brain
  (PR #1394) instead of returning a bare fail-closed summary.
- **#1340**: silent-dead-turn and stale "Working…" header fixed in the UI.

## Appendix B — Measurement sheet (fields are real since PR #1330)

Per question log: `question_id`, `gold_pattern`, `gold_agents`,
`routing_pattern`, `agents_dispatched`, `ttfb_ms`, `first_progress_ms`,
`total_ms`, `classification_latency_ms`, `used_llm_layer`, `answer_correct`,
`suggestion_pills_relevant`, `notes`. On `/chat/stream` these arrive in the
`dispatch_info` SSE frame and (for the pipeline) in `classification_logs`; the
populated master for the recorded baseline is
`results/2026-07-29_copilot_chat_perf/measurements.csv`.

Pass criteria by tier — **values carried over from v1 UNCHANGED; budgets
pending re-baseline — #1338**:
T1 < 3s total; T2 < 8s total; T3 first progress < 5s, total < 40s;
T4 first decomposition visible < 8s, total < 90s; T5 paraphrase repeat
acknowledges/reuses the earlier analysis with consistent grounded numbers at
acceptable latency (not materially slower than cold — and typically faster,
since a recap can skip re-running the tool; re-running to re-validate is not a
failure — #1339); T6 never hallucinates on ambiguity.
