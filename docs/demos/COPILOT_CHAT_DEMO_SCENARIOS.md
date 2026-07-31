# Copilot Chat — Performance Test Questions & Demo Narratives

> **⚠ SUPERSEDED (2026-07-31, #1345)** by
> [`COPILOT_CHAT_DEMO_SCENARIOS_V2.md`](COPILOT_CHAT_DEMO_SCENARIOS_V2.md).
> This file predates the recorded baseline
> (`results/2026-07-29_copilot_chat_perf/`), the two-chat-surfaces finding, and
> the contract-verified routing gold set — its expected-intent/path annotations
> are hand-distilled and partly wrong (e.g. 2.1 cohort_constructor, 4.3
> multi_faceted, the T4 tier description). It is retained unchanged as the
> point-in-time record the 2026-07-29 measurements were scripted against.
> Latency-budget re-baselining is owned by #1338 — **done 2026-07-31 in v2**
> (p90 basis; T1 merged into T2, 3.4 → T4): this file's budgets are historical.

A cohesive question set for exercising the copilot chat UI end-to-end, grounded in
the system's actual routing surfaces (chatbot intents, orchestrator 4-stage
classifier, tool composer) and the 22-agent tier architecture. Each narrative is a
self-contained business story that escalates in complexity, so the same script
doubles as:

1. a **performance test plan** — every question is annotated with the intent it
   should classify to, the routing path it exercises, and the latency tier to
   expect, and
2. a **demo narrative** — each scenario follows a real commercial-analytics
   workflow (observe → diagnose → predict → act) for one brand/persona.

Sources used to build this set: `tests/fixtures/dspy_lane_golden_queries.json`
(real production chat traffic, hand-labeled), `config/kpi_definitions.yaml`
(44 KPIs), `config/domain_vocabulary.yaml` (brands, regions, specialties,
segments), `config/cohort_vocabulary.yaml` (cohort query grammar), the chatbot
intent taxonomy in `src/api/routes/chatbot_graph.py`, and the orchestrator
classifier schemas in `src/agents/orchestrator/classifier/schemas.py`.

---

## Latency / complexity tiers

Use these tiers when recording timings. What matters per tier is different: a T1
answer that takes 8s is a failure; a T4 answer at 45s with good streaming
progress may be fine.

| Tier | Path exercised | Examples of intents | What "good" looks like |
|------|---------------|---------------------|------------------------|
| **T1 — Conversational** | classify → generate, no agents, no RAG depth | `greeting`, `help`, `general` | Near-instant first token; correct capability description |
| **T2 — KPI fast path** | classify → KPI engine lookup (brand/segment/LoT/window routing) | `kpi_query` | Seconds, not tens of seconds; a real number with period + comparison, chart where applicable |
| **T3 — Single agent** | classify → orchestrator → SINGLE_AGENT dispatch | `causal_analysis`, `recommendation`, `search`, `agent_status` | Streaming progress events visible; structured result (effect estimate + refutations, cohort definition, ranked opportunities) |
| **T4 — Multi-agent / composer** | classify → orchestrator → PARALLEL_DELEGATION or TOOL_COMPOSER decomposition | `multi_faceted` | Visible decomposition into sub-questions; each facet resolved; coherent synthesis |
| **T5 — Context / memory** | follow-up turns resolving "that/those/why is this happening", and paraphrase repeats of an earlier question, against session memory | any (follow-up) | Correct referent resolution; no re-asking. On a paraphrase repeat, the answer acknowledges/reuses the earlier analysis (episodic recap) with consistent grounded numbers — typically faster than cold when the recap skips re-running tools. Verbatim answer caching is out of scope (#1339). |
| **T6 — Robustness** | typo handler, ambiguity → clarification | any | Typos silently corrected; ambiguous queries produce a clarifying question, not a hallucinated answer |

Record per question: time-to-first-token, time-to-first-progress-event (T3+),
total completion time, intent classified (vs expected), agent(s) dispatched, and
whether suggestion pills refresh sensibly after the answer.

---

## Narrative 1 — Kisqali brand pulse: "the Monday morning question"

**Persona:** Kisqali brand lead. **Arc:** what's happening → why → who's
affected → what do we do.

| # | Question | Expected intent | Path / agents | Tier |
|---|----------|-----------------|---------------|------|
| 1.1 | What is TRx for Kisqali? | `kpi_query` | KPI engine (highest-volume production query, n=137) | T2 |
| 1.2 | What are the TRx trends for Kisqali in Q4? | `kpi_query` | KPI engine, trend charting | T2 |
| 1.3 | What is the market share for Kisqali compared to competitors? | `kpi_query` | TRx Share windows | T2 |
| 1.4 | Why did Kisqali TRx drop in Q1 in the northeast region? | `causal_analysis` | causal_impact (effect estimation + 5 DoWhy refutation tests) | T3 |
| 1.5 | What is the causal impact of rep visits on TRx for Kisqali? | `causal_analysis` | causal_impact (ATE with refutation gating) | T3 |
| 1.6 | *(follow-up)* Which HCP segments show the strongest effect? | `causal_analysis` | heterogeneous_optimizer (CATE by segment); tests reference-chain follow-up | T5 |
| 1.7 | *(follow-up)* Based on that, how should we reallocate rep effort next quarter? | `recommendation` | resource_optimizer | T5 |

**Demo beat:** the story lands when 1.6 returns *different* effect sizes for
high-volume vs community oncologists — that's the "causal AI, not dashboards"
moment.

---

## Narrative 2 — Remibrutinib CSU launch: finding growth

**Persona:** launch/marketing analytics for Remibrutinib (BTK inhibitor,
chronic spontaneous urticaria). **Arc:** define the population → measure
uptake → diagnose softness → find the ROI.

| # | Question | Expected intent | Path / agents | Tier |
|---|----------|-----------------|---------------|------|
| 2.1 | Build a patient cohort for Remibrutinib CSU with inclusion criteria for adults over 18 diagnosed in 2024 | `search` / `multi_faceted` | cohort_constructor (COHORT_DEFINITION domain) | T3 |
| 2.2 | Give me an NRx breakdown by patient clinical segment for Remibrutinib | `kpi_query` | KPI engine, clinical segment breakdown | T2 |
| 2.3 | What are biologic-naive vs biologic-experienced NRx numbers for Remibrutinib? | `kpi_query` | KPI engine, segment routing | T2 |
| 2.4 | What is driving the drop in Remibrutinib NRx in the northeast region this quarter? | `causal_analysis` | causal_impact | T3 |
| 2.5 | Where are the biggest untapped opportunities to grow Remibrutinib market share? | `recommendation` | gap_analyzer (ROI opportunity ranking) | T3 |
| 2.6 | How can I optimize resource allocation for Remibrutinib in the northeast region? | `recommendation` | resource_optimizer | T3 |
| 2.7 | What is the intent-to-prescribe delta for Remibrutinib? | `kpi_query` | brand-specific KPI (`Remi - Intent-to-Prescribe Δ`) | T2 |

---

## Narrative 3 — Fabhalta PNH funnel: test before you spend

**Persona:** Fabhalta (Factor B inhibitor, PNH) commercial lead considering a
speaker-program investment. **Arc:** funnel health → forecast → design the
experiment → pre-screen it in silico. This is the **Digital Twin showcase**.

| # | Question | Expected intent | Path / agents | Tier |
|---|----------|-----------------|---------------|------|
| 3.1 | What percentage of PNH patients have been tested? | `kpi_query` | brand-specific KPI (`Fabhalta - % PNH Tested`) | T2 |
| 3.2 | What is the current TRx volume for Fabhalta? | `kpi_query` | KPI engine | T2 |
| 3.3 | Predict which HCP segments are most likely to increase Fabhalta prescriptions next quarter | `prediction` | prediction_synthesizer | T3 |
| 3.4 | Design an experiment to measure whether speaker programs increase Fabhalta NRx | `recommendation` | experiment_designer with Digital Twin pre-screening | T3–T4 |
| 3.5 | *(follow-up)* What did the digital twin simulation say about expected lift and sample size? | `recommendation` | digital_twin results via experiment_designer | T5 |
| 3.6 | Design an experiment to test whether increasing rep visits improves Fabhalta adoption | `recommendation` | experiment_designer (variation of 3.4 — checks consistency) | T3 |

**Demo beat:** 3.4 → 3.5 is the "we pre-screen A/B tests with ML simulation
before spending field budget" story.

---

## Narrative 4 — HCP targeting & trigger quality: field operations

**Persona:** field-force effectiveness / omnichannel ops. **Arc:** who to
target → how they segment → do the triggers we send actually work.

| # | Question | Expected intent | Path / agents | Tier |
|---|----------|-----------------|---------------|------|
| 4.1 | Create a cohort of HCPs who are oncologists in the northeast | `search` | cohort_constructor | T3 |
| 4.2 | Build a cohort of high-value HCPs who prescribed more than 50 TRx last quarter | `search` | cohort_constructor with quantitative criterion | T3 |
| 4.3 | Segment HCPs by prescription volume into high, medium, and low tiers | `multi_faceted` | tool composer / segmentation | T4 |
| 4.4 | Which HCP segments show the strongest treatment effect for Remibrutinib? | `causal_analysis` | heterogeneous_optimizer (CATE) | T3 |
| 4.5 | What is our trigger precision and acceptance rate this month? | `kpi_query` | trigger KPIs (Trigger Precision, Acceptance Rate) | T2 |
| 4.6 | What is the false alert rate and override rate for triggers? | `kpi_query` | trigger KPIs | T2 |
| 4.7 | Did rep actions driven by triggers actually lift prescriptions? | `causal_analysis` / `kpi_query` | Action Rate Uplift KPI → causal framing | T3 |

---

## Narrative 5 — Trust & governance: "can I believe this model?"

**Persona:** data-science lead / MLOps reviewer, or a skeptical medical
reviewer in the room. **Arc:** system health → model quality → drift →
explainability. Differentiates the platform on rigor.

| # | Question | Expected intent | Path / agents | Tier |
|---|----------|-----------------|---------------|------|
| 5.1 | What is the current system health score? | `agent_status` / `kpi_query` | health_score | T2–T3 |
| 5.2 | What agents are available in the system? | `agent_status` / `help` | agent registry | T1 |
| 5.3 | What is the ROC-AUC and calibration of the current Kisqali model? | `kpi_query` | ML KPIs (ROC-AUC, Calibration Slope Deviation, Brier) | T2 |
| 5.4 | Is there any feature drift in the Kisqali model? | `kpi_query` / monitoring | drift_monitor (Feature Drift PSI) | T3 |
| 5.5 | Why did the model flag this HCP segment — what features drove the prediction? | `causal_analysis` / explanation | explainer (SHAP, 50–500ms interpretability API) | T3 |
| 5.6 | Explain what heterogeneous treatment effects mean in our analyses | `general` / `help` | explanation, no data fetch | T1 |
| 5.7 | How confident are we in the rep-visit effect — did it pass refutation tests? | `causal_analysis` | causal_impact refutation report (fail-closed gating, E-value) | T3 |

---

## Narrative 6 — Executive multi-faceted: one question, whole system

**Persona:** franchise head who asks compound questions. These are the **tool
composer stress tests** — the classifier must decompose, order dependencies,
and synthesize.

| # | Question | Expected routing | Tier |
|---|----------|------------------|------|
| 6.1 | Compare TRx market share for Kisqali vs its competitors over the last 6 months, explain what's driving the difference, and recommend where to focus reps next quarter | TOOL_COMPOSER — 3 sub-questions with LOGICAL_SEQUENCE dependencies (KPI → causal → resource) | T4 |
| 6.2 | Which regions are underperforming on Remibrutinib conversion rate, and for those regions, what would be the ROI of shifting 20% more rep capacity there? | TOOL_COMPOSER — ENTITY_TRANSFORMATION dependency (filtered region set feeds gap/ROI) | T4 |
| 6.3 | If conversion rate in the west is below 15%, which patient segments should we prioritize? | CONDITIONAL dependency detection | T4 |
| 6.4 | Give me a launch-readiness snapshot for Fabhalta: % PNH tested, NRx trend, top adoption barriers, and one experiment we should run next | TOOL_COMPOSER — 4-facet decomposition ending in experiment_designer | T4 |
| 6.5 | Forecast Kisqali TRx volume for the next two quarters and tell me the biggest risk to that forecast | prediction_synthesizer + causal/drift context | T4 |

**Demo beat:** show the streamed decomposition — the UI's agent-progress
renderer displaying sub-questions being dispatched in parallel is the visual
payoff.

---

## Appendix A — Robustness & UX edge cases (performance hygiene)

Not demo material, but required for a performance pass:

| # | Question / action | What it tests |
|---|-------------------|---------------|
| A.1 | Hello, how are you? | `greeting` — should NOT dispatch agents; instant |
| A.2 | What can you do? / What KPIs are available for Kisqali? | `help` — capability grounding, instant |
| A.3 | Waht is the TRx for Kisqali? | typo handler (`src/nlp/typo_handler.py`) — same answer as 1.1, no added latency cliff |
| A.4 | Show me converson rate for Remibrutnib | typo in both metric and brand |
| A.5 | Why did it drop? *(as a cold first message)* | CLARIFICATION_NEEDED — should ask "which metric/brand", not hallucinate |
| A.6 | What is TRx? | definitional edge (production-ambiguous: `kpi_query`/`help`/`general`) — should define, not dump data |
| A.7 | Ask 1.1, then immediately ask 1.4 while streaming | concurrent/interrupt behavior of the chat UI |
| A.8 | Paraphrase 1.1 later in the same session (e.g. "Remind me — what was Kisqali's total prescription count again?") | episodic recap: acknowledges/reuses the earlier answer with a consistent number at acceptable latency. (Re-running the tool to re-validate is fine — the pass is a correct, consistent, acknowledged answer, not zero tool calls; it's typically faster than cold precisely because a recap *can* skip the tool.) Verbatim caching out of scope (#1339); measured 2026-07-31 — see `docs/demos/results/2026-07-31_t5_paraphrase_repeat/` |
| A.9 | Ask 1.4, wait, then ask: Why is this happening? | T5 memory: episodic context retrieval |
| A.10 | A 60+ word compound question mixing 4 domains | complexity warning / graceful decomposition, no timeout |

## Appendix B — Suggested measurement sheet

For each question log: `question_id`, `intent_expected`, `intent_actual`,
`routing_pattern`, `agents_dispatched`, `ttfb_ms` (first token),
`first_progress_ms`, `total_ms`, `classification_latency_ms` (from
`ClassificationResult`), `used_llm_layer`, `answer_correct` (Y/N),
`suggestion_pills_relevant` (Y/N), `notes`.

Pass criteria by tier (starting points — tighten from observed baselines):
T1 < 3s total; T2 < 8s total; T3 first progress < 5s, total < 40s;
T4 first decomposition visible < 8s, total < 90s; T5 paraphrase repeat
acknowledges/reuses the earlier analysis with consistent grounded numbers at
acceptable latency (not materially slower than cold — and typically faster,
since a recap can skip re-running the tool; re-running to re-validate is not a
failure — #1339); T6 never hallucinates on ambiguity.
