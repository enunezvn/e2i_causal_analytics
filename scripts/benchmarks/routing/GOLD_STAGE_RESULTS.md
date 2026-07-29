# Gold-judge stage results (#1337 Step 0)

Generated 2026-07-29 by `gold_judge.py` (axis 1) and `build_answer_quality.py` (axis 2),
following `GOLD_STAGE_PROTOCOL.md`. The contract registry `data/agent_contracts.json`
is the single source of truth for routing ownership; the 22 human-ratified verdicts in
`review/proposed_verdicts.json` are the calibration anchors.

Two independent axes:
- **Axis 1 — routing gold**: which route SHOULD own each of the 337 benchmark queries
  (contract ownership, not classifier reach).
- **Axis 2 — answer quality**: how good the 22 live AG-UI answers actually are, graded
  through three ordered layers. Axis-2 grades never modify axis-1 labels.

---

## Axis 1 — routing gold over 337 queries

### How each row was labeled (precedence cascade)

| gold_source | rows | how |
|---|---|---|
| `human-ratified` | 19 | normalized text matches one of the 22 ratified verdicts → the human `final` carried directly, no LLM call |
| `authored-ground-truth` | 76 | `source=authored` rows carry `authored_gold_pattern`/`authored_gold_agents` (the benchmark author's designed probe-cell labels) → used directly; the judge also labeled them as a cross-check |
| `llm-judge` | 242 | historical / perturbation / non-anchor demo → labeled by `claude-sonnet-5` against the contract digest + the 10 human few-shot anchors + the normative composition rules |

Output: `data/benchmark_queries_gold.jsonl` (every input row + `gold_pattern`,
`gold_agents`, `gold_confidence`, `gold_rationale`, `gold_source`, and — where the
judge ran — `judge_pattern`/`judge_agents`/`judge_confidence` as an independent cross-check).

### Label distribution

**Overall (337):**

| pattern | count |
|---|---|
| SINGLE_AGENT | 269 |
| CLARIFICATION_NEEDED | 36 |
| TOOL_COMPOSER | 29 |
| PARALLEL_DELEGATION | 3 |

**By source:**

| source | n | SINGLE_AGENT | TOOL_COMPOSER | PARALLEL | CLARIFICATION |
|---|---|---|---|---|---|
| demo | 48 | 38 | 7 | 0 | 3 |
| historical | 115 | 107 | 0 | 1 | 7 |
| perturbation | 98 | 70 | 14 | 2 | 12 |
| authored | 76 | 54 | 8 | 0 | 14 |

**SINGLE_AGENT owner distribution** (gold_agents, 269 rows): explainer 102 (the catch-all
KPI/narration owner — historical traffic is KPI-heavy), cohort_profiler 31, causal_impact 28,
health_score 24, resource_optimizer 17, drift_monitor 15, heterogeneous_optimizer 14,
experiment_monitor 11, feedback_learner 10, experiment_designer 9, prediction_synthesizer 4,
gap_analyzer 4.

### Anchor matches & low-confidence

- **19** benchmark rows matched a human-ratified verdict (all in the demo source; 3 of the
  22 ratified queries came from real classification-log traffic that isn't in the pool).
- **76** authored ground-truth cells.
- **52** LLM-judged rows fell below the 0.6 confidence floor (34 historical, 14 perturbation,
  4 demo) → all listed in `review/gold_low_confidence.md` for human follow-up. These are
  dominated by terse real-traffic fragments ("yes", "run both", "causal analysis", "what data
  do you have?") where routing genuinely needs the prior turn.

### Judge calibration signal (strong)

The gold judge independently labeled all 76 authored cells (it was not shown the authored
gold) and agreed **76/76 on pattern**, including the hard cases: all **8 authored
TOOL_COMPOSER** and all **14 authored CLARIFICATION_NEEDED** cells. That the same
`claude-sonnet-5` + contract-digest + few-shot prompt reproduces the human-authored gold
perfectly on the probe cells is the best evidence we have that the 242 llm-judged labels
are trustworthy (modulo the 52 flagged low-confidence rows).

### The 5 most interesting disagreements with the nightly judge's TOOL_COMPOSER corrections

The #1341/#1342 nightly judge had corrected 22 distinct disputed queries to TOOL_COMPOSER.
Gold (human-ratified) disagrees on **13** of them — every one a case of the nightly judge's
step-counting rule ("needs multiple dependent steps → TOOL_COMPOSER"), which the contract
registry's composition gate rejects. The five that best illustrate the distinct failure modes:

1. **q05** "What is the causal impact of rep visits on TRx for Kisqali?" — nightly:
   *"dependent, multi-step tool calls across sales and clinical domains."* Gold:
   **SINGLE_AGENT / causal_impact**. Failure mode: **data sources mistaken for capability
   domains** — sales-force activity and prescription outcomes are two inputs to one
   causal_impact ATE dispatch, not two agent domains. (This is causal_impact's verbatim
   first covers entry.)

2. **q16** "Which HCP segments show the strongest treatment effect for Remibrutinib?" —
   nightly: *"first identify treatment effects, then segment HCPs by those effects."* Gold:
   **SINGLE_AGENT / heterogeneous_optimizer**. Failure mode: **an agent's own internal
   pipeline mistaken for composition** — estimate_cate → analyze_segments is
   heterogeneous_optimizer's single dispatch, not a hand-off.

3. **q11** "Build a patient cohort for Remibrutinib CSU with inclusion criteria..." —
   nightly: *"cohort_profiler alone cannot independently validate drug-indication pairing or
   execute cross-domain filtering."* Gold: **SINGLE_AGENT / cohort_profiler**. Failure mode:
   **an invented capability gap** — "drug-indication validation" is not a separate registry
   domain; brand/indication/date criteria are filters inside one cohort dispatch.

4. **q03** "Design an experiment to measure whether speaker programs increase Fabhalta NRx" —
   nightly: *"dependent steps across clinical/commercial domains."* Gold: **SINGLE_AGENT /
   experiment_designer**. Failure mode: **the owning agent's internal phases counted as
   domains** — metrics/controls/power/validity are experiment_designer's own pipeline stages.

5. **q13** "How can I optimize resource allocation for Remibrutinib in the northeast region?" —
   nightly: *"dependent steps span clinical and operational domains."* Gold: **SINGLE_AGENT /
   resource_optimizer**. Failure mode: **multi-lever single-domain allocation mistaken for
   multi-domain**, compounded by resource_optimizer being classifier-unmapped (a coverage gap
   the benchmark exposes, never a reason to route to the composer).

The other 8 disagreements (q06, q07, q12, q14, q15, q17, q18, and the extend-flagged q15)
are the same pattern applied to experiment design, refutation reporting, causal attribution,
prediction, and explanation.

---

## Axis 2 — answer quality on the 22 live records

Graded from `review/empirical_results/raw_empirical22.jsonl` (live AG-UI answers), three
ordered layers, each presuming the previous. Numeric claims were spot-checked READ-ONLY
against the live Supabase DB (`causal_paths`, `business_metrics`) on 2026-07-29. Output:
`review/answer_quality_22.json`.

### Per-layer results

| layer | PASS | PARTIAL | N/A |
|---|---|---|---|
| L1 — hallucinated? | 22 | 0 | 0 |
| L2 — faithfully retrieved? | 19 | 1 | 2 |
| L3 — accurate & business-appropriate? | 21 | 1 | 0 |

**Layer 1 (zero hallucinations, 22/22).** No answer fabricated data. Six answers (q11, q12,
q14, q15, q18, q19) explicitly *decline* to invent — refusing to build a patient/HCP cohort
the tools can't produce, refusing to assert a "drop" with no baseline, refusing a
non-existent per-HCP prediction, requesting disambiguation. Refusal quality is the standout
strength of this surface.

**Layer 2 (faithful retrieval — verified against the DB).** Roughly 50 individual numeric
claims across `causal_paths` and `business_metrics` were checked and matched **byte-for-byte**,
including:
- The protocol's q05 worked example (row `scp_c696e9a45437`: 0.166 / 0.796 / 77d /
  backdoor.linear_regression / $38,244) and the full Kisqali driver table (6/6).
- q17's cross-brand rep-detailing effects (Remibrutinib 0.298/$91,289, Fabhalta 0.134/$42,707,
  Kisqali 0.166/$38,244 — 3/3).
- q13/q22's Remibrutinib-northeast `business_metrics` block (TRx 63,670/82,575 = 77.1%, NRx
  109.5% / +40.4% / ROI 3.3, share 0.25/0.28 = 89.3%, conversion 0.49, engagement 10.0/14.65
  = 68.3% — 5/5), and Kisqali regional market-share (NE 0.49/0.57, S 0.43/0.48, W 0.47/0.50).
- q04, q12, q16, q03, q06 causal paths — all exact.

The 2 N/A are the honest cohort refusals (q11, q15) — no data claims to verify. The 1 PARTIAL
is q19 (below).

**Layer 3 (accurate & business-appropriate — 21/22).** The single PARTIAL is q16: the
numbers it quotes are faithful, but it frames the 0.437/0.493-effect `treatment_arm` paths as
the *"strongest treatment effect"* when higher-effect rows (0.548, 0.542) exist for the same
node — a selection/labeling imprecision, disclosed alongside the honest note that the real
heterogeneous_optimizer agent returned no Remibrutinib records.

### Worst offenders

- **q16 — L3 PARTIAL**: "strongest effect" selection imprecision (numbers themselves exact).
- **q19 — L2 PARTIAL**: the answer's *62.8%* West conversion (a trigger-data tool computation)
  does **not** reconcile with `business_metrics` conversion_rate (West averages **48.3%**) —
  the one numeric claim across all 22 that I could not pin to a source row. It does not change
  the answer's conclusion: the conditional premise ("conversion < 15%") is false in data by any
  measure, so the correct early-exit (no segment-prioritization warranted) stands.

### Cross-cutting findings

- **Validation-status probe resolved (q07).** The schema confirms `causal_paths` has **no**
  `refutation_passed` column and `validation_status='validated'` on all 2,729 rows — so q07's
  live claim ("no explicit refutation-test flag in the registry; entries carry
  validation_status") is **accurate**. The protocol's open q07 probe closes in the answer's favor.
- **Synthetic provenance** (`is_synthetic=true` universally) is disclosed by the causal answers
  but less consistently by the KPI answers — a transparency gap, not an accuracy defect.
- **`business_impact_estimate` completeness** is inconsistent (surfaced in q05/q13/q22, omitted
  elsewhere) — the completeness pattern the protocol first noticed on q05, confirmed as
  answer-dependent rather than systemic.
- **Route vs surface**: the 6 TOOL_COMPOSER-class asks (q02, q08, q09, q19, q20, q21) were all
  answered on the AG-UI brain because the orchestrator `tool_composer` route crashes on
  dispatch (#1350). Answer quality is high *despite* the broken composer route — the AG-UI
  surface degrades gracefully.

---

## Cost & reproducibility

- **Axis-1 judge**: `claude-sonnet-5`, 318 calls, 3,003,872 input + 71,361 output tokens,
  **≈ $10.08** (one-time, quality-over-cost; sonnet-5 pricing $3/$15 per 1M). Idempotent/
  resumable via `review/empirical_results/gold_progress.jsonl` (a crash never re-bills a done row).
- **Axis-2**: no LLM; grades are human assessments backed by READ-ONLY DB SELECTs.
- Re-run axis 1: `.venv/bin/python scripts/benchmarks/routing/gold_judge.py`
  (add `--assemble-only` to rebuild outputs from existing progress).
- Re-run axis 2: `.venv/bin/python scripts/benchmarks/routing/build_answer_quality.py`.

---

## Addendum — low-confidence triage (2026-07-29, human)

The 52 sub-0.6-confidence llm-judge rows were human-triaged the same day: **39 confirmed,
13 overturned** (details + evidence in `review/gold_low_confidence.md`, "Triage resolutions").
Deciding evidence the judge never had: live `chatbot_messages` transcripts (which resolve the
"yes" / "run both" / "causal analysis" fragments to concrete accepted offers), demo_meta
intent documentation, and the prod `triggers` table proving trigger precision/acceptance are
real platform KPIs (coverage gap, not ambiguity).

Post-triage distribution: **SINGLE_AGENT 276 · CLARIFICATION_NEEDED 28 · TOOL_COMPOSER 28 ·
PARALLEL_DELEGATION 5**. Zero unresolved low-confidence rows remain; `gold_source` now
distinguishes `llm-judge` (190) / `llm-judge+human-confirmed` (39) / `human-triage` (13) /
`human-ratified` (19) / `authored-ground-truth` (76).

New finding: the five trigger-KPI rows expose an unowned real capability (trigger
effectiveness metrics) — candidate contract extension in the spirit of #1356.
