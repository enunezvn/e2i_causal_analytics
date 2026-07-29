# Judge corrections to TOOL_COMPOSER — human review worksheet

35 labeled rows, 22 distinct queries.

**The verdict question is contract coverage, not step-counting.** Per the platform ruling below, a query is TOOL_COMPOSER only if it spans >= 2 distinct agent contracts AND the cross-agent parts depend on each other. A single agent's internal multi-step pipeline is one SINGLE_AGENT dispatch. Fill **Verdict** per query with one of:

- `agree` — the ask genuinely spans multiple agent contracts with dependencies; TOOL_COMPOSER stands.
- `single_agent:<name>` — that agent's contract covers the ask end-to-end; judge label is wrong.
- `extend:<name> — <what to add>` — product intent is single-agent but the named agent's contract needs extending (design input, not just a label).
- `clarify` — the query is genuinely ambiguous; CLARIFICATION_NEEDED was right after all.

Verdicts feed the #1341 judge-prompt fix and the #1337 Step 0 gold protocol; `extend:` verdicts become contract-change work items.

**Proposed verdicts**: each query below carries a contract-based proposed verdict with reasoning (from review/proposed_verdicts.json). These are HYPOTHESES derived from the verified contracts only — not validated against actual agent responses. The planned empirical pass (execute each disputed query, capture the real response per candidate route) is the cheapest disproof; overriding a proposal needs no justification, and confirming one is best done with response evidence in hand. To accept a proposal, write `accept` in the Verdict slot; otherwise write your own.

## The platform's own composition ruling (verified from code, 2026-07-29)

Per the platform's own code, the criterion is 'no single agent's capability/domain covers the ask' (the query must span >=2 DISTINCT agent-capability domains, or >=2 distinct strong intents), AND those cross-domain sub-questions must be dependency-linked. 'Query decomposes into multiple dependent steps' is NOT sufficient on its own. If all steps live in ONE domain, the query routes SINGLE_AGENT no matter how many dependent internal steps it has - the domain agent (e.g. causal_impact) runs that multi-step pipeline itself under one dispatch. The dependency signal's ONLY job is to split a multi-DOMAIN query into PARALLEL_DELEGATION (no deps) vs TOOL_COMPOSER (deps). The LLM judge's rule ('needs multiple dependent steps -> TOOL_COMPOSER') is therefore WRONG by the platform's definition; the correct gate is cross-domain/cross-agent spanning first, dependency second.

Key evidence:

- pattern_selector.py:127-147 (Rule 3): `if not domain_mapping.is_multi_domain:` returns SINGLE_AGENT BEFORE dependency analysis is ever consulted for composition - single-domain can never reach TOOL_COMPOSER regardless of sub-questions/dependencies.
- pattern_selector.py:194-212 (Rule 6): TOOL_COMPOSER is reachable only after Rule 3 (multi-domain established) AND Rule 5 fails (is_parallelizable False, i.e. dependencies exist).
- schemas.py:129: is_multi_domain = len(domains_detected) > 1, where domains are distinct AGENT-CAPABILITY domains - the spanning test is over capabilities, not steps.
- intent_classifier.py:429-450: promotion to 'multi_faceted' (->tool_composer) requires `len(strong_components) >= 2` DISTINCT strong intents PLUS has_sequential_composition PLUS not a defined parallel pair.

Ambiguity resolutions (settled 2026-07-29 with code/git/measured evidence — review under these rules; measurement in empirical_results/classifier_measurement_22.json):

- RESOLVED — DOC vs CODE gap (the judge's trap): the docs are fixed. PR #1347 (merged 4d0305e8, 2026-07-29) added a 'When Does a Query Route Here? (Routing Gate)' section to tier1-tool-composer-contracts.md (lines 23-70) stating the multi-DOMAIN gate, the Rule 3/5/6 order, and the parallel-pair override. Code and doc now agree; the code was and remains normative. The trap that misled the LLM judge is closed for all future labeling.
- RESOLVED — two live classifiers with non-identical reach (norming rule): gold labels are normed against the contract registry (SSOT), NOT against what either classifier can express. Measured proof neither classifier can be the norm (empirical_results/classifier_measurement_22.json): on q13/q22 legacy scores resource_allocation 0.87/1.00 while the 4-stage maps q13 to explainer ('how' -> EXPLANATION 0.51) and q22 to CLARIFICATION, because DOMAIN_TO_AGENT has no resource domain. Classifier coverage gaps are findings the benchmark exposes (#1337 'unmapped agents'), never constraints on gold.
- RESOLVED — defined parallel pairs override the dependency signal (intentional design, empirically inert on this set): inline intent at intent_classifier.py:160-172 ('deliberately routes as PARALLEL'; the marker is often an incidental preamble; >=3 strong intents still promote) plus mirror test test_multipart_tool_composer_routing.py::test_parallel_pairs_mirror_router establish it as a product decision, not a bug. A/B measurement (stock vs PARALLEL_INTENT_PAIRS emptied) over all 22 disputed queries: ZERO intents change, and has_sequential_composition fires on none of the 22 — the override binds no row, so no PARALLEL verdict option is needed for this review. Nearest candidate q20 (forecast + risk) doesn't trip it: both classifiers route it as plain prediction; whether prediction_synthesizer's uncertainty/risk coverage makes q20 single_agent is that row's normative question, unaffected by the pair rule. If a future query IS a 2-strong-intent defined pair with a sequential marker, gold follows the pair (PARALLEL), per design intent.
- RESOLVED — advisory decomposition (out of scope for labels): the routing label names the correct OWNER of the dispatch; tool_composer re-decomposes from scratch (router.py:451-458), so the classifier's routing-time sub-questions/DAG never bind execution. This distinction affects execution-quality evaluation only — currently moot regardless, since tool_composer crashes on orchestrator dispatch (#1350). No impact on any verdict.
- RESOLVED — domain granularity (bright-line test): 'spans >=2 domains' is decided by the contract registry's covers/does_not_cover ownership. If every capability the ask needs sits in ONE agent's covers list, it is single-domain no matter how many internal steps; it spans only when a needed capability appears in that agent's does_not_cover with a handoff to another agent. domain_mapper keyword scores are the runtime mechanism, NOT the norm — measured (classifier_measurement_22.json): 'how' alone puts EXPLANATION(0.51) on top of q13 (a resource-allocation ask) and drags q16 (verbatim in heterogeneous_optimizer's covers) into a 4-domain PARALLEL_DELEGATION; 9/22 disputed queries collapse to CLARIFICATION_NEEDED. Keyword-tip evidence is a classifier finding, never verdict grounds.

Boundary notes:

- health_score vs drift_monitor: the prior 'health_score = NOT ML-model quality' premise is WRONG. health_score HAS a model-health dimension (model_health.py + ModelMetrics accuracy/precision/recall/f1/auc) - it scores whether deployed models are performing acceptably RIGHT NOW against fixed thresholds (point-in-time operational snapshot). drift_monitor owns TEMPORAL/DISTRIBUTIONAL SHIFT vs a baseline (PSI/KS/chi-square on prediction distributions, concept drift, causal-DAG structural drift). Router keyword split proves it: 'model.*perform'->system_health(health_score), 'model.*degrad'->drift_check(drift_monitor). Practical reviewer rule: 'is model X healthy / what's its accuracy' = health_score; 'has model X drifted/degraded/PSI/distribution-shift over time' = drift_monitor.
- All FOUR agents ARE wired into INTENT_TO_AGENTS and reachable from intent_classifier keyword patterns - the 'believed UNMAPPED' premise is FALSE for every one of them. drift_check->drift_monitor (critical/10s), system_health->health_score (critical/5s), resource_allocation->resource_optimizer (critical/20s, plus 'high' in the performance_gap+resource_allocation multi-agent pattern), feedback->feedback_learner (critical/30s). CONTRACT_VALIDATION.md exists for all four.
- 'Unmapped agents' in #1337 means unmapped in the 4-stage classifier's DOMAIN_TO_AGENT (8 domains only, pattern_selector.py:26-38): resource_optimizer, health_score, feedback_learner, experiment_monitor have no classifier Domain. ALL 14 chat agents ARE reachable via the legacy router INTENT_TO_AGENTS - the gap is classifier coverage, not prod reachability.

## Agent contract reference (verified registry: data/agent_contracts.json)

- **cohort_constructor**: Tier-0 ML-pipeline agent that applies FDA/EMA-label inclusion/exclusion criteria to a real patient DataFrame to materialize an eligible cohort (patient IDs + eligibility audit trail + temporal lookback/followup validation) for downstream ML training.
  - covers: Given structured study params + a patient DataFrame, produce the list of eligible patient_journey_ids for a brand/indication (Remibrutinib/CSU, Fabhalta/PNH, Kisqali/HR+HER2-); Apply AND inclusion / AND-NOT exclusion criteria with a per-criterion eligibility log (removed/remaining counts); Validate temporal eligibility (lookback_days/followup_days) and emit exclusion stats + config_hash for reproducibility; Hand off cohort_spec + eligible_patient_ids to data_preparer
  - not: Chat/free-text 'how many patients are eligible / size the cohort' questions -> owned by cohort_profiler (constructor cannot run from a chat payload; its resolver fails closed); Natural-language explanation of an analysis -> owned by explainer; Upstream study-scoping (target var, problem type, brand/indication) -> owned by scope_definer (its upstream); Downstream feature/QC prep on the eligible set -> owned by data_preparer (its downstream)
- **cohort_profiler**: Tier-0 chat companion to cohort_constructor that answers the free-text 'size / define a cohort of ... patients' question with REAL DB-backed per-segment prescribing counts (new-Rx headline + severity tier + line-of-therapy breakdown) for one brand or all brands, never fabricating.
  - covers: How many patients are eligible / in the Remibrutinib (or Fabhalta / Kisqali) prescribing population?; Size / define the cohort for <brand>; Break down the <brand> population by disease-severity tier (low/medium/high); Break down the <brand> population by line of therapy (0/1/2/3+ prior lines); Same across all supported brands when no brand is named in the query
  - not: Materializing the actual eligible patient rows / IDs with FDA-EMA criteria + audit trail for an ML pipeline -> owned by cohort_constructor; Explaining an upstream analysis in narrative form -> owned by explainer; Arbitrary KPI/metric questions beyond cohort sizing -> the domain KPI agents (e.g. gap_analyzer) / chat KPI tool
- **tool_composer**: Tier-1 orchestration agent that answers a MULTI-FACETED query via a 4-phase pipeline (DECOMPOSE -> PLAN -> EXECUTE -> SYNTHESIZE): break the query into 2-6 atomic sub-questions with a dependency DAG, map each sub-question to a registered tool/agent-capability, execute tools in dependency order with retry + circuit-breaker resilience, and synthesize ONE coherent answer with confidence/citations/caveats. It composes OTHER agents' capabilities; it does not compute the analytics itself.
  - covers: Decomposing a multi-faceted query into 2-6 atomic sub-questions with a DAG (tier1-tool-composer-contracts.md:16-19, 96-116; decomposer.py); Planning: mapping each sub-question to a tool across MULTIPLE agent capabilities via the ToolRegistry (planner.py; CONTRACT_VALIDATION.md:8; categories CAUSAL/SEGMENTATION/GAP/EXPERIMENT/PREDICTION/MONITORING/EXPLANATION); Dependency-ordered execution with resilience: exponential backoff, circuit breaker, parallel groups (executor.py; tier1-tool-composer-contracts.md:336-529); Synthesis of cross-domain results into one NL answer with confidence, citations, caveats (synthesizer.py; ComposedResponse); Invoked ONLY when routing classifies the query as MULTI_FACETED / multi-domain-with-dependencies (agent.py:245-247)
  - not: Single-domain queries EVEN multi-STEP ones. A single mapped domain/intent always routes SINGLE_AGENT to the domain agent (e.g. causal_impact), which runs its OWN multi-step internal pipeline under one dispatch (pattern_selector.py:127-147); Independent multi-domain queries with NO dependencies -> route PARALLEL_DELEGATION, not tool_composer (pattern_selector.py:165-192); The routing decision itself: RouterNode/classifier decides; tool_composer is a target, and 'multi_faceted' is a META-signal, NOT an agent domain (router.py:284-298); The underlying analytical methods (DoWhy/EconML, SHAP, forecasting): it calls other agents' registered tools; Ambiguity resolution: that is CLARIFICATION_NEEDED, decided upstream in Stage 4
- **causal_impact**: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
  - covers: What is the causal effect of HCP engagement frequency on TRx (ATE + 95% CI + p-value)?; Does increasing copay assistance actually cause higher adherence once we control for these confounders?; Is this relationship causal or just correlation - does the effect survive placebo / random-common-cause / subset refutation?; How sensitive is the estimated effect to unmeasured confounding (E-value / robustness value)?; Build (or auto-discover) the causal DAG for treatment X on outcome Y and estimate the adjusted effect.
  - not: Which segments respond best / how the effect varies across segments (deep CATE + responder discovery) - owned by heterogeneous_optimizer (causal_impact only emits an optional cate_by_segment hint and hands off).; Where the biggest ROI/performance-gap opportunities are and their payback - owned by gap_analyzer.; Designing an A/B test or pre-registration to validate the effect - owned by experiment_designer.; Forecasting/predicting future outcome values - owned by prediction_synthesizer.; Solving optimal budget/resource allocation across territories - owned by resource_optimizer.
- **gap_analyzer**: Detects performance gaps across segments for a brand, estimates ROI/payback to close each gap, and prioritizes the opportunities into actionable buckets (quick wins, steady plays, strategic bets) with value-destroying ones suppressed.
  - covers: Where are the biggest ROI opportunities for Kisqali?; Which regions/specialties/deciles have the largest performance gaps vs potential (or vs target/benchmark)?; What's the ROI, payback period, and difficulty of closing each gap, and which are quick wins vs strategic bets?; Rank our opportunities by expected revenue impact and tell me the total addressable value.
  - not: Whether closing a gap will *causally* lift the outcome / estimating the intervention's treatment effect - owned by causal_impact (gap_analyzer suggests it as validation).; Which segments respond best to an intervention (treatment-effect heterogeneity / CATE) - owned by heterogeneous_optimizer.; Designing a controlled experiment to validate the top opportunity - owned by experiment_designer (its default suggested_next_agent).; Running the constrained optimal-allocation solver to distribute a fixed budget across opportunities - owned by resource_optimizer (gap_analyzer sizes/prioritizes but does not solve the allocation).
- **heterogeneous_optimizer**: Estimates treatment-effect heterogeneity (CATE) with EconML CausalForestDML, discovers high/low-responder segments, ranks effect-modifier importance, and recommends an optimal treatment-allocation policy across segments.
  - covers: Which HCP segments respond best (and worst) to increased engagement?; How does the treatment effect vary across segments - give me the CATE by segment?; Who are the high vs low responders and what features define them?; What's the optimal treatment-allocation policy across segments to maximize total lift?; Which effect modifiers drive the heterogeneity (feature importance)?
  - not: The single overall ATE with full refutation/robustness and E-value sensitivity plus DAG construction - owned by causal_impact (heterogeneous_optimizer reports overall_ate as a byproduct but not the refutation/sensitivity suite).; ROI/opportunity sizing and payback across KPIs - owned by gap_analyzer (its fallback_agent).; Designing an experiment to validate the segment-specific strategies - owned by experiment_designer (its default suggested_next_agent).; Forecasting future outcomes - owned by prediction_synthesizer.; The org-wide constrained budget-allocation solve - owned by resource_optimizer (heterogeneous_optimizer emits per-segment PolicyRecommendation treatment rates, not the global optimizer solution).
- **experiment_designer**: Designs a rigorous causal experiment end-to-end BEFORE it runs — turning a business question into a full, pre-registered A/B/RCT design with power analysis, validity audit, and runnable DoWhy analysis code.
  - covers: Design an A/B test / RCT / quasi-experiment to validate that X causes Y (e.g. 'design an experiment to test whether higher HCP engagement raises TRx'); What sample size / duration do I need to detect a Z% effect at 80% power?; Choose a randomization strategy (unit, method, stratification, blocking, cluster adjustment with ICC); Adversarially audit the validity of a proposed design and get mitigation recommendations (selection, confounding, contamination, attrition, external validity); Generate a pre-registration document + DoWhy causal-model spec + Python analysis template for an experiment; Optionally digital-twin pre-screen an intervention before committing to the design (enable_twin_simulation, default OFF)
  - not: Monitoring/health of an ALREADY-RUNNING experiment — SRM, enrollment, interim triggers, twin fidelity → experiment_monitor; Forecasting / ML point predictions → prediction_synthesizer; Estimating a causal effect (ATE) from existing observational data → causal_impact; Segment/CATE heterogeneity of an effect → heterogeneous_optimizer
- **experiment_monitor**: Monitors ALREADY-RUNNING A/B experiments for health problems — SRM, enrollment shortfalls, stale data, interim-analysis triggers, and digital-twin fidelity — using deterministic (no-LLM) checks.
  - covers: Check the health / status / issues of my active (running) experiments; Is there sample ratio mismatch (SRM) in experiment X? (chi-squared test); Have we reached any interim-analysis trigger / milestone?; Which running experiments have low enrollment or stale data?; Is the digital-twin prediction fidelity drifting for this experiment (simulated vs actual ATE)?; Show alerts/recommendations across all active experiments (check_all_active)
  - not: DESIGNING a new experiment, sample-size / power calc, validity audit → experiment_designer; Forecasting / ML predictions → prediction_synthesizer; Broad system/model health (not experiment-specific) → health_score; Data/model drift outside an experiment context → drift_monitor
- **drift_monitor**: Detects temporal/distributional drift (data, model-prediction, concept, and causal-DAG structural) by comparing a current window against a baseline window with statistical tests, then aggregates a composite score and alerts.
  - covers: Has feature/input data drifted vs baseline? (PSI + KS on feature distributions); Have model PREDICTIONS drifted / degraded over time? (KS on score dist, chi-square on class dist, PSI); Has the feature->target relationship changed? (concept drift: feature-importance correlation + performance degradation across periods); Has the causal DAG structure changed? (V4.4 structural drift: added/removed edges, edge-type changes, drift_score) - distinctive to this platform; Give me drift alerts with severity + recommended actions and an overall drift score over a time window
  - not: Point-in-time 'is model X performing acceptably RIGHT NOW / what's its current accuracy/auc/latency' operational snapshot -> health_score (its model_health node); Overall platform/component/pipeline/agent-availability health rollup -> health_score; Acting on drift (retraining, reallocating) -> not this agent
- **health_score**: Fast-path, no-LLM platform health rollup that measures FOUR dimensions - component, model, pipeline, agent - at a point in time and composes an overall 0-100 score + letter grade.
  - covers: What is overall system/platform health right now? (composite score A-F, dashboard-ready); Are system COMPONENTS up? (component_health: parallel component status checks); Are DATA PIPELINES fresh/succeeding? (pipeline_health: last run/success, freshness, rows); Are the deployed MODELS performing acceptably right now? (model_health: current accuracy/precision/recall/f1/auc/latency/error-rate/volume vs fixed thresholds -> healthy/degraded/unhealthy); Are the other AGENTS available/reliable? (agent_health: availability, latency, success rate); Quick health check (component-only fast variant)
  - not: Has anything DRIFTED / shifted / degraded relative to a baseline over time (PSI, KS, prediction-distribution shift, concept/structural drift) -> drift_monitor; Experiment/trial health (SRM, interim, enrollment) -> experiment_monitor; Why a specific model/metric is bad (root-cause) -> drift_monitor / explainer
- **prediction_synthesizer**: Aggregates predictions from multiple ML models into a single ensemble forecast with model provenance, agreement score, and prediction intervals (optionally enriched with historical context).
  - covers: Predict/forecast a target for an entity (e.g. 'forecast churn / TRx / adherence for HCP X'); What is the expected / likely value (probability) of Y over a given time horizon?; Combine several models into an ensemble prediction (average / weighted / voting / stacking) with a confidence interval; Give a prediction with model-agreement and uncertainty quantification; Enrich a forecast with similar historical cases, feature importance, historical accuracy, and trend direction (full graph)
  - not: WHY / causal-effect estimation (ATE) → causal_impact; In-depth explanation of a prediction's drivers → explainer (it even sets suggested_next_agent='explainer'); Designing an experiment to test a hypothesis → experiment_designer; Segment/CATE heterogeneity → heterogeneous_optimizer
- **resource_optimizer**: Computational (no-LLM core) optimizer that allocates a constrained resource (budget/rep-time/samples/calls) across entities to maximize an objective, with optional what-if scenarios, sensitivity, and ROI/impact projection.
  - covers: How should we allocate budget/reps/samples/calls across territories/HCPs/segments? (linear/MILP/nonlinear solve); Optimize for maximize_outcome / maximize_roi / minimize_cost / balance under budget/capacity/min-coverage/max-frequency constraints; What-if scenario analysis across allocation options (run_scenarios); Constraint sensitivity analysis (shadow prices / sensitivity_analysis); Projected total outcome, ROI, and impact-by-segment for a recommended allocation
  - not: Where is the performance GAP / opportunity sizing (the 'what to fix' before allocation) -> gap_analyzer; Causal effect of a lever (does spend CAUSE uplift) -> causal_impact / heterogeneous_optimizer; Predicting an outcome value -> prediction_synthesizer
- **explainer**: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
  - covers: Explain / summarize / interpret these analysis results in plain language (narrative | structured | presentation | brief); Tailor an explanation to an audience (executive / analyst / data_scientist); Extract findings/recommendations/warnings/opportunities + suggest visuals + follow-up questions from provided results; Catch-all narration for general/unclassified queries (default dispatch)
  - not: Producing the underlying causal effect estimate -> owned by causal_impact (explainer only narrates its output); Sizing a gap / opportunity -> owned by gap_analyzer; Segment/CATE estimation -> owned by heterogeneous_optimizer; Cohort sizing -> owned by cohort_profiler; cohort materialization -> cohort_constructor; Multi-sub-question decomposition -> owned by tool_composer (explainer is its fallback, not its substitute)
- **feedback_learner**: Async/batch self-improvement agent: collects a window of user feedback + agent outcomes, mines patterns (LLM reasoning), evaluates responses against a rubric, and emits learning recommendations + knowledge/prompt updates (incl. DSPy training signals).
  - covers: Analyze a batch of feedback over a time range and find recurring failure/quality patterns (detected_patterns, clusters, root-cause hypotheses); Produce learning recommendations (prompt_update/model_retrain/data_update/config_change/new_capability) + priority improvements; Propose/apply knowledge-base updates (prompts, thresholds, baselines, agent configs); Rubric / AI-as-judge evaluation of agent responses (rubric_node); Causal-discovery-specific feedback loop: track DAG accuracy by algorithm, recommend params (V4.4 discovery_feedback_node); DSPy training-signal collection for MIPROv2/GEPA optimization and learning summaries
  - not: Real-time answer to a single user query (it is batch/async, keyed by batch_id + time range, not a per-query responder); Live model performance/drift measurement -> health_score / drift_monitor; Serving the optimized prompts at inference time -> target agents / DSPy runtime, not this agent

## 1. Why did Kisqali TRx drop in Q1 in the northeast region?

- rows: 4  |  pipeline said: SINGLE_AGENT×4  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: Query requires multi-step analysis: retrieve Q1 sales data for Kisqali TRx in northeast, identify trend, investigate root causes (market, clinical, competitive factors). Dependent steps across sales and market intelligence domains.
- demo doc expectation: intent `causal_analysis` → seed turn for the memory follow-up (question A.9-seed)
- contract check — candidate single agent(s):
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
- proposed verdict (contract-based): `single_agent:causal_impact`
  - reasoning: 'Why did X drop' is a causal-driver ask — one domain (CAUSAL_ANALYSIS). The judge's rationale (retrieve data -> analyze -> investigate) counts causal_impact's own internal pipeline steps as composition. Note: the pipeline actually dispatched explainer, i.e. the classifier under-detected the causal domain — a classifier miss, not a composition case.
  - empirical status: qualified — domain confirmed (live surface answered with KPI + causal drivers) but the forced route FAILS CLOSED: causal_impact has no input resolver and hard-requires treatment_var/outcome_var/confounders/data_source no chat query supplies
- empirical evidence (q01, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `causal_impact` (intent `causal_effect`): FAILED CLOSED — causal_impact: Missing required field(s): treatment_var, outcome_var, confounders, data_source; explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[kpi_calculate_tool] (16086.4ms, 1878 chars); response: “**Kisqali TRx – Northeast, Q1 2025 (Jan 1–Apr 1, 2025):** 142 total prescriptions (window applied as requested). Note: no non-overlapping prior-period figure was returned for the Northeast, so I can't quantify the "drop" magnitude directly — the tools didn't return a Q4 2024 (or earlier) baseline for comparison. What I can offer is the likely **drivers** behind a decline, based on the brand-level causal model and regional trigger signals: **Top c…”
- **Verdict**: _____

## 2. Where are the biggest untapped opportunities to grow Remibrutinib market share?

- rows: 3  |  pipeline said: CLARIFICATION_NEEDED×3  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires multi-step analysis: market data (competitive landscape, patient segments), clinical/regulatory insights (indications, pipeline), and commercial strategy. Dependent tool calls across domains needed.
- demo doc expectation: intent `recommendation` → gap_analyzer (ROI opportunity ranking) (question 2.5)
- contract check — candidate single agent(s):
  - `gap_analyzer`: Detects performance gaps across segments for a brand, estimates ROI/payback to close each gap, and prioritizes the opportunities into actionable buckets (quick wins, steady plays, strategic bets) with value-destroying ones suppressed.
- proposed verdict (contract-based): `single_agent:gap_analyzer`
  - reasoning: Verbatim in gap_analyzer's covers list ('Where are the biggest ROI opportunities for <brand>'). One domain (GAP_ANALYSIS); opportunity detection + ROI + prioritization is its single-dispatch pipeline.
  - empirical status: qualified — right owner but fails closed on bare chat: the resolver reads parameters/parsed_query (never populated), not query text, so even the in-text brand 'Remibrutinib' is invisible; live answered via kpi_calculate_tool
- empirical evidence (q02, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `gap_analyzer` (intent `performance_gap`): FAILED CLOSED — gap_analyzer: gap_analyzer needs structured inputs that could not be grounded in real data (the dispatch names no brand (parameters / parsed_query entities / user_context), so there is no real business_metrics substrate to derive metr…
  - live AG-UI response (real UI brain): tools=[kpi_calculate_tool] (25315.5ms, 2666 chars); response: “## Remibrutinib TRx Share — Growth Opportunities **Current position:** Remibrutinib holds **33.2% TRx share** of the tracked portfolio (Fabhalta + Kisqali + Remibrutinib), based on the most recent 30 days of data (through 2026‑07‑27). Note: this is share of the tracked portfolio only — not a comparison against external competitors like Xolair or Dupixent, which aren't in the data model. ### Biggest untapped opportunities **1. HCP coverage expansi…”
- **Verdict**: _____

## 3. Design an experiment to measure whether speaker programs increase Fabhalta NRx

- rows: 3  |  pipeline said: CLARIFICATION_NEEDED×3  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires multi-step planning: define speaker program metrics, identify Fabhalta NRx data sources, design experimental controls, and integrate findings. Dependent steps across clinical/commercial domains demand TOOL_COMPOSER, not clarification.
- demo doc expectation: intent `recommendation` → experiment_designer with Digital Twin pre-screening (question 3.4)
- contract check — candidate single agent(s):
  - `experiment_designer`: Designs a rigorous causal experiment end-to-end BEFORE it runs — turning a business question into a full, pre-registered A/B/RCT design with power analysis, validity audit, and runnable DoWhy analysis code.
- proposed verdict (contract-based): `single_agent:experiment_designer`
  - reasoning: End-to-end experiment design (metrics, controls, power, validity audit) is experiment_designer's single-dispatch contract. One domain (EXPERIMENTATION); the judge decomposed the agent's own internal phases.
  - empirical status: qualified — experiment_designer ACCEPTS the chat query (no fail-closed) but blew its 60s dispatch budget; the live surface took 88s — route right, timeout budget unrealistic
- empirical evidence (q03, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `experiment_designer` (intent `experiment_design`): FAILED CLOSED — experiment_designer: Agent timed out after 60000ms
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (88351.7ms, 3118 chars); response: “The dedicated experiment-designer agent timed out, so here's a direct design based on Fabhalta's causal drivers (data source: causal path registry, synthetic): ## Relevant causal context (Fabhalta → NRx) None of the registered causal paths for Fabhalta NRx directly include "speaker program" as a node, but the closest proxies are: - **intent_to_prescribe → NRx** (via new_patient_starts): effect size 0.336, confidence 0.766, ~43-day lag, est. busin…”
- **Verdict**: _____

## 4. What drove the change in new patient starts for Fabhalta?

- rows: 2  |  pipeline said: PARALLEL_DELEGATION×2  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: Query requires sequential analysis: first retrieve Fabhalta patient-start trends, then investigate causal factors (clinical, competitive, regulatory). Dependent steps warrant TOOL_COMPOSER over parallel agents.
- demo doc expectation: — (not one of the 51 authored questions)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
  - `drift_monitor`: Detects temporal/distributional drift (data, model-prediction, concept, and causal-DAG structural) by comparing a current window against a baseline window with statistical tests, then aggregates a composite score and alerts.
- proposed verdict (contract-based): `single_agent:causal_impact`
  - reasoning: Causal-driver question on a business KPI. drift_monitor (the pipeline's other parallel target) is inapplicable — nothing here is about model/data drift vs baseline. One domain.
  - empirical status: qualified — same causal_impact fail-closed as q01; live causal_analysis_tool answer corroborates the causal-driver domain
- empirical evidence (q04, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `causal_impact` (intent `causal_effect`): FAILED CLOSED — causal_impact: Missing required field(s): treatment_var, outcome_var, confounders, data_source; explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (13947.3ms, 1542 chars); response: “**Fabhalta NBRx: current level = 3,312 new-patient starts** (most recent 30-day reporting window, data through 2026-07-27, per E2I synthetic engine — no specific period was requested, so this reflects the engine's default window rather than a custom range). **Key drivers identified (causal analysis, min. confidence 0.5):** | Driver | Effect on NBRx | Confidence | Time Lag | Est. Business Impact | |---|---|---|---|---| | **HCP coverage** → prescri…”
- **Verdict**: _____

## 5. What is the causal impact of rep visits on TRx for Kisqali?

- rows: 2  |  pipeline said: CLARIFICATION_NEEDED×2  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires causal analysis linking rep activity (sales force data) to prescription outcomes (TRx metrics) for a specific drug. This demands dependent, multi-step tool calls across sales and clinical domains—not clarification.
- demo doc expectation: intent `causal_analysis` → causal_impact (ATE with refutation gating) (question 1.5)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
- proposed verdict (contract-based): `single_agent:causal_impact`
  - reasoning: Textbook ATE ask — the literal first entry in causal_impact's covers list. The judge's 'sales + clinical domains' framing mistakes data sources for agent-capability domains.
  - empirical status: qualified — textbook-ATE domain corroborated (live surface chose causal_analysis_tool); forced route fails closed (no input resolver)
- empirical evidence (q05, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `causal_impact` (intent `causal_effect`): FAILED CLOSED — causal_impact: Missing required field(s): treatment_var, outcome_var, confounders, data_source; explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (12838.5ms, 1524 chars); response: “**Rep visits (detailing frequency) → TRx for Kisqali** (source: causal analysis registry, synthetic data, engine's standard reporting window — not a specific period you requested): - **Path:** `rep_detailing_frequency → hcp_engagement → trx_volume` - **Effect size:** **0.166** (a moderate positive causal effect on TRx volume) - **Confidence:** **0.796** (fairly high, via backdoor linear regression) - **Time lag:** **77 days** — impact on TRx show…”
- **Verdict**: _____

## 6. Design an experiment to test whether increasing rep visits improves Fabhalta adoption

- rows: 2  |  pipeline said: SINGLE_AGENT×2  |  judge: TOOL_COMPOSER @ 0.82
- judge rationale: Query requires dependent steps: clinical/adoption domain expertise (experiment design), market analytics (rep visit data analysis), and statistical methodology. These domains interact; not independent parallel tasks.
- demo doc expectation: intent `recommendation` → experiment_designer (consistency check vs 3.4) (question 3.6)
- contract check — candidate single agent(s):
  - `experiment_designer`: Designs a rigorous causal experiment end-to-end BEFORE it runs — turning a business question into a full, pre-registered A/B/RCT design with power analysis, validity audit, and runnable DoWhy analysis code.
- proposed verdict (contract-based): `single_agent:experiment_designer`
  - reasoning: Same as the speaker-programs design ask: single-domain EXPERIMENTATION, covered end-to-end by experiment_designer under one dispatch. Pipeline's SINGLE_AGENT decision was correct.
  - empirical status: qualified — same as q03: real work, 60s timeout; live took 90s
- empirical evidence (q06, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `experiment_designer` (intent `experiment_design`): FAILED CLOSED — experiment_designer: Agent timed out after 60000ms
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (90466.4ms, 3423 chars); response: “# Experiment Design: Rep Visit Frequency → Fabhalta Adoption ## Context from the data - Current **NBRx (Fabhalta)** = 3,312, based on the engine's default reporting window (most recent 30 days, data through 2026-07-27) — not a specific period you requested, so treat this as the current baseline snapshot, not a trend. - Causal analysis (min confidence 0.5) confirms `treatment_arm` (a proxy for rep-engagement/promotional exposure) → `treatment_init…”
- **Verdict**: _____

## 7. How confident are we in the rep-visit effect - did it pass refutation tests?

- rows: 2  |  pipeline said: PARALLEL_DELEGATION×2  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: Query requires sequential reasoning: first establish rep-visit effect magnitude, then evaluate refutation test results against it. Dependent steps demand TOOL_COMPOSER, not parallel independent agents.
- demo doc expectation: intent `causal_analysis` → causal_impact refutation report (fail-closed gating, E-value) (question 5.7)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
- proposed verdict (contract-based): `single_agent:causal_impact`
  - reasoning: Refutation + sensitivity (E-value) reporting are stages INSIDE causal_impact's contract, not a second domain. Caveat: as a session follow-up on an already-computed effect, narrating the stored result via explainer is defensible — but that is still SINGLE_AGENT, never TOOL_COMPOSER.
  - empirical status: qualified — forced fails closed; live causal_analysis_tool answered; caveat: ran standalone, without the session context this follow-up assumes
- empirical evidence (q07, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `causal_impact` (intent `causal_effect`): FAILED CLOSED — causal_impact: Missing required field(s): treatment_var, outcome_var, confounders, data_source; explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (22188.9ms, 2679 chars); response: “**Rep-visit effect: strong confidence, generally consistent across brands — but no explicit refutation-test flag in the causal registry itself.** From the causal paths registry (12 chains involving rep detailing/visits, all brands, no date window applied), confidence scores range **0.77–0.93**, all well above typical significance thresholds, using `backdoor.linear_regression` estimation: | Cause | Effect | Brand | Effect Size | Confidence | Lag (…”
- **Verdict**: _____

## 8. Compare TRx market share for Kisqali vs its competitors over the last 6 months, explain what's driving the difference, and recommend where to focus reps next quarter

- rows: 2  |  pipeline said: PARALLEL_DELEGATION×2  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires sequential dependency: first fetch market share data, then analyze competitive drivers based on that data, then synthesize insights for rep allocation. Steps are interdependent, not parallel.
- demo doc expectation: intent `multi_faceted` → TOOL_COMPOSER - 3 sub-questions with LOGICAL_SEQUENCE dependencies (KPI -> causal -> resource) (question 6.1)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
  - `gap_analyzer`: Detects performance gaps across segments for a brand, estimates ROI/payback to close each gap, and prioritizes the opportunities into actionable buckets (quick wins, steady plays, strategic bets) with value-destroying ones suppressed.
  - `heterogeneous_optimizer`: Estimates treatment-effect heterogeneity (CATE) with EconML CausalForestDML, discovers high/low-responder segments, ranks effect-modifier importance, and recommends an optimal treatment-allocation policy across segments.
- proposed verdict (contract-based): `agree`
  - reasoning: Genuinely spans >=2 distinct domains (KPI/gap compare -> causal drivers -> resource allocation) AND the recommendation depends on the drivers — both gate conditions hold. Matches the demo doc's authored multi_faceted (6.1). Not the defined 2-intent parallel-pair case: three linked facets.
  - empirical status: qualified — composition domain stands (live answered all three facets, 4.1k chars) but tool_composer CRASHES in planning: AIMessage.content list-vs-str (#1350) — the multi_faceted route is operationally broken
- empirical evidence (q08, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `tool_composer` (intent `multi_faceted`): SUCCEEDED (30112.3ms); response: “Unable to complete analysis: Planning failed: Failed to create execution plan: the JSON object must be str, bytes or bytearray, not list”
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (27887.9ms, 4091 chars); response: “I'll pull Kisqali's TRx share trend, causal drivers, and existing recommendations. Note upfront: this data model tracks **TRx Share within the Novartis tracked portfolio** (Kisqali + Fabhalta + Remibrutinib), not share vs. external competitors like Ibrance/Verzenio — those aren't in the data model, so I'll frame accordingly. ## Kisqali TRx Share — Last 6 Months (Jan 29 – Jul 29, 2026) **Important framing correction:** The "TRx Share" metric in ou…”
- **Verdict**: _____

## 9. Give me a launch-readiness snapshot for Fabhalta: % PNH tested, NRx trend, top adoption barriers, and one experiment we should run next

- rows: 2  |  pipeline said: CLARIFICATION_NEEDED×2  |  judge: TOOL_COMPOSER @ 0.92
- judge rationale: Query requires dependent steps: fetch PNH testing data, trend NRx, analyze adoption barriers, then synthesize experiment recommendation. Multi-domain coordination needed, not genuinely ambiguous.
- demo doc expectation: intent `multi_faceted` → TOOL_COMPOSER - 4-facet decomposition ending in experiment_designer (question 6.4)
- contract check — candidate single agent(s):
  - `experiment_designer`: Designs a rigorous causal experiment end-to-end BEFORE it runs — turning a business question into a full, pre-registered A/B/RCT design with power analysis, validity audit, and runnable DoWhy analysis code.
- proposed verdict (contract-based): `agree`
  - reasoning: Cross-domain (KPI + gap/causal barriers + experiment design) and the 'experiment we should run next' facet depends on the barriers facet. Demo doc authored it as 4-facet TOOL_COMPOSER (6.4). Weakest link: the first three facets alone would be PARALLEL — the dependent final facet is what makes composition.
  - empirical status: qualified — same #1350 crash; live kpi_calculate_tool answered the snapshot (3.5k chars)
- empirical evidence (q09, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `tool_composer` (intent `multi_faceted`): SUCCEEDED (7987.1ms); response: “Unable to complete analysis: Decomposition failed: Failed to decompose query: the JSON object must be str, bytes or bytearray, not list”
  - live AG-UI response (real UI brain): tools=[kpi_calculate_tool] (27173.2ms, 3497 chars); response: “# Fabhalta Launch-Readiness Snapshot **Note on scope:** No direct "% PNH tested" metric exists in the data model — PNH testing/diagnosis rate isn't a tracked KPI in this system. I can't report that figure; it would need to come from a separate diagnostic-funnel data source. ## NRx Trend (Jan 29 – Jul 29, 2026, 181-day window) - **NRx = 3,610** new prescriptions - ⚠️ **Coverage warning from the tool**: *"91% of this 181-day total falls in its most…”
- **Verdict**: _____

## 10. Show me the KPI summary for Kisqali

- rows: 1  |  pipeline said: SINGLE_AGENT×1  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: KPI summary requires aggregating multiple data domains: clinical efficacy, safety metrics, market performance, and regulatory status. This demands coordinated multi-step tool calls across specialized domains, not a single explainer agent.
- demo doc expectation: — (not one of the 51 authored questions)
- contract check — candidate single agent(s):
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
- proposed verdict (contract-based): `single_agent:explainer`
  - reasoning: A KPI lookup/summary — no second domain, no dependency; the judge's 'clinical/safety/market/regulatory domains' rationale invents facets not in the query. In prod this is really the chatbot KPI fast path (kpi_query), outside the 14-agent registry; within the orchestrator taxonomy the catch-all narration owner is explainer.
  - empirical status: qualified — explainer fails closed with no upstream agent results, confirming the proposal's own note that prod serves this via the KPI fast path; live: e2i_data_query_tool
- empirical evidence (q10, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `explainer` (intent `explanation`): FAILED CLOSED — explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[e2i_data_query_tool] (18167.7ms, 1612 chars); response: “## Kisqali KPI Summary (data through 2026-07-27, most recent 30-day reporting window) | KPI | Value | |---|---| | **TRx** | 13,242 | | **NRx** | 3,216 | | **TRx Share** | 30.9% | | **Conversion Rate** | 62.4% | *TRx Share note: this is Kisqali's share of the tracked portfolio's total prescriptions (Fabhalta + Kisqali + Remibrutinib), not share vs. external competitors like Xolair/Dupixent — those aren't in the data model.* ### Regional Detail (mo…”
- **Verdict**: _____

## 11. Build a patient cohort for Remibrutinib CSU with inclusion criteria for adults over 18 diagnosed in 2024

- rows: 1  |  pipeline said: SINGLE_AGENT×1  |  judge: TOOL_COMPOSER @ 0.88
- judge rationale: Query requires multi-step dependent operations: drug-indication validation (Remibrutinib→CSU), temporal filtering (2024 diagnoses), demographic criteria application, then cohort assembly. Cohort_profiler alone cannot independently validate drug-indication pairing or execute cross-domain filtering.
- demo doc expectation: intent `search` → cohort_constructor (COHORT_DEFINITION domain) (question 2.1)
- contract check — candidate single agent(s):
  - `cohort_constructor`: Tier-0 ML-pipeline agent that applies FDA/EMA-label inclusion/exclusion criteria to a real patient DataFrame to materialize an eligible cohort (patient IDs + eligibility audit trail + temporal lookback/followup validation) for downstream ML training.
  - `cohort_profiler`: Tier-0 chat companion to cohort_constructor that answers the free-text 'size / define a cohort of ... patients' question with REAL DB-backed per-segment prescribing counts (new-Rx headline + severity tier + line-of-therapy breakdown) for one brand or all brands, never fabricating.
- proposed verdict (contract-based): `single_agent:cohort_profiler`
  - reasoning: One domain (COHORT_DEFINITION). cohort_constructor can never serve chat (resolver fails closed to NeedsStructuredInput by design); cohort_profiler owns the chat cohort ask. If the user truly wants materialization with criteria, the correct platform behavior is the fail-closed guidance to the ML pipeline — still not composition.
  - empirical status: confirmed — the ONLY operational single-agent route in the pass (real data, 3.6s); caveat: response is query-insensitive (canned all-brands profile; brand filter and inclusion criteria ignored)
- empirical evidence (q11, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `cohort_profiler` (intent `cohort_definition`): SUCCEEDED (3575.8ms); response: “**Patient cohort profile — all brands** Eligible prescribing population sized by the clinical segment axes that exist in the data today (new prescriptions, most recent 30 days): ### Remibrutinib — 3,283 new-Rx patients _By disease-severity tier:_ | Severity tier | New-Rx patients | |---|---| | Low severity | 834 | | Medium severity | 1,738 | | High severity | 711 | _By line of therapy:_ | Line of therapy | New-Rx patients | |---|---| | 0 prior li…”
  - live AG-UI response (real UI brain): tools=[clinical_context_tool] (16832.6ms, 1939 chars); response: “I can't build an actual patient cohort dataset here — this system doesn't have patient-level medical records or a cohort-construction database. What I retrieved instead is the **label/clinical context for Remibrutinib (RHAPSIDO)** from OpenFDA/ClinicalTrials.gov, which can inform how such criteria *should* be scoped, but I'm not able to generate or query real 2024-diagnosed patient records. **What the real label supports for inclusion-criteria de…”
- **Verdict**: _____

## 12. What is driving the drop in Remibrutinib NRx in the northeast region this quarter?

- rows: 1  |  pipeline said: CLARIFICATION_NEEDED×1  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires dependent steps: retrieve Remibrutinib NRx trends (market data agent), filter northeast region (geographic agent), analyze drivers (clinical/competitive intelligence agent). Multi-step orchestration needed, not clarification.
- demo doc expectation: intent `causal_analysis` → causal_impact (question 2.4)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
- proposed verdict (contract-based): `single_agent:causal_impact`
  - reasoning: Same shape as the Kisqali Q1 driver ask: single-domain causal attribution; region/quarter are filters, not domains.
  - empirical status: qualified — same causal_impact fail-closed pattern; live kpi_calculate_tool answered
- empirical evidence (q12, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `causal_impact` (intent `causal_effect`): FAILED CLOSED — causal_impact: Missing required field(s): treatment_var, outcome_var, confounders, data_source; explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[kpi_calculate_tool] (24902.5ms, 2139 chars); response: “**Bottom line: I can't confirm there's actually a quarter-over-quarter drop in Northeast Remibrutinib NRx** — the data available doesn't support that conclusion, for two reasons: **1. Coverage issue with the reported figure** - Remibrutinib NRx, Northeast, **91-day window (Apr 29 – Jul 29, 2026)**: **482 prescriptions** - Tool flagged: *"93% of this 91-day total falls in its most recent 30 days — the data is not evenly distributed across the wind…”
- **Verdict**: _____

## 13. How can I optimize resource allocation for Remibrutinib in the northeast region?

- rows: 1  |  pipeline said: SINGLE_AGENT×1  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: Query requires multi-step analysis: drug mechanism/efficacy data, regional supply/demand metrics, and allocation optimization logic. These dependent steps span clinical and operational domains, necessitating coordinated tool composition rather than single-agent explanation.
- demo doc expectation: intent `recommendation` → resource_optimizer (question 2.6)
- contract check — candidate single agent(s):
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
  - `resource_optimizer`: Computational (no-LLM core) optimizer that allocates a constrained resource (budget/rep-time/samples/calls) across entities to maximize an objective, with optional what-if scenarios, sensitivity, and ROI/impact projection.
- proposed verdict (contract-based): `single_agent:resource_optimizer`
  - reasoning: Verbatim resource_optimizer territory (allocate reps/budget across a region). One domain. Note: resource_optimizer is classifier-unmapped (legacy INTENT_TO_AGENTS only) — the pipeline dispatching explainer reflects that coverage gap, not a composition need.
  - empirical status: qualified — right owner; the resolver demands a structured allocation problem (entities+coefficients+budget) no chat query supplies; live answered via e2i_data_query_tool
- empirical evidence (q13, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `resource_optimizer` (intent `resource_allocation`): FAILED CLOSED — resource_optimizer: resource_optimizer needs structured inputs that could not be grounded in real data (a real allocation problem (entities with response coefficients AND a budget constraint) must be supplied; no per-entity allocation/respo…
  - live AG-UI response (real UI brain): tools=[e2i_data_query_tool] (20618.1ms, 2897 chars); response: “## Remibrutinib – Northeast Region: Resource Allocation Optimization **Current performance snapshot** (most recent 30-day window, data through 2026-07-27): - **TRx**: 3,045 (portfolio-tracked, no window override applied — this is the engine's default reporting window) - **TRx Share**: 34.1% of tracked portfolio (Fabhalta + Kisqali + Remibrutinib) — status "good" - From the recommendations feed (2026-07-01 snapshot): TRx at 63,670 vs. target 82,57…”
- **Verdict**: _____

## 14. Predict which HCP segments are most likely to increase Fabhalta prescriptions next quarter

- rows: 1  |  pipeline said: CLARIFICATION_NEEDED×1  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires multi-step analysis: market data retrieval, HCP segmentation analysis, prescription trend modeling, and forecasting. These dependent steps span multiple domains, necessitating orchestrated tool composition rather than clarification.
- demo doc expectation: intent `prediction` → prediction_synthesizer (question 3.3)
- contract check — candidate single agent(s):
  - `prediction_synthesizer`: Aggregates predictions from multiple ML models into a single ensemble forecast with model provenance, agreement score, and prediction intervals (optionally enriched with historical context).
- proposed verdict (contract-based): `single_agent:prediction_synthesizer`
  - reasoning: A forecast-and-rank ask — PREDICTION domain, covered by prediction_synthesizer (ensemble + intervals). Not heterogeneous_optimizer: no treatment/intervention whose effect varies; it's an outcome forecast by segment.
  - empirical status: qualified — right domain owner but the ask is unservable on BOTH surfaces: forced fails closed (no champion model / no entity); live explicitly says no per-HCP-segment likelihood prediction exists
- empirical evidence (q14, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `prediction_synthesizer` (intent `prediction`): FAILED CLOSED — prediction_synthesizer: prediction_synthesizer needs structured inputs that could not be grounded in real data (no registered champion model and no specific real entity to predict for; a chat query names neither, so a prediction cannot be synth…
  - live AG-UI response (real UI brain): tools=[e2i_data_query_tool] (21900.4ms, 2903 chars); response: “**Fabhalta HCP-Segment Prediction — data caveat first:** The available tools don't produce a per-HCP-segment "likelihood to increase Rx" prediction (no such registry output was returned). What I can give you is the regional TRx momentum (forecast for Jun–Jul 2026) plus the causal drivers of TRx growth, which together point to which HCP segments/regions to prioritize. **Regional TRx trajectory (forecast, Jun 1 – Jul 1 2026, Fabhalta):** | Region |…”
- **Verdict**: _____

## 15. Build a cohort of high-value HCPs who prescribed more than 50 TRx last quarter

- rows: 1  |  pipeline said: PARALLEL_DELEGATION×1  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires sequential steps: (1) filter prescribers by TRx threshold, (2) identify HCP attributes, (3) build cohort. Dependencies exist between steps; not independent parallel tasks.
- demo doc expectation: intent `search` → cohort_constructor with quantitative criterion (question 4.2)
- contract check — candidate single agent(s):
  - `cohort_constructor`: Tier-0 ML-pipeline agent that applies FDA/EMA-label inclusion/exclusion criteria to a real patient DataFrame to materialize an eligible cohort (patient IDs + eligibility audit trail + temporal lookback/followup validation) for downstream ML training.
  - `cohort_profiler`: Tier-0 chat companion to cohort_constructor that answers the free-text 'size / define a cohort of ... patients' question with REAL DB-backed per-segment prescribing counts (new-Rx headline + severity tier + line-of-therapy breakdown) for one brand or all brands, never fabricating.
  - `heterogeneous_optimizer`: Estimates treatment-effect heterogeneity (CATE) with EconML CausalForestDML, discovers high/low-responder segments, ranks effect-modifier importance, and recommends an optimal treatment-allocation policy across segments.
- proposed verdict (contract-based): `extend:cohort_profiler — HCP-entity cohorts with quantitative KPI thresholds (current contract profiles patient prescribing populations by brand/severity/LoT only)`
  - reasoning: Still single-domain (COHORT_DEFINITION) — TOOL_COMPOSER is wrong either way. But cohort_profiler's verified contract covers patient populations per brand, not HCP-level cohorts by TRx threshold; no agent today cleanly owns this. Product call: extend profiler's contract, or fail closed with guidance. This is the one genuine contract-gap row in the batch.
  - empirical status: confirmed — the contract gap is real on both surfaces: forced returns the canned patient profile ignoring the HCP ask entirely; live explicitly states its tools cannot build HCP-level cohorts (extend remains the user's product call)
- empirical evidence (q15, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `cohort_profiler` (intent `cohort_definition`): SUCCEEDED (26.4ms); response: “**Patient cohort profile — all brands** Eligible prescribing population sized by the clinical segment axes that exist in the data today (new prescriptions, most recent 30 days): ### Remibrutinib — 3,283 new-Rx patients _By disease-severity tier:_ | Severity tier | New-Rx patients | |---|---| | Low severity | 834 | | Medium severity | 1,738 | | High severity | 711 | _By line of therapy:_ | Line of therapy | New-Rx patients | |---|---| | 0 prior li…”
  - live AG-UI response (real UI brain): tools=[e2i_data_query_tool] (18920.5ms, 2540 chars); response: “I need to be upfront about a limitation here: the E2I tools I have access to don't support building an HCP-level cohort (i.e., a list of individual healthcare providers with their per-HCP TRx counts). Here's what's actually available versus what your request needs: **What I can compute:** - Aggregate TRx totals for a brand over a time window (e.g., "last quarter") - Breakdowns by severity tier, line-of-therapy, region, and (Remibrutinib only) bio…”
- **Verdict**: _____

## 16. Which HCP segments show the strongest treatment effect for Remibrutinib?

- rows: 1  |  pipeline said: PARALLEL_DELEGATION×1  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: Query requires sequential analysis: first identify treatment effects, then segment HCPs by those effects. Dependencies between steps demand TOOL_COMPOSER, not independent parallel agents.
- demo doc expectation: intent `causal_analysis` → heterogeneous_optimizer (CATE) (question 4.4)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
  - `gap_analyzer`: Detects performance gaps across segments for a brand, estimates ROI/payback to close each gap, and prioritizes the opportunities into actionable buckets (quick wins, steady plays, strategic bets) with value-destroying ones suppressed.
  - `heterogeneous_optimizer`: Estimates treatment-effect heterogeneity (CATE) with EconML CausalForestDML, discovers high/low-responder segments, ranks effect-modifier importance, and recommends an optimal treatment-allocation policy across segments.
- proposed verdict (contract-based): `single_agent:heterogeneous_optimizer`
  - reasoning: CATE-by-segment is verbatim in heterogeneous_optimizer's covers list ('Which HCP segments respond best'). The judge's 'first effects, then segmentation' sequence is the agent's own estimate_cate -> analyze_segments pipeline — internal steps, one dispatch.
  - empirical status: qualified — covers-verbatim domain stands; forced fails closed (no KPI substrate/treatment/rows groundable from chat) and even the live surface's heterogeneous_optimizer call returned no records for Remibrutinib
- empirical evidence (q16, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `heterogeneous_optimizer` (intent `segment_analysis`): FAILED CLOSED — heterogeneous_optimizer: heterogeneous_optimizer needs structured inputs that could not be grounded in real data (no recognized KPI substrate with a defined treatment and >=100 real rows to bind the causal spec; a chat query alone cannot name th…; gap_analyzer: gap_analyzer needs structured inputs that could not be grounded in real data (the dispatch names no brand (parameters / parsed_query entities / user_context), so there is no real business_metrics substrate to derive metr…
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (13599.1ms, 1534 chars); response: “Here's what the causal analysis engine (Remibrutinib) shows on treatment effect heterogeneity: **Note on segmentation:** The dedicated `heterogeneous_optimizer` agent analysis returned **no records** for Remibrutinib, so I can't report a formal HCP-segment-level treatment-effect ranking from that source. **What the causal paths registry does show** (via `disease_severity` and `engagement_score` as mediators — confidence ~0.94–0.945): | Causal Pat…”
- **Verdict**: _____

## 17. Did rep actions driven by triggers actually lift prescriptions?

- rows: 1  |  pipeline said: CLARIFICATION_NEEDED×1  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: Query requires multi-step analysis: (1) identify rep actions/triggers in CRM/field data, (2) correlate with prescription outcomes, (3) measure lift. Dependent cross-domain tool calls needed—not genuinely ambiguous.
- demo doc expectation: intent `causal_analysis` → Action Rate Uplift KPI -> causal framing (question 4.7)
- contract check: no candidate agent on record (pipeline dispatched none; pick from the reference above)
- proposed verdict (contract-based): `single_agent:causal_impact`
  - reasoning: 'Did X actually lift Y' is a causal-uplift ask (treatment = trigger-driven rep actions, outcome = prescriptions). One domain; CRM-vs-prescription data sources are not separate agent-capability domains.
  - empirical status: qualified — same causal_impact fail-closed; live causal_analysis_tool answered
- empirical evidence (q17, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `causal_impact` (intent `causal_effect`): FAILED CLOSED — causal_impact: Missing required field(s): treatment_var, outcome_var, confounders, data_source; explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (15660.5ms, 2228 chars); response: “**Yes — rep detailing (a key trigger-driven action) shows a consistent, statistically credible positive effect on TRx across all three brands.** From the causal-paths registry (min confidence 0.5): | Brand | Cause → Effect | Effect size | Confidence | Time lag | Est. business impact | |---|---|---|---|---|---| | Remibrutinib | rep_detailing_frequency → trx_volume (via hcp_engagement) | +0.298 | 0.897 | 86 days | ~$91.3K | | Fabhalta | rep_detaili…”
- **Verdict**: _____

## 18. Why did the model flag this HCP segment - what features drove the prediction?

- rows: 1  |  pipeline said: PARALLEL_DELEGATION×1  |  judge: TOOL_COMPOSER @ 0.78
- judge rationale: Query requires sequential dependency: first retrieve prediction context, then explain features, then assess causal impact. Parallel agents cannot establish necessary context ordering.
- demo doc expectation: intent `causal_analysis` → explainer (SHAP interpretability API) (question 5.5)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
  - `heterogeneous_optimizer`: Estimates treatment-effect heterogeneity (CATE) with EconML CausalForestDML, discovers high/low-responder segments, ranks effect-modifier importance, and recommends an optimal treatment-allocation policy across segments.
  - `prediction_synthesizer`: Aggregates predictions from multiple ML models into a single ensemble forecast with model provenance, agreement score, and prediction intervals (optionally enriched with historical context).
- proposed verdict (contract-based): `single_agent:explainer`
  - reasoning: Explaining an existing prediction's drivers is explainer's contract (demo doc: SHAP interpretability path). (prediction, explanation) is also a defined PARALLEL pair — so even read as two domains it can never be TOOL_COMPOSER under the ruling.
  - empirical status: qualified — explainer fails closed without upstream artifacts to narrate (standalone chat supplies none); live answered via e2i_data_query_tool; same standalone-context caveat as q07
- empirical evidence (q18, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `explainer` (intent `explanation`): FAILED CLOSED — explainer: explainer needs structured inputs that could not be grounded in real data (no successful upstream agent results exist in this conversation state to explain, and dispatch.parameters supplies none — run an analysis first (…
  - live AG-UI response (real UI brain): tools=[e2i_data_query_tool] (22505.7ms, 3285 chars); response: “I don't have enough context yet to pull the right prediction record — "this HCP segment" could refer to any brand/region/segment, and prediction flags (and their feature attributions) are brand- and model-specific. Could you tell me: 1. **Which brand** (Kisqali, Fabhalta, or Remibrutinib)? 2. **Which HCP segment or region** was flagged (e.g., high-severity tier, a specific territory, line-of-therapy group)? 3. Roughly **when** this flag occurred …”
- **Verdict**: _____

## 19. If conversion rate in the west is below 15%, which patient segments should we prioritize?

- rows: 1  |  pipeline said: CLARIFICATION_NEEDED×1  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires multi-step reasoning: (1) validate conversion rate threshold against west region data, (2) analyze patient segments, (3) prioritize based on conditional logic. Dependent steps across analytics domains demand TOOL_COMPOSER, not clarification.
- demo doc expectation: intent `multi_faceted` → CONDITIONAL dependency detection (question 6.3)
- contract check: no candidate agent on record (pipeline dispatched none; pick from the reference above)
- proposed verdict (contract-based): `agree`
  - reasoning: Conditional composition: a KPI/gap check gates a segment-prioritization step — two domains with a hard dependency (the demo doc authored exactly this as the CONDITIONAL-dependency showcase, 6.3). Both gate conditions hold.
  - empirical status: qualified — #1350 crash on forced; live PREMISE-CHECKED instead of composing: West conversion is 62.8%, not <15%, so the composite's condition is false in data
- empirical evidence (q19, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `tool_composer` (intent `multi_faceted`): SUCCEEDED (28244.5ms); response: “Unable to complete analysis: Decomposition failed: Failed to decompose query: the JSON object must be str, bytes or bytearray, not list”
  - live AG-UI response (real UI brain): tools=[kpi_calculate_tool] (9508.4ms, 705 chars); response: “The premise doesn't hold: actual conversion rate in the **West region** (across brands, no specific brand filter) is **62.8%**, not below 15% — well above that threshold. This figure covers the engine's default reporting window (most recent 30 days of trigger data, through 2026-07-27), not any specific period you may have had in mind. Since West conversion is actually strong, there's no low-conversion trigger here to justify a segment-prioritizat…”
- **Verdict**: _____

## 20. Forecast Kisqali TRx volume for the next two quarters and tell me the biggest risk to that forecast

- rows: 1  |  pipeline said: SINGLE_AGENT×1  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires dependent steps: (1) forecast TRx volume using predictive models, (2) identify risk factors specific to that forecast. Risk analysis depends on forecast output, necessitating sequential tool composition across domains.
- demo doc expectation: intent `multi_faceted` → prediction_synthesizer + causal/drift context (question 6.5)
- contract check — candidate single agent(s):
  - `prediction_synthesizer`: Aggregates predictions from multiple ML models into a single ensemble forecast with model provenance, agreement score, and prediction intervals (optionally enriched with historical context).
- proposed verdict (contract-based): `agree`
  - reasoning: Prediction + risk-to-that-forecast (drift/causal context) with an explicit dependency on the forecast output — demo doc authored as multi_faceted (6.5). Defensible alternative: extend:prediction_synthesizer to narrate forecast risk from its own uncertainty/enrichment outputs, which would make this single-agent; empirical responses should decide.
  - empirical status: qualified — #1350 crash; live answered forecast+risk via e2i_data_query_tool (3.0k chars)
- empirical evidence (q20, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `tool_composer` (intent `multi_faceted`): SUCCEEDED (7860.0ms); response: “Unable to complete analysis: Decomposition failed: Failed to decompose query: the JSON object must be str, bytes or bytearray, not list”
  - live AG-UI response (real UI brain): tools=[e2i_data_query_tool] (21012.5ms, 2984 chars); response: “## Kisqali TRx Forecast — Next Two Quarters (Q3 & Q4 2026) **Forecast basis:** Regional prediction records for June/July 2026 (the latest forward-looking data available) show these regional TRx projections holding roughly flat month-over-month: | Region | Predicted TRx | Target | Achievement | YoY | |---|---|---|---|---| | West | 160,203 | 231,136 | 69.3% | +25.1% | | South | 203,508 | 215,586 | 94.4% | +30.0% | | Northeast | 252,475 | 269,839 | …”
- **Verdict**: _____

## 21. Our Kisqali TRx dropped in the northeast last quarter while conversion rates for Remibrutinib stayed flat, and I need to understand several things: what actually caused the Kisqali decline, whether biologic-experienced patient segments were disproportionately affected compared to biologic-naive ones, what the models predict for both brands next quarter, whether any data drift could be confounding these reads, and finally what experiment we should run to test whether adding rep capacity in the northeast would recover the trend.

- rows: 1  |  pipeline said: PARALLEL_DELEGATION×1  |  judge: TOOL_COMPOSER @ 0.85
- judge rationale: Query requires sequential dependencies: causal analysis must precede cohort comparison; drift assessment must inform prediction validity; experiment design depends on causal findings. PARALLEL_DELEGATION assumes independence; these analyses are causally linked.
- demo doc expectation: intent `multi_faceted` → complexity warning / graceful decomposition, no timeout (60+ words, 4+ domains) (question A.10)
- contract check — candidate single agent(s):
  - `causal_impact`: Estimates the average causal effect of a single treatment on an outcome using DoWhy/EconML, then validates it with refutation and sensitivity tests and renders a natural-language interpretation.
  - `cohort_profiler`: Tier-0 chat companion to cohort_constructor that answers the free-text 'size / define a cohort of ... patients' question with REAL DB-backed per-segment prescribing counts (new-Rx headline + severity tier + line-of-therapy breakdown) for one brand or all brands, never fabricating.
  - `drift_monitor`: Detects temporal/distributional drift (data, model-prediction, concept, and causal-DAG structural) by comparing a current window against a baseline window with statistical tests, then aggregates a composite score and alerts.
  - `experiment_designer`: Designs a rigorous causal experiment end-to-end BEFORE it runs — turning a business question into a full, pre-registered A/B/RCT design with power analysis, validity audit, and runnable DoWhy analysis code.
  - `explainer`: Synthesizes upstream agents' analysis_results into an audience-tailored natural-language narrative (executive summary, detailed explanation, insights, visual suggestions, follow-ups), auto-selecting LLM vs deterministic mode by input complexity; also serves as the orchestrator's universal default/fallback agent.
  - `gap_analyzer`: Detects performance gaps across segments for a brand, estimates ROI/payback to close each gap, and prioritizes the opportunities into actionable buckets (quick wins, steady plays, strategic bets) with value-destroying ones suppressed.
  - `heterogeneous_optimizer`: Estimates treatment-effect heterogeneity (CATE) with EconML CausalForestDML, discovers high/low-responder segments, ranks effect-modifier importance, and recommends an optimal treatment-allocation policy across segments.
  - `prediction_synthesizer`: Aggregates predictions from multiple ML models into a single ensemble forecast with model provenance, agreement score, and prediction intervals (optionally enriched with historical context).
- proposed verdict (contract-based): `agree`
  - reasoning: Five explicit facets across >=4 domains (causal, heterogeneity/cohort, prediction, drift, experiment design) with stated dependencies (drift confounds the reads; the experiment depends on the causal finding). The clearest genuine TOOL_COMPOSER in the batch (authored A.10).
  - empirical status: qualified — #1350 crash; live causal_analysis_tool produced the richest answer of the pass (4.6k chars, 6 causal paths) — the composite ask is genuinely answerable
- empirical evidence (q21, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `tool_composer` (intent `multi_faceted`): SUCCEEDED (22227.1ms); response: “Unable to complete analysis: Decomposition failed: Failed to decompose query: the JSON object must be str, bytes or bytearray, not list”
  - live AG-UI response (real UI brain): tools=[causal_analysis_tool] (45775.3ms, 4585 chars); response: “## Kisqali Northeast TRx Decline — Causal Drivers The causal engine surfaced 6 significant paths (confidence ≥0.5) for Kisqali TRx/market share in the Northeast, last 90 days: | Cause | Effect (via) | Effect size | Confidence | Lag | Est. business impact | |---|---|---|---|---|---| | Persistent 180d discontinuation | trx_volume (refill continuity) | +0.285 | 0.892 | 24d | $127.5K | | Formulary status | trx_volume (patient access) | +0.223 | 0.805…”
- **Verdict**: _____

## 22. Optimize resource allocation for Remibrutinib in the Northeast region — where should reps, detailing effort, and access/support investment be prioritized to grow TRx/NRx share?

- rows: 1  |  pipeline said: CLARIFICATION_NEEDED×1  |  judge: TOOL_COMPOSER @ 0.92
- judge rationale: Query requires multi-step dependent analysis: market data → competitive positioning → rep allocation strategy → access program design. These domains interconnect; not independent parallel tasks nor single-agent solvable.
- demo doc expectation: — (not one of the 51 authored questions)
- contract check: no candidate agent on record (pipeline dispatched none; pick from the reference above)
- proposed verdict (contract-based): `single_agent:resource_optimizer`
  - reasoning: Elaborated variant of the same allocation ask: multiple LEVERS (reps, detailing, access) are allocation targets within resource_optimizer's one domain, not multiple agent-capability domains.
  - empirical status: qualified — same as q13: resolver needs a structured allocation problem; live kpi_calculate_tool answered (3.6k chars)
- empirical evidence (q22, 2026-07-29 pass; full records in review/empirical_results/):
  - forced route `resource_optimizer` (intent `resource_allocation`): FAILED CLOSED — resource_optimizer: resource_optimizer needs structured inputs that could not be grounded in real data (a real allocation problem (entities with response coefficients AND a budget constraint) must be supplied; no per-entity allocation/respo…
  - live AG-UI response (real UI brain): tools=[kpi_calculate_tool] (27176.2ms, 3550 chars); response: “## Remibrutinib – Northeast Resource Allocation **Current snapshot** (Remibrutinib, Northeast, most recent 30-day reporting window, data through 2026‑07‑27): - **TRx**: 3,045 | **NRx**: 462 | **TRx Share** (of tracked Fabhalta+Kisqali+Remibrutinib portfolio): **34.1%** (status: good) - **HCP Coverage**: 93.2% (status: good) — reps are already reaching most of the target prescriber base - Supporting historical logs (Jun–Jul 2026) show TRx at 77% o…”
- **Verdict**: _____

