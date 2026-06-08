# DSPy Self-Improvement Feedback Loop — Audit

**Date:** 2026-06-07
**Auditor:** Claude (Opus 4.8), ultracode multi-agent verification
**Scope:** The DSPy "self-improvement" loop — computational agents emit training signals → `feedback_learner` optimizes prompts via MIPROv2/GEPA → recipient agents consume optimized templates.
**Method:** Direct source reading + real code execution (no mocks) + a 6-investigator adversarial verification workflow (each agent instructed to *refute* a specific claim with real repo/test evidence) + independent synthesis. dspy 3.1.0 is installed and imports cleanly.

---

## 1. Bottom line

> **The self-improvement loop does not close in production. It is an open loop.**

Computational agents *do* compute and emit real training signals, but **no production code path ever runs the optimizer, and no recipient agent ever receives an optimized prompt.** The optimizer (`FeedbackLearnerOptimizer`), the trigger (`GEPAOptimizationTrigger`), and the scheduler (`FeedbackLearnerScheduler`) are fully implemented and unit-tested, but nothing in `src/` instantiates or starts them. Signals accumulate in two disconnected sinks (an in-memory deque and a Supabase table); neither is ever read back to drive optimization. The one agent that genuinely *calls* the prompt getters (`experiment_monitor`) only ever serves its **constructor-default** templates, because the method that would install optimized prompts (`update_optimized_prompts()`) has zero production callers.

This is presented as a **completed** capability — commits `feat(tier5): implement fixes for Tier 5 self-improvement agents` and `feat(gepa): complete GEPA migration with agent integration`, and `src/agents/feedback_learner/CONTRACT_VALIDATION.md` declares **Status: COMPLIANT** — so the gaps below are **latent defects in a feature claimed operational**, not acknowledged scaffolding. The system produces the *appearance* of self-improvement (signals collected, rewards computed, MLflow tracking, schedulers, DB tables) while never improving a single prompt.

**Importantly, this is not a "fake values" anti-pattern.** Recipient agents fall back to sensible static templates and produce honest output. The harm is (a) wasted/inert infrastructure, (b) a false belief that the platform self-improves, and (c) careful correctness work already done in this subsystem (e.g. the F-015 honest-`None` reward handling, issue #424) is moot because nothing ever consumes the reward.

---

## 2. Two things are called "feedback loop" — disambiguation

Do **not** conflate these. Only the first is this audit's subject.

| | **A. DSPy prompt-optimization loop** (this audit) | **B. Prediction-outcome feedback loop** |
|---|---|---|
| Purpose | Optimize agent *prompts* from emitted training signals via MIPROv2/GEPA | Assign ground-truth labels to predictions; detect concept drift |
| Code | `src/agents/feedback_learner/*`, `tier2_signal_router.py`, per-agent `dspy_integration.py`, `src/optimization/gepa/*` | `src/tasks/feedback_loop_tasks.py` + PL/pgSQL `run_feedback_loop()` (migration 006) |
| Trigger | **None in production** (open loop) | **Live**: Celery beat — short (4h), medium (daily 2AM), long (weekly) windows + concept-drift analysis |
| Status | Open / inert | Functioning (independent of A) |

Loop **B** is wired and scheduled (`src/workers/celery_app.py:309-328`) and does real work (calls `run_feedback_loop` RPC for `trigger`, `next_best_action`, `churn`, `market_share_impact`, `risk`; then `analyze_concept_drift_from_truth`). It touches neither DSPy, MIPROv2, nor the `feedback_learner` agent. It is healthy and out of scope; named here only to prevent misattribution.

---

## 3. Intended vs. actual data flow

**Intended (per docstrings / contract):**

```
Tier-2 / computational agents
   │  emit AgentTrainingSignal (compute_reward → scalar)
   ▼
Tier2SignalRouter ──► dspy_receiver (buffer)  ─┐
CognitiveRAG ──► memory_adapters ─► Supabase   ─┤  training data
                                                ▼
        GEPAOptimizationTrigger.should_trigger()
                                                ▼
        FeedbackLearnerOptimizer.optimize()  (MIPROv2 / GEPA)
                                                ▼
        save_optimized_module()  ──►  recipient.update_optimized_prompts()
                                                ▼
        Recipient agents serve OPTIMIZED templates
```

**Actual (verified):**

```
Tier-2 agents ──► router ──► dspy_receiver deque   ✗ never read, lost on restart
CognitiveRAG ──► memory_adapters ──► Supabase      ✗ never read back
feedback_learner.learn() ──► finalize node         ✗ signal built, never persisted
                                                    ✗ optimizer NEVER invoked
                                                    ✗ update_optimized_prompts NEVER called
Recipient agents ──► constructor-DEFAULT templates  (loop never closes)
```

Every `✗` is an independently confirmed break. **Four sequential links are severed**: persist → trigger → optimize → install.

---

## 4. The break-point map

| Stage | Intended | Actual | Status |
|---|---|---|---|
| 1. Emit signals | All sender agents emit | 4 of ~7 senders actually emit (§7) | ⚠️ partial |
| 2. Transport | router → receiver, + DB | Both paths run; fallback `_store_signals_locally` is a no-op | ⚠️ lossy |
| 3. Persist self-improver signal | `feedback_learner` writes its signal | `learn()`/finalize never persist it (F5) | ❌ broken |
| 4. Read back for training | `get_signals_for_optimization()` feeds optimizer | Zero production callers; wrong column filter (F3, F4) | ❌ broken |
| 5. Trigger | `GEPAOptimizationTrigger` + scheduler | Never instantiated; scheduler never started (F1) | ❌ broken |
| 6. Optimize | `FeedbackLearnerOptimizer.optimize()` | Never called; trainset would be empty even if it were (F6) | ❌ broken |
| 7. Install optimized prompts | `update_optimized_prompts()` | Zero production callers (F2) | ❌ broken |
| 8. Consume | Recipients serve optimized prompts | One real caller; serves defaults only (F2) | ❌ inert |

---

## 5. Findings

Severity reflects impact on the *stated* capability (self-improvement), not on current user-facing output (which is honest fallback).

### F1 — Optimizer is never triggered in production · **CRITICAL** · CONFIRMED
`FeedbackLearnerOptimizer`, `GEPAOptimizationTrigger`, and `FeedbackLearnerScheduler` are never instantiated/started by any production code.
- `FeedbackLearnerScheduler._execute_cycle()` calls `self._agent.learn()` only — never the optimizer (`scheduler.py:329-421`).
- `agent.learn()` runs `self.graph.ainvoke()` only (`agent.py:136-219`); the LangGraph DAG is `[audit_init, enrich, collect, analyze, rubric, extract, update, finalize]` — **no optimize node** (`graph.py:36-144`).
- `FeedbackLearnerOptimizer` (def `dspy_integration.py:606-752`) and `GEPAOptimizationTrigger` (def `:455-565`) have zero instantiations outside their own module + tests.
- `src/api/main.py` lifespan initializes Redis/Supabase/MLflow/Feast/Opik/OTel but never references the scheduler; `create_scheduler`/`scheduler.start()` appear only in a docstring example (`scheduler.py:109`).
- The parallel `ChatbotOptimizer` in `chatbot_dspy.py:3025+` is *also* orphaned (its `get_chatbot_optimizer()` has no `@router` decoration and no schedule).
- **Intent:** roadmapped-but-unwired build-out, not intentional scaffold (the trigger/scheduler are fully written and documented as the intended mechanism). **Keystone fix — nothing else in the loop matters without it.**
- **Fix:** Start `FeedbackLearnerScheduler` in `main.py` lifespan, **or** add a Celery beat task that periodically calls `should_trigger()` → `optimize()` and persists the result.

### F2 — Optimized prompts are never installed into recipients · **CRITICAL** · CONFIRMED
`update_optimized_prompts()` has **zero** production callers (only test callers). The one agent that genuinely calls the prompt getters, `experiment_monitor` (`nodes/alert_generator.py:145-242` → `get_srm_prompt`/`get_enrollment_prompt`/`get_fidelity_prompt`), therefore always returns its **constructor defaults** (`dspy_integration.py:80-82`: `version="1.0"`, `last_optimized=""`, `optimization_score=0.0`).
- **Intent:** real defect — the recipient "write side" was designed (`"Update prompts with optimized versions from feedback_learner"`) but never connected to F1's output.
- **Fix:** After F1 produces an optimized module, persist it and call each recipient's `update_optimized_prompts()` at startup / on refresh. Until then, label recipient prompt infra as *default-only* so it is not mistaken for active self-improvement.

### F3 — Persisted signals are never read back · **CRITICAL** · CONFIRMED
`memory_adapters.get_signals_for_optimization()` (def `memory_adapters.py:792`) has zero production callers (tests only). In-memory `dspy_receiver`/`tier2` deques are never drained into an optimizer. Migration `033_drop_orphan_dspy_tables.sql:18-21` even labels `:814` as the table "reader" — aspirational; it has never been wired.
- **Fix:** Have the F1 trigger call `get_signals_for_optimization()` (after F4) to assemble the trainset, then feed `FeedbackLearnerOptimizer.optimize()`.

### F4 — Reader filters a non-existent column · **HIGH** · CONFIRMED
`get_signals_for_optimization()` filters `.eq("signal_type", signal_type)` (`memory_adapters.py:814-819`), but `dspy_agent_training_signals` has **no `signal_type` column** — it has `source_agent` (`database/memory/014_dspy_training_signals.sql:56`). The writer correctly maps `source_agent = s.signal_type` (`:751`). A *filtered* read targets a missing column (PostgREST error/empty); only the unfiltered (`signal_type=None`) path works.
- **Intent:** a real copy-paste bug (Python field name pasted into a DB filter), latent only because F3 means the method never runs.
- **Fix:** `.eq("source_agent", signal_type)`; add a test exercising the filtered path against the real schema.

### F5 — feedback_learner never persists its own signal · **HIGH** · CONFIRMED
The finalize node builds a `FeedbackLearnerTrainingSignal` and calls `compute_reward()` (`graph.py:227-303`), but stores it only in `state["training_signal"]`; `agent.py:186-192` returns it as `FeedbackLearnerOutput.training_reward` with **no DB insert**. The *only* writer to the table is the separate CognitiveRAG path (`cognitive_rag_dspy.py:616-617` → `memory_adapters.flush()`).
- **Fix:** In the finalize node, persist `training_signal.to_dict()` to `dspy_agent_training_signals` so the self-improver's own signals are durable and readable by F1.

### F6 — Even if invoked, the trainset is empty / single-phase · **HIGH** · PARTIAL
**Proven by real execution** (no mocks):
- `to_dict()` `input_context` stores `batch_id/feedback_count/time_range/focus_agents/has_cognitive_context` but **not** `feedback_batch` (`dspy_integration.py:259-306`).
- `_signals_to_examples` reads `input_context.get("feedback_batch", [])` (`:910-911`) → **always `[]`**; and only `phase=="pattern"` builds an example (`:908-919`) — `recommendation`/`update`/`summary` yield **0** examples → short-circuit on the `len < 5` guard.

```
$ .venv/bin/python  # real run against to_dict()-produced signals
input_context keys: ['batch_id','feedback_count','focus_agents','has_cognitive_context','time_range_end','time_range_start']
has 'feedback_batch' key: False
phase=pattern        -> 10 examples; first.feedback_batch='[]'   # empty input
phase=recommendation ->  0 examples
phase=update         ->  0 examples
phase=summary        ->  0 examples
```

- **PARTIAL because** the MIPROv2/GEPA **API calls are valid for dspy 3.1.0** — verified by `inspect.signature`: `MIPROv2(metric, num_candidates, max_bootstrapped_demos, num_threads)` + `.compile(student, trainset, num_trials)` and `GEPA.compile(student, *, trainset, valset=None)` all match. The code would not crash on API mismatch. One latent quality bug: `valset` is built (`:781-782`) but **not passed** to `compile()` (`:817`) → GEPA would validate on the trainset (overfit), not raise.
- **Fix (sequence AFTER F1/F3/F5):** add `feedback_batch` to `to_dict()`; implement/guard the non-`pattern` phases; pass `valset` to `compile()`.

### F7 — Tier-2 local-store fallback is a silent no-op · **MEDIUM** · CONFIRMED
`Tier2SignalRouter._store_signals_locally()` (`tier2_signal_router.py:209-250`) evaluates `entry["signal"]` as a **bare expression** (`:220`) then discards it; each branch only fetches a collector reference (e.g. `get_causal_impact_signal_collector()`) and throws it away, while logging the misleading `"Signals will be stored locally for later retrieval."` (`:199`). On the import-failure fallback path, signals are silently lost.
- **Fix:** actually append `entry["signal"]` to the collector buffer, or delete the dead expression and the misleading log.

### F8 — Compute-and-discard dead expressions · **LOW** · CONFIRMED
`graph.py:256-257` compute efficiency/coverage targets as **bare statements with no assignment**. Harmless: `compute_reward()` (`dspy_integration.py:167-257`) recomputes both internally and is **verified responsive** by real execution (15s/5-patterns → 0.80 vs 30s/5-patterns → 0.725; `pattern_accuracy=None` correctly skipped per issue #424, not fabricated as 0.0). The discarded lines even use a *different* formula (5000 ms target vs 3.33 items/s), confirming they were never wired to reward.
- **Fix:** delete `graph.py:256-257`; optionally enable ruff `B018` to prevent recurrence.

---

## 6. What *is* real (don't break these)

- **Reward math** (`compute_reward`) is pure, deterministic, responsive, and honestly handles missing ground truth (`pattern_accuracy=None` skip + weight redistribution, issue #424 / F-015). Good.
- **Signal emission** by `causal_impact`, `gap_analyzer`, `heterogeneous_optimizer`, `prediction_synthesizer` is real production code.
- **DB write path** (`SignalCollectorAdapter.collect()/flush()` from CognitiveRAG) genuinely persists to `dspy_agent_training_signals` (survives restart) — it's just never read back.
- **MIPROv2/GEPA API usage** is correct for the installed dspy 3.1.0.
- **Prediction-outcome loop B** (Celery) is wired and functioning.

---

## 7. Per-agent consumption matrix

13 agents define `dspy_integration.py`. "Emits/consumes in real execution" = called from the agent's own graph nodes / `agent.py`, not just defined/exported.

| Agent | Declared role | Defines optimized prompts | Real emit/consume | Evidence |
|---|---|---|---|---|
| causal_impact | sender | – | ✅ emits | `nodes/interpretation.py:86` → `collect_analysis_signal` |
| gap_analyzer | sender | – | ✅ emits | `nodes/formatter.py:91` → `_collect_dspy_signal` |
| heterogeneous_optimizer | sender | – | ✅ emits | `nodes/profile_generator.py` → `collect_optimization_signal` |
| prediction_synthesizer | sender | – | ✅ emits | `agent.py` → `collect_and_emit_signal` (`dspy_integration.py:414-483`) |
| **drift_monitor** | sender | – | ❌ never emits | collector defined; **no call site** in `nodes/`/`agent.py` |
| **experiment_designer** | sender | – | ❌ never emits | collector defined; **no call site** in `nodes/`/`agent.py` |
| **experiment_monitor** | recipient | ✅ | ⚠️ calls getters but **defaults only** | `nodes/alert_generator.py:145-242`; `update_optimized_prompts` never called → `last_optimized=""` |
| **explainer** | recipient | ✅ | ❌ dead | no call site to `get_explainer_dspy_integration()` |
| **health_score** | recipient | ✅ | ❌ dead | getters defined; no call site in `nodes/`/`agent.py` |
| **resource_optimizer** | recipient | ✅ | ❌ dead | getters defined; no call site in `nodes/`/`agent.py` |
| tool_composer | hybrid | – | ⚠️ infra only | `get_optimized_*_prompt()`/`request_optimization()` defined, no call site |
| orchestrator | hub | – | ⚠️ infra only | `request_optimization()` (`dspy_integration.py:290-326`) never invoked |
| feedback_learner | self-improver | – | ⚠️ analysis only | runs graph; `optimize()` never called (F1) |

**Net:** ~60% of the per-agent DSPy infrastructure is defined-but-unused. Even the single live consumer (`experiment_monitor`) is inert because no optimization ever updates its templates.

---

## 8. Remediation roadmap (sequenced)

The links must be repaired **in dependency order** — fixing downstream bugs first is wasted effort while the loop is open.

1. **F1 (keystone):** add a production trigger (lifespan scheduler or Celery beat) → `should_trigger()` → `optimize()`.
2. **F5 + F3 + F4:** persist `feedback_learner`'s signal; wire `get_signals_for_optimization()` as the trainset source; fix the `source_agent` filter.
3. **F6:** serialize `feedback_batch`; implement/guard non-`pattern` phases; pass `valset` to `compile()`.
4. **F2:** persist optimized modules and call `update_optimized_prompts()` on recipients (start with `experiment_monitor`, the one already wired to consume).
5. **F7, F8:** fix the no-op fallback and delete dead expressions (cleanup; low risk).
6. **Honesty:** until F1–F2 land, update `CONTRACT_VALIDATION.md` / docstrings so the loop is described as **collection-only / optimization-not-yet-wired**, not COMPLIANT/operational. Track the build-out under a t*racking issue (analogous to the existing #424/#426 honest-labeling work).

Cheapest disproof before building: F1 is a ~1-line reachability check (`grep` confirmed zero callers) — validated. The single highest-leverage experiment to confirm the loop can ever produce value is to **manually** call `optimize()` once on real persisted signals after F5/F6, before investing in scheduling.

---

## 9. Methodology & confidence

- **Direct reading** of every core module (`dspy_integration.py`, `dspy_receiver.py`, `graph.py`, `scheduler.py`, `agent.py`, `tier2_signal_router.py`, `memory_adapters.py`, `feedback_loop_tasks.py`, one+ recipient).
- **Real execution** (no mocks): signal→example conversion (F6), `compute_reward` responsiveness (F8), dspy 3.1.0 API introspection (F6), DSPy/MIPROv2/GEPA import checks.
- **Adversarial verification workflow:** 6 independent investigators, each tasked to *refute* a specific claim across the whole repo (src/, scripts/, config/, Celery beat, API routes, app lifespan, dynamic dispatch) + a synthesis pass that reconciled the one apparent contradiction (experiment_monitor "consumes" vs "loop is open" → both true: consumes *defaults*).
- **Schema verification** against migration `014` (table columns) and `033` (table still LIVE).
- **Intent investigation** (REASON-BEFORE-RULES): git history, `CONTRACT_VALIDATION.md`, issue refs (#424/#426). Findings classified defect-vs-placeholder accordingly.

**Confidence:** HIGH on F1–F5, F7, F8 (consistent across direct evidence + adversarial agents + real runs); F6 HIGH on the data-pipeline break, with the API-correctness sub-claim verified by introspection (hence PARTIAL: code wouldn't crash, but trainset would be empty).

**Residual uncertainty:** none of the verdicts depend on prod runtime state. If a non-`src/` orchestrator (external cron, separate service, notebook) calls `optimize()` out-of-band, F1 would soften — but no such caller exists in this repo, and the migration-033 "reader" annotation suggests the team itself believed the reader was wired when it was not.

---

## 10. Remediation status (2026-06-08)

Closed by the `dspy-feedback-loop-closure` plan (`.claude/plans/dspy-feedback-loop-closure/`), shards 01–09:

| Finding | Shard | Resolution |
|---|---|---|
| F8 dead compute-discard exprs | 01 | deleted |
| F7 `_store_signals_locally` silent no-op | 01 | drops + logs honestly; misleading "stored locally" log removed |
| F5 learner signal never persisted | 02 | `learn()` persists via `signal_store.persist_training_signal` (best-effort) |
| F4 reader filters phantom `signal_type` | 03 | filters real `source_agent` column |
| F3 signals never read back | 03+05+08 | `get_feedback_learner_training_signals` → orchestrator → beat task |
| F6 empty/single-phase conversion | 04 | carried content + real pattern/recommendation/summary examples; `valset` passed |
| `save_optimized_module` await/kwarg mismatch | 05 | removed; saving moved to the orchestrator (sync, correct kwargs) |
| F1 optimizer/trigger/scheduler never invoked | 05+06+08 | orchestrator + `PatternAnalyzerNode` consume + daily Celery beat |
| F2 `update_optimized_prompts` never called | 07 (install) + 09 (produce) | `prompt_bundles` install path wired into app startup + beat task |

**Latent bugs found by the faithful real-LM run (cheapest disproof) — the audit's
introspection-only check could not see these because the optimizer was never
invoked:** GEPA `budget=`→`auto=` kwarg (would `TypeError`); `FeedbackLearnerGEPAMetric`
returned a plain dict, crashing GEPA's valset `dspy.Evaluate` with `int + dict`,
and read output fields that never existed on the signatures; the LM was not
propagated to GEPA's worker threads (fixed with `set_lm` on the student module);
and `save_optimized_module` extracted instructions from `extended_signature`
(dspy<3) yielding an empty `instruction_hash` on dspy 3.x. All fixed; a bounded
real Sonnet GEPA run scores real predictions and saves a load-round-tripping
artifact.

**Deploy note:** this is code-complete and wired for active production (daily
beat, auto-install). Per project state `deploy.yml` is `disabled_manually`, so
going live is the owner's deploy action; the worker + beat services must be
running. The per-recipient bundle *producer* (Shard 09) defaults to a golden
seed set — see that shard's open design decision before treating recipient F2 as
fully closed on real supervision.
