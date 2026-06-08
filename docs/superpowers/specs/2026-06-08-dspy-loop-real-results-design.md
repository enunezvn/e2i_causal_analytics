# DSPy Self-Improvement Loop — Real Results, No Mocks (Design Spec)

**Date:** 2026-06-08
**Author:** Claude (Opus 4.8) + owner (enunezvn)
**Status:** Approved design — pending implementation plan
**Predecessor:** `docs/reports/dspy-feedback-loop-audit-20260607.md` (audit) → PR #792 (`f6a282f6`, closed F1–F8 at the wiring level)
**Memory:** `dspy_feedback_loop_closure_20260608`, `dspy_feedback_loop_open_audit_20260607`

---

## 1. Problem

PR #792 closed the audit's findings F1–F8 **at the wiring level** — the loop compiles, the beat is scheduled, the install path runs. Verification against live source (this session) confirms that wiring is real. But the loop **does not yet produce real self-improvement**, for two reasons, and one half runs on synthetic data:

1. **Gap A — the loop can't fire on real data.** The optimizer's trainset is exclusively `source_agent='feedback_learner'` signals (`signal_store.py:106`), gated at `≥100` signals with no time-based escape below that (`dspy_integration.py:552`). But (a) the documented production endpoint `POST /feedback/learn` calls `build_feedback_learner_graph()` + `graph.ainvoke()` directly (`feedback.py:1320,1346`), **bypassing** `agent.learn()` where F5 persistence lives (`agent.py:204-213`); and (b) nothing drives `learn()` on a cadence (the `FeedbackLearnerScheduler` is never started). Each cycle yields ~1 signal, so 100 is unreachable in practice. Net: even deployed, the learner side sits at `"skipped_insufficient_signals"` forever.

2. **Gap B — recipient optimization runs on hand-authored GOLDEN SEEDS.** The beat calls `optimize_and_save_recipient(recipient)` with no `example_provider` → static seeds in `recipient_seeds.py` (`recipient_optimizer.py:124-127`), and only `experiment_monitor` is registered in `RECIPIENT_SIGNATURE_FIELDS`. The other three recipients (`explainer`, `health_score`, `resource_optimizer`) install bundles but produce none, and their `get_*_prompt` getters have **no node consume site** (audit §7 "dead"). This synthetic supervision is documented as an intentional cold-start placeholder (shard 09 "OPEN DESIGN DECISION"), but it is exactly the mock the owner wants removed.

**This spec covers Gap A + Gap B. The deploy step (Gap C) is explicitly out of scope (owner action).**

## 2. Goal & success criteria

Build correct, validated loop wiring that produces **real** self-improvement the moment real data flows — without ever shipping synthetic-derived outputs to production as if they were real.

### Reconciliation (2026-06-08 update) — synthetic data vs the "no mock" line
The premise investigation (`docs/reports/dspy-loop-disproof-20260608/`) found the loop is **starved in production**: no real user feedback and the 4 recipient agents aren't invoked (only Tier-0 ML agents run). That is an honest **operational note**, NOT a reason to avoid building. Two distinct things — only one is forbidden:

- ✅ **Synthetic data to build + validate the wiring** is legitimate and expected (golden seeds, synthesized feedback/experiments). "Faithful" (cheapest-disproof) = **real code paths — real LM, real DB, real GEPA — not stubbed**; it does *not* require real production *data*. Synthetic inputs through the real pipeline is a faithful mechanism test.
- ❌ **Shipping synthetic-derived outputs to production as if real self-improvement** is the prohibited mock. Guard: production **runs on real data and skips / serves defaults when real data is absent** — it never silently installs a synthetic-supervised prompt and presents it as "the platform learned."

**Done when** the wiring is validated end-to-end through the real LM/DB/GEPA pipeline (synthetic inputs OK) and:

- `feedback_learner` signals are persisted from the documented endpoint **and** an autonomous cadence (validated by synthesizing feedback rows).
- The trigger fires on accumulated signals at a realistic threshold (validated with synthetic volume; force only as a fallback).
- GEPA optimizes the learner phases **and** each of the four recipients; in production each recipient optimizes on **real emitted data or honestly skips** (synthetic seeds drive validation only).
- Optimized templates are installed and **genuinely consumed** by all four recipients (`experiment_monitor` already consumes; the other three are wired).
- **No `src/` production module imports the seed fixture** (guardrail test); the seed set is a relocated test-only fixture.

**Stop at:** code-complete + synthetic-validated faithful proof + merge (no-squash). **No deploy.** Real *production* self-improvement awaits real usage of the target agents (documented, tracked separately).

## 3. Non-goals (YAGNI)

- No deploy / `deploy.yml` changes (owner action).
- No LM-judge or human-feedback reward source (heuristic reward chosen — §5).
- No new optimization algorithm; reuse the existing GEPA path and `FeedbackLearnerOptimizer`.
- No rework of the prediction-outcome Loop B (healthy, independent).
- No change to the four already-working senders (`causal_impact`, `gap_analyzer`, `heterogeneous_optimizer`, `prediction_synthesizer`).

## 4. Gap A — generate real learner signals & let the loop fire

Three coordinated fixes:

### 4.1 Fix the persistence bypass
Move `persist_training_signal` into the graph's **finalize node** (`graph.py`), so **every** caller of the graph persists — the orchestrator path, the `agent.learn()` wrapper, and the API `_execute_learning_cycle` (`feedback.py:1302-1373`) which invokes the graph directly. This is more robust than patching only the API route. `agent.learn()` keeps a guard so it does not double-persist (e.g., persistence becomes the finalize node's responsibility; `learn()` stops calling `persist_training_signal` and instead trusts the node, gated by the same `persist_signals` flag threaded into the graph build).

- **Decision:** finalize-node persistence, flag-gated, single write per cycle. Unit test asserts exactly one persist per graph run and that the API path persists.

### 4.2 Add a generation cadence
New Celery beat task (e.g., `run_feedback_learning_cycle`) in `src/tasks/dspy_optimization_tasks.py` (or a sibling module), registered in `src/tasks/__init__.py` `__all__` and scheduled in `src/workers/celery_app.py` beat config + `task_routes`. It runs `agent.learn()` over a recent feedback window, reusing the `FeedbackLearnerScheduler`/`_execute_cycle` **decision logic** (cooldown, cycle config) by calling it from the worker — **not** by running the scheduler as an in-process asyncio loop. Mirrors the existing `run_feedback_loop_*` beat conventions (`run_async` helper, queue `analytics`).

- Cadence: a few times per day (final value set in plan; aligned to feedback availability so cycles aren't run over empty windows).
- Manual/operator run supported (a `force` arg, mirroring `run_dspy_prompt_optimization`).

### 4.3 Right-size the trigger threshold
`DSPY_MIN_SIGNALS` default `100` is unreachable at ~1 signal/cycle. Make it env-configurable with a realistic default (~15–25; final value justified in plan) and keep the reward-delta and forced-after-N-hours escapes. **No fabrication** — below threshold it honestly returns `"skipped"`.

## 5. Gap B — real per-recipient supervision (all 4 recipients)

A uniform per-recipient pattern, applied to `experiment_monitor`, `explainer`, `health_score`, `resource_optimizer`.

### 5.1 Reward = deterministic heuristic
Each recipient gets a deterministic heuristic metric that scores a freshly generated output for task completeness/grounding over the real signature inputs — consistent with the learner side's `compute_reward`, no LM cost, no CI flakiness (#504). Concrete per-recipient heuristics (refined in plan):

- **experiment_monitor** (`SRMDescriptionSignature`, `MonitorSummarySignature`, `AlertGenerationSignature`): output cites the test statistic + p-value + severity + a concrete recommended action; length within bound.
- **explainer** (`ExplanationSynthesisSignature`, `InsightExtractionSignature`, `NarrativeStructureSignature`): references the provided evidence/inputs; contains the required structural sections; ends with actionable framing.
- **health_score** (`HealthSummarySignature`, `HealthRecommendationSignature`): cites the dimension score(s), names the failing component(s), gives a remediation action.
- **resource_optimizer** (`OptimizationSummarySignature`, `AllocationRecommendationSignature`, `ScenarioNarrativeSignature`): cites the allocation numbers + ROI, flags any violated constraint, gives a concrete reallocation.

### 5.2 Emit (self-emission)
In each recipient's generating node, after producing output via its `get_*_prompt` getter, log a training signal to `dspy_agent_training_signals` with `source_agent=<recipient>`, capturing the **real signature inputs**, the **generated output**, and the **heuristic reward**. Reuse the existing `SignalCollectorAdapter`/`signal_store` write path.

- **No enum migration needed:** `source_agent` is `VARCHAR(50)` free text (`014_dspy_training_signals.sql:56`), so recipient values (`experiment_monitor`, etc.) write without DDL. (Verify no other constraint blocks them.)
- Emission is best-effort (never fails the agent run), mirroring the existing senders.

### 5.3 Consume (wire the dead getters)
`experiment_monitor` already consumes (`alert_generator.py:149,237,413`). Wire the other three so the optimized template is actually used at runtime — swap inline string assembly for the existing getter call, **preserving current output shape and inputs**:

| Recipient | Consume point (today) | Route through getter |
|---|---|---|
| explainer | `narrative_generator.py`, `deep_reasoner._reason_with_llm` | `get_executive_summary_prompt` / `get_narrative_section_prompt` / `get_detailed_explanation_prompt` |
| health_score | `score_composer.py:71` → `_generate_summary` (+ recommendation/issue methods) | `get_summary_prompt` / `get_recommendation_prompt` / `get_issue_description_prompt` |
| resource_optimizer | `impact_projector.py:102,105` → `_generate_summary` / `_generate_recommendations` | `get_summary_prompt` / `get_recommendation_prompt` / `get_scenario_comparison_prompt` / `get_constraint_warning_prompt` |

- **REASON-BEFORE-RULES per recipient:** before swapping, confirm each method is inert-by-omission (not deliberately bypassed for a reason — feature flag, semantic mismatch). If a recipient is dead for a deliberate reason, surface it instead of force-wiring. Align template placeholders to the inputs the node actually has.

### 5.4 Optimize on real data
- Extend `RECIPIENT_SIGNATURE_FIELDS` (`recipient_optimizer.py:28`) to all four recipients (template-field → signature name).
- Replace the golden-seed `example_provider` default with one that reads each recipient's **real emitted signals** (`source_agent=<recipient>`) from the table and builds `dspy.Example(...).with_inputs(<signature input fields>)`. The heuristic metric (§5.1) scores freshly generated outputs during GEPA.
- Wire the per-recipient metric into `optimize_recipient` (replacing/augmenting `_wrap_metric` over `get_metric_for_agent`).

### 5.5 Cold-start = skip, no fallback
If a recipient has `< N` real emitted examples (`N` set in plan, ≥2 minimum for a train/val split), the optimizer **skips** it — the recipient keeps serving its current default template. **Remove the golden-seed default from the production path** (`recipient_optimizer.py:124-127` no longer imports `recipient_seeds`). Relocate `recipient_seeds.py` to a clearly test-only fixture (e.g., under `tests/`), imported only by integration tests, never by `src/`.

## 6. Data flow (end state)

```
recipient node runs → get_*_prompt() → real output
        │  (emit) real inputs + generated output + heuristic reward
        ▼
dspy_agent_training_signals
   source_agent ∈ {causal_impact, gap_analyzer, heterogeneous_optimizer,
                   prediction_synthesizer, experiment_monitor, explainer,
                   health_score, resource_optimizer, feedback_learner}
        ▲ (generate) feedback-learning-cycle beat → agent.learn() → finalize node persists feedback_learner signal
        │
beat: run_dspy_prompt_optimization (daily)
   ├─ learner side:  get_feedback_learner_training_signals → trigger gate (realistic N) → FeedbackLearnerOptimizer.optimize() → save_optimized_module
   └─ recipient side (per recipient): read REAL emitted signals → (≥N ? GEPA optimize : skip) → materialize placeholder-safe bundle → save_prompt_bundle
        ▼
install_all_prompt_bundles()  (FastAPI startup + post-beat)
        ▼
recipients serve OPTIMIZED templates — now genuinely consumed by ALL FOUR
```

## 7. Components & boundaries

- **Generation:** `dspy_optimization_tasks.py` (new beat) + `graph.py` finalize node (persist) + `agent.py` (de-dup guard). Interface: persisted rows in `dspy_agent_training_signals`.
- **Trigger:** `GEPAOptimizationTrigger` (threshold env-config). Interface: `should_trigger(...) -> (bool, reason)`.
- **Learner optimize:** `optimization_runner.py` + `FeedbackLearnerOptimizer` (unchanged). Interface: saved optimized modules per phase.
- **Recipient emit:** small helper per recipient node (or a shared `recipient_signal_emit(...)` util) — one clear purpose: log (inputs, output, reward).
- **Recipient consume:** recipient nodes call their own `get_*_prompt` getters (no cross-agent coupling).
- **Recipient optimize:** `recipient_optimizer.py` (real `example_provider` + per-recipient metric + all-4 `RECIPIENT_SIGNATURE_FIELDS`).
- **Install:** `prompt_bundles.py` (unchanged).

## 8. Testing & faithful proof (the "real results" gate)

- **TDD, red-first** for each unit. Unit tests use inline fixtures + the relocated **test-only** seed fixture; offline, no LM — the CI arbiter.
- **Faithful disproof FIRST** (cheapest experiment, before the full Gap B build, per CLAUDE.md): generate real learner signals via `agent.learn()` on a real feedback window, force-run `run_dspy_prompt_optimization(force=True)` against **real Supabase + real Anthropic LM**, confirm the full chain including `experiment_monitor` serving a non-default template. Proves the mechanism on real data before investing in all four recipients' consume-wiring.
- **Real-LM E2E tests gated behind `E2I_RUN_REAL_LLM_E2E=1`** (#504 precedent — CI's pytest-timeout thread method can't interrupt GEPA's thread-pool LM calls; offline units stay the CI arbiter).
- **No-mock guardrail test:** assert no `src/` module imports the seed fixture (e.g., grep-style import test, mirroring `test_no_unconditional_nest_asyncio_apply`).
- **Test-pollution guard:** `learn()` persists by default and live Supabase is reachable in tests → keep/extend the autouse conftest that patches `get_supabase_client→None` for feedback_learner unit dirs (per closure memory). Recipient emit tests need the same guard.
- **Heavy verification** via the project's faithful-real + `codex:codex-rescue` convergence loop; verify codex findings against source before accepting.

## 9. Sequence

1. **Faithful disproof** on the current learner path (prove value before building Gap B).
2. **Gap A**: finalize-node persistence + de-dup guard; generation beat; threshold right-size. Re-prove the loop fires **without** `force`.
3. **Gap B — experiment_monitor first** (consume already real → fastest faithful proof): emit + real example_provider + heuristic metric; remove golden default; relocate seeds.
4. **Gap B — explainer, health_score, resource_optimizer**: emit + consume-wiring + heuristic each (REASON-BEFORE-RULES per recipient).
5. **Cleanup + honesty**: delete golden-seed production path; update `CONTRACT_VALIDATION.md` to describe the now-real loop and the realistic threshold.
6. **Merge** (no-squash, `--merge`). **No deploy** — owner enables `deploy.yml` + ensures worker/beat run.

## 10. Risks & mitigations

- **Consume-wiring changes recipient runtime output** (highest risk). Mitigation: preserve output shape; per-recipient intent check; faithful before/after comparison; staged (one recipient at a time).
- **GEPA on few real examples is weak.** Mitigation: skip below `N`; the loop strengthens as real data accumulates; faithful proof uses force to validate mechanism, not quality.
- **CI timeouts on real GEPA.** Mitigation: `E2I_RUN_REAL_LLM_E2E=1` gate; offline units arbiter.
- **Test pollution of the live signals table.** Mitigation: conftest `get_supabase_client→None` guard for all emit/learn unit dirs; clean any rows created during faithful runs.
- **Merged-main mypy ceiling / Ruff whole-tree** (recurring gotcha). Mitigation: scope-type changed files, format every touched file, watch the numeric mypy ceiling on the union.

## 11. Open items deferred to the plan (not unknowns — values to pin)

- Final `DSPY_MIN_SIGNALS` default and recipient `N`.
- Generation beat cadence + feedback-window size.
- Whether a shared `recipient_signal_emit` util vs per-node inline (lean: shared util).
- Exact placeholder/inputs alignment per recipient getter.
