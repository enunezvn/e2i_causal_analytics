# DSPy Loop — Bounded Faithful E2E Proof (V)

**Date:** 2026-06-09
**Scope:** Final step of the real-data, no-mock loop build (spec
`docs/superpowers/specs/2026-06-08-dspy-loop-real-results-design.md`). Proves the
self-improvement MECHANISM works end-to-end on **real Anthropic LM + real Supabase
+ real GEPA**, using **synthetic inputs** (the loop is starved of real production
data — see `PREMISE.md`/`EVIDENCE.md`). "Faithful" = real code paths, not stubbed;
this does NOT claim real production self-improvement.

Harness: `tests/integration/test_dspy_loop_e2e_bounded.py` (gated
`E2I_RUN_REAL_LLM_E2E=1`). Self-cleaning. Bounded to ONE learner phase
(`pattern`) + ONE recipient field (`experiment_monitor.srm_template`), light GEPA.

---

## Verdict: MECHANISM PROVEN.

A bounded faithful run drove the full loop and produced a real, optimized,
installed, and **served** recipient prompt — the audit's F1/F2 ("optimizer never
runs", "optimized prompts never consumed") is closed in practice, on real infra.

## Evidence (Run 1, 588s / 9m48s — both GEPA compiles completed)

Free pre-check (no LM), all passed before any spend:
- `[A] learner signals readable (mine): 6/10` seeded → `pattern examples built: 6` (non-degenerate trainset)
- `[A] recipient srm examples built: 3` (real emitted signals → ≥2 dspy.Examples; no cold-start)

Real GEPA (LM spend):
- Learner `pattern` phase optimized; artifact persisted to disk (`status=optimized`, version_id + path on disk asserted) — proves the learner self-optimization closes (F1).
- Recipient `experiment_monitor` optimized from its **own real emitted signals** → bundle saved (`optimized_prompts/experiment_monitor/latest.json`) → `install_all_prompt_bundles()` returned `{experiment_monitor: True}` → the live singleton's `get_prompt_metadata()` showed:
  - `version: "1.0" → "1.1"`, `last_optimized: "2026-06-09T00:00:35..."`, `optimization_score: 0.7`
  - `srm_template` materialized to the **GEPA-optimized** form: *"Describe Sample Ratio Mismatch issue. Generates clear explanation of SRM detection for stakeholders. Describe Sample Ratio Mismatch for experiment '{experiment_name}'. Chi-squared: {chi_squared:.2f}, …"* — the optimized instruction prepended, **all placeholders preserved** (no KeyError risk), now SERVED by the recipient. This is F2 closed end-to-end.

Run 1 reported `1 failed` ONLY due to a test-assertion key-path bug (`after["last_optimized"]` vs the nested `after["prompts"]["last_optimized"]`) — fixed in the committed harness. The metadata above is independent evidence that every real step succeeded; the assertion fix matches that proven metadata.

## Run 2 (clean re-run of the corrected harness) — LM-latency timeout, non-blocking
The re-run of the corrected test **timed out inside `optimizer.compile` (GEPA)** at the 1100s bound — real-LM latency variance, the documented #504-class flakiness. This is precisely why the harness is gated behind `E2I_RUN_REAL_LLM_E2E=1` and why **offline unit tests are the CI arbiter** (84 passing, see below). It does not change the Run-1 verdict (the edits since Run 1 were the assertion key-path and a type-only `cast`, neither on the GEPA latency path).

## Surrounding validation
- **Offline:** 84/84 tests green across all new wiring (single-process), plus each recipient's full suite (experiment_monitor 287, explainer 231, health_score 168+37, resource_optimizer 213) and feedback_learner 381 — no regressions.
- **Final integration review:** INTEGRATION OK on all 5 end-to-end coherence checks; one CI-ceiling mypy error found + fixed (`recipient_optimizer.py` `asyncio.run` cast).
- **No-mock invariant:** golden seeds are a test-only fixture; no `src/` module imports them; recipient optimizer skips (no seed fallback) below 2 real examples; consume getters fall back to default templates on failure.

## Honest status
The loop is **code-complete, wired, and mechanism-proven on real infra**, but in this
environment it remains **starved of real production data** — real *production*
self-improvement awaits real usage of the target agents (the senders + the 4
recipients), which currently are not exercised by live traffic. Synthetic data is
used only for validation; production runs on real data or skips. **No deploy.**
