# DSPy Loop — Faithful Disproof: Evidence & Decision Gate

**Date:** 2026-06-08
**Method:** read-only source + live DB (`docker exec supabase-db psql`) SELECTs. No mocks, no LM spend, no writes.
**Companion:** `PREMISE.md` (feedback/signal inventory).

---

## Verdict: STOP-AND-RETHINK — the loop is correctly wired but STARVED of real fuel.

The audit's F1–F8 are genuinely closed (PR #792); the plumbing is real. But the DSPy
self-improvement loop is **inert because the agents it optimizes are not being exercised**, and
there is **no real feedback**. Building Gap A or Gap B now would optimize on synthetic data — i.e.
a mock — which is exactly what the owner ruled out. Real results require real usage of the target
agents FIRST.

## The link-by-link evidence

| Link | Finding | Source |
|---|---|---|
| Feedback input | No production store implements `get_feedback()`; collector returns `[]` in prod | `feedback_collector.py:98` |
| Feedback data | `feedback_items`=0, `validation_outcomes`=0, `chatbot_message_feedback`=2 (stale Jan-2026) | live DB |
| Learner signals | **0** `feedback_learner` rows; table holds only `response`=189 (synthetic `query_0…`) + `mipro_test`=55; latest Jan-2026 | `dspy_agent_training_signals` |
| Recipient runtime | **0** real records for all 4 recipients (`experiment_monitor`=0 everywhere; explainer/health_score/resource_optimizer only in Nov-2025 **seed**) | `episodic_memories`, `audit_chain_entries`, `agent_activities` |
| Sender runtime | senders absent from real logs except `causal_impact`×2 (Jun-7); `agent_activities` rows are all one seed timestamp `2025-11-28 15:39:47` | same |
| What IS running | Tier-0 ML-pipeline agents only: `model_selector` (62), `model_trainer` (10), `data_preparer`, `scope_definer`, `feature_analyzer`, `corpus_ingestion`, `observability_connector` — latest **today** | `episodic_memories` |

## Interpretation

- The platform's recent real activity is **Tier-0 model-building** (the disc/HCP cohort work), driven
  by scripts — **not** the conversational/analytics agents (senders + recipients) the DSPy
  prompt-optimization loop targets.
- `agent_activities` is a **seeded** table (one identical insert timestamp across all agents), not a
  live activity log; do not mistake it for traffic.
- Therefore **neither** loop half has organic fuel:
  - **Gap A (learner):** needs real user feedback → none exists, and the capture path isn't wired.
  - **Gap B (recipients):** needs real recipient runtime outputs to self-emit from → those agents
    aren't invoked.

## Why no real-LM harness run
The cheap read-only evidence is already decisive (no fuel on either side). Running `learn()`×6 → GEPA
on empty feedback would only confirm degeneracy at real-LM cost. Per CLAUDE.md cheapest-disproof-first,
stop at the cheapest decisive evidence.

## Options surfaced to the owner (next move)
1. **Generate real workloads** for the loop's target agents (exercise the 4 recipients on the real
   `ml_experiments`/causal results; capture real feedback) so the loop has genuine fuel — then optimize.
2. **Document inert-by-lack-of-usage** and gate the build until the platform has real usage of those
   agents / a real feedback-capture path.
3. **Wire a real feedback-capture path** (UI thumbs / Loop-B outcome labels → `feedback_items`) as the
   true upstream prerequisite, independent of the optimizer.
4. **Reframe** the loop toward the agents that ARE running (Tier-0 pipeline) — large redesign; likely
   out of the loop's intent.

Owner selected (2026-06-08): verify traffic/feedback first → done; this verdict is the result.
