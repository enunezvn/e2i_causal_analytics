# 21-Agent Audit — Remediation Status Tracker

Living tracker for executing the audit's §7 prioritized remediation. The audit
(`21-agent-audit-20260609.md`) is a point-in-time snapshot; this file tracks what
has been **re-verified against current main** and **fixed** since.

**Method per finding:** worktree isolation → TDD red-first → real (no-mock) fix →
scoped tests (memory-guarded) → `codex:codex-rescue` to fixed point → PR. Merges +
a single deploy are **batched at the very end** (OOM-prone box; deploy held during
remediation).

**Re-verification (2026-06-09, read-only fan-out vs current main HEAD `3253790e`):**
all 18 open findings **LIVE-CONFIRMED**; F5 already resolved on main. Raw verdicts +
current fix locations in `21-agent-audit-20260609-repro/reverify_results.json`.

| Finding | Sev | Agent | Re-verified | Status | PR |
|---------|-----|-------|-------------|--------|----|
| F5 | HIGH | orchestrator | RESOLVED by #814/#816/#818 (`allow_mock=False`, fails closed) | ✅ DONE | #814 |
| F1 | CRIT | health_score | LIVE (route:789 + provenance-on-import + F1b mock endpoints) | ✅ PR | #823 |
| F2 | HIGH | gap_analyzer | LIVE (formatter:98 always "completed"; nodes ignore `errors`) | ✅ PR | #824 |
| F6 | HIGH | tool_composer | LIVE (success=True on 0/N tools) → fail-closed gate + route | ✅ PR | #827 |
| F8 | HIGH | feature_analyzer | LIVE (np.random SHAP bg unlabeled) → fail-closed + provenance e2e | ✅ PR | #828 |
| F3 | HIGH | observability_connector | RESOLVED by #826 (`client=`→`supabase_client=` at agent.py:91 + metrics_aggregator.py:30; async repo reaches 5313 real spans, proven by real-DB test) | ✅ DONE | #826 |
| F4 | HIGH | model_deployer | LIVE (simulated→success=True, 0 rows) → fail-closed flags + honest db_persisted; persistence rewire deferred to #829 | ✅ PR | #830 |
| F7 | HIGH | experiment_monitor | RESOLVED by #820 (4 nodes await get_async_supabase_client) | ✅ DONE | #820 |
| F12 | MED | heterogeneous_optimizer | LIVE (no input bridge → dead via chat, fail-closed) | ⏳ pending | — |
| F13 | MED | resource_optimizer | LIVE (same) | ⏳ pending | — |
| F14 | MED | prediction_synthesizer | LIVE (same) | ⏳ pending | — |
| F9 | MED | model_selector | LIVE (`execute_query` missing → 40% frozen) → real PostgREST queries + unmask | ✅ PR | #831 |
| F10 | MED | experiment_designer | LIVE (MockKnowledgeStore unmarked) | ⏳ pending | — |
| F11 | MED | drift_monitor | LIVE (structural_drift dropped) → wired end-to-end: input DAG fields threaded + aggregator/summary/recs/output/memory include structural | ✅ PR | #832 |
| F15 | MED | feedback_learner | LIVE (empty stores → 0.0 effectiveness) | ⏳ pending | — |
| F16 | MED | data_preparer | LIVE (`DataFrame.append` removed pandas 2.x) | ⏳ pending | — |
| F17 | MED | cohort_constructor | LIVE (bad top-level import) | ⏳ pending | — |
| F18 | LOW | causal_impact | LIVE but fail-closed-CORRECT (chat-completeness only) | ◻ optional | — |
| F19 | LOW | model_trainer | LIVE (no in-agent thread cap → 5.9 GB) | ⏳ pending | — |

**Security (separate workstream):** §6 RLS — 73 prod tables with RLS disabled.
Decide per-table policies; do not blindly enable. Not auto-actioned here.

## Follow-ups discovered during remediation

- **Orchestrator dispatch envelope masks domain-level failure** (found in F6 codex
  review, gpt-5.5). `dispatcher.py:537-540` wraps every normal agent return in
  `AgentResult(success=True)`; `synthesizer.py:66-68` filters on that outer flag, so a
  domain-failed agent (incl. a 0/N tool_composer) is counted successful
  (`chatbot_graph.py:951-974`). The honest response text + 0.0 confidence still flow
  through `_extract_response`, so the *fabricated-answer* harm is closed by F6 (#827);
  the residual is **transport-`success` vs domain-`success` accounting**, a general
  behavior across ALL agents. Needs its own scoped PR (broad dispatcher/synthesizer
  change, high ripple — cf. #814); NOT bundled into F6. (Could not file as a GitHub
  issue — external-write blocked under the loop authorization.)
- **SHAP provenance not persisted / not consumed downstream** (found in F8 codex review,
  gpt-5.5). F8 (#828) propagates `data_provenance` through the agent output contract and
  stops persisting skipped runs, but: (a) a `data_provenance` COLUMN on `ml_shap_analyses`
  needs a migration (deferred under no-migration constraint); (b) `causal_ranker`,
  `mlflow_tracker`, `memory_hooks` read/store importances without checking provenance.
  Bounded today because `allow_synthetic_background` is NOT threaded through `agent.run`
  (the agent path is real-or-skip, never synthetic). Own scoped follow-up.

_Statuses updated as PRs land._
