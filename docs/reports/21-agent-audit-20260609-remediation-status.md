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
| F1 | CRIT | health_score | LIVE (route:789 + provenance-on-import + F1b mock endpoints) | ⏳ pending | — |
| F2 | HIGH | gap_analyzer | LIVE (formatter:98 always "completed"; nodes ignore `errors`) | ⏳ pending | — |
| F6 | HIGH | tool_composer | LIVE (success=True on 0/N tools) | ⏳ pending | — |
| F8 | HIGH | feature_analyzer | LIVE (np.random SHAP bg unlabeled) | ⏳ pending | — |
| F3 | HIGH | observability_connector | LIVE (`client=` kwarg → mock spans while 5313 real) | ⏳ pending | — |
| F4 | HIGH | model_deployer | LIVE (simulated→success=True, 0 rows) | ⏳ pending | — |
| F7 | HIGH | experiment_monitor | LIVE on main; **fix WIP in `.worktrees/fix-h1-h3b`** (await async client) | 🔧 wip | — |
| F12 | MED | heterogeneous_optimizer | LIVE (no input bridge → dead via chat, fail-closed) | ⏳ pending | — |
| F13 | MED | resource_optimizer | LIVE (same) | ⏳ pending | — |
| F14 | MED | prediction_synthesizer | LIVE (same) | ⏳ pending | — |
| F9 | MED | model_selector | LIVE (`execute_query` missing → 40% frozen) | ⏳ pending | — |
| F10 | MED | experiment_designer | LIVE (MockKnowledgeStore unmarked) | ⏳ pending | — |
| F11 | MED | drift_monitor | LIVE (structural_drift dropped) | ⏳ pending | — |
| F15 | MED | feedback_learner | LIVE (empty stores → 0.0 effectiveness) | ⏳ pending | — |
| F16 | MED | data_preparer | LIVE (`DataFrame.append` removed pandas 2.x) | ⏳ pending | — |
| F17 | MED | cohort_constructor | LIVE (bad top-level import) | ⏳ pending | — |
| F18 | LOW | causal_impact | LIVE but fail-closed-CORRECT (chat-completeness only) | ◻ optional | — |
| F19 | LOW | model_trainer | LIVE (no in-agent thread cap → 5.9 GB) | ⏳ pending | — |

**Security (separate workstream):** §6 RLS — 73 prod tables with RLS disabled.
Decide per-table policies; do not blindly enable. Not auto-actioned here.

_Statuses updated as PRs land._
