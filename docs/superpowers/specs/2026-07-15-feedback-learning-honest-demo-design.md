# Feedback-Learning Honest Demo — Design Spec

**Date:** 2026-07-15
**Status:** Approved (Approach A)
**Page:** https://eznomics.site/feedback-learning (Tier 5 self-improvement dashboard)
**Related:** issue #1240 (copilot signal wiring follow-up, out of scope here)

## Problem

The /feedback-learning page renders empty — no batches, patterns, or knowledge
updates — so the Tier 5 self-improvement loop cannot be demoed. The premise
"no learning has accumulated" is **false**: 123 real graded reward signals
exist in `learning_signals` (2026-06-14 → 2026-07-07, `is_synthetic=false`,
`domain_signal='dspy_signal'`). The page is empty for three structural reasons:

1. **Window starvation.** All 28 recorded learning batches failed with
   "No feedback items collected" because their lookback windows (backend
   default 24 h; frontend quick-cycle 7 d) postdate the last real signal.
   Signals stopped accruing 2026-07-07 because only the API-only
   `POST /api/copilotkit/chat` path (`run_chatbot`,
   `src/api/routes/chatbot_graph.py:1957`) runs the cognitive pipeline whose
   `SignalCollector` writes `learning_signals`; the frontend CopilotKit UI
   path does not (tracked in #1240).
2. **No cycle has ever run over the window that contains the real signals.**
3. **Sparse-pattern material.** Real rewards are high (agent 0.805,
   investigator 0.907, summarizer 0.916 avg; only 3/123 below 0.5), so even a
   correctly-windowed cycle will honestly report "agents performing well" with
   few patterns — thin material for demoing the Patterns/Updates tabs.

**Hard constraint (honesty guardrail, unchanged):** the learner must never
consume synthetic showcase rows. `LearningSignalsFeedbackStore`
(`src/repositories/learning_signals_feedback.py`) filters
`is_synthetic=false` — this stays exactly as is. The 1,800 seeded synthetic
signals remain excluded because knowledge updates apply to **real** agent
knowledge stores; learning from fiction would modify real agents.

## Latent bug in scope: `auto_apply` is silently ignored

`RunLearningRequest.auto_apply` (`src/api/routes/feedback.py:192`, default
`False`; the UI sends `false`) is never threaded into graph state.
`KnowledgeUpdaterNode.execute` (`src/agents/feedback_learner/nodes/knowledge_updater.py:50`)
unconditionally applies every proposed update to the real knowledge stores.
The UI promises proposed-then-Apply semantics (it has Apply/Rollback buttons
and sends `auto_apply: false`) but the backend auto-applies everything. This
must be fixed **before** any cycle is run as part of this work.

## Design (Approach A — four workstreams)

### W1 — Thread the `auto_apply` gate (backend, prerequisite)

Thread `auto_apply` from the request into the graph and honor it at the
apply site:

- `FeedbackLearnerState` (`src/agents/feedback_learner/state.py:162`): add
  `auto_apply: NotRequired[bool]` field.
- `_execute_learning_cycle` (`src/api/routes/feedback.py:1392`): include
  `"auto_apply": request.auto_apply` in `initial_state` — the single state
  construction site. Audit the other callers that build the request object
  (batch-submit path near line 713, scheduled path near line 1713) so each
  passes an explicit `auto_apply`, defaulting to `False`.
- `KnowledgeUpdaterNode.execute`: when `state.get("auto_apply")` is not
  `True`, skip the `_apply_update` loop — `proposed_updates` still populates,
  `applied_updates` stays empty, and the summary reports
  "N updates proposed, awaiting manual apply" instead of implying failure.
  The existing `update_backend_wired` honesty field is unaffected (stores are
  wired; applying was withheld by request).
- Default is `False` everywhere: fail-closed. Cycles propose; a human applies
  via the page's existing Apply endpoint (which also keeps Rollback
  meaningful).

### W2 — Backfill cycle over real history (operation, no code)

After W1 deploys, run one real learning cycle on prod:

- `POST /api/feedback/learn?async_mode=false` with operator JWT (GoTrue
  password grant: `POST $SUPABASE_URL/auth/v1/token?grant_type=password` with
  anon key + `E2I_ADMIN_EMAIL/PASSWORD`).
- Body: `time_range_start=2026-06-14T00:00:00Z`,
  `time_range_end=2026-07-08T00:00:00Z`, `min_feedback_count=5`,
  `pattern_threshold=0.1`, `auto_apply=false`.
- Sync mode persists artifacts via `_persist_cycle_artifacts` into
  `feedback_learning_batches` / `feedback_patterns` /
  `feedback_knowledge_updates` — the tables the page reads.
- Expected honest outcome: batch processes ~123 items + 3 thumbs; rewards are
  high, so likely few patterns ("system healthy" is the true story). All
  numbers real.

### W3 — Golden-set replay (script + operation)

Generate fresh, genuine signals by playing the 30 curated RAGAS golden QA
samples through the real chat pipeline:

- New `scripts/replay_golden_set.py`:
  - Loads questions from `get_default_evaluation_dataset()`
    (`src/rag/evaluation.py:641`). RAGAS **evaluation** stays manual-only
    (incident #504); this script only reuses the dataset's questions.
  - Authenticates like W2 (GoTrue password grant → JWT).
  - Sends each question sequentially (rate-limited, e.g. small sleep between
    turns) to `POST /api/copilotkit/chat` — the same `run_chatbot` cognitive
    path that produced the 123 historical signals — with
    `session_id="goldset-replay-<YYYYMMDD>-<n>"` for provenance.
  - Fail-soft per question: log and continue on a failed turn; exit non-zero
    only if all fail.
  - `--dry-run` prints the questions and target URL without sending.
  - `--limit N` for a cheap smoke run (cheapest-disproof: run `--limit 2`
    first and verify signal rows appear in `learning_signals` before the
    full 30).
  - Prints a summary: questions sent, per-question status, and a verification
    query hint (`SELECT count(*) FROM learning_signals WHERE
    signal_details->'metadata'->>'conversation_id' LIKE 'goldset-replay-%'`
    — planning must confirm the exact provenance column/path the collector
    persists, session id vs conversation id).
- Signals produced are **real**: the real system grading its real answers to
  real pharma questions (`is_synthetic=false` is correct for them), and they
  remain distinguishable by the session-id prefix.
- Expected yield: ~3 signals/turn (agent, investigator, summarizer) → ~90
  signals. Harder golden questions should produce genuinely low rewards →
  real patterns → real proposed updates that exercise the Patterns and
  Updates tabs plus the manual Apply flow.
- Then run a second learning cycle (same mechanics as W2) over the replay
  window.

### W4 — Window fix + honest empty states (frontend)

- `quickLearningCycle` (`frontend/src/api/feedback.ts:386`): widen the
  hardcoded 7-day lookback to 30 days and update the comment. Rationale: the
  7-day window can structurally never see signals older than a week while
  signal accrual is bursty — same "structurally zeroed" defect class as the
  Fabhalta gap floor (#1237). Other knobs (`min_feedback_count: 5`,
  `pattern_threshold: 0.1`, `auto_apply: false`, sync mode) unchanged.
- `FeedbackLearning.tsx` empty states ("No patterns detected" ~line 678,
  "No knowledge updates available" ~line 760): extend the copy to say *why*
  the tab can be empty — signals accrue per chat turn on the API chat path
  and cycles scan a bounded window — instead of a bare no-data message. The
  existing F-010 `cycleWarnings` banner already surfaces cycle warnings
  ("No feedback items collected"); keep it as the mechanism, no new state.

## Data flow (end state)

```
real chat turns ──POST /api/copilotkit/chat──▶ run_chatbot ▶ cognitive pipeline
                                                    │ SignalCollector
                                                    ▼
                                          learning_signals (is_synthetic=false)
                                                    │
POST /api/feedback/learn (window, auto_apply=false) │
        └─▶ feedback_collector ◀────────────────────┘  (+ chatbot_message_feedback)
             ▶ pattern_analyzer ▶ learning_extractor ▶ knowledge_updater
                                                          │ auto_apply=false →
                                                          │ propose only
                                                          ▼
        feedback_learning_batches / feedback_patterns / feedback_knowledge_updates
                                                          │
             /feedback-learning page ◀────────────────────┘
                  └─ human clicks Apply ─▶ existing /apply endpoint ─▶ knowledge stores
```

## Error handling

- **W1:** missing/absent `auto_apply` in state → treat as `False` (propose
  only). Never apply by default.
- **W2/W3 cycles:** cycle failures keep persisting failed batches with their
  warning strings (existing behavior); the F-010 banner surfaces them.
- **W3 script:** per-question failures logged and skipped; summary reports
  the failure count; `--dry-run` and `--limit` allow cheap validation before
  spending the full 30 turns.
- **Auth:** both operations fail closed on missing credentials — the script
  refuses to run without a token rather than falling back to unauthenticated
  calls.

## Testing & verification

- **Unit (backend):** `KnowledgeUpdaterNode` with `auto_apply=False` (and
  absent) → `_apply_update` not called, `proposed_updates` populated,
  `applied_updates == []`; with `auto_apply=True` → applies (existing
  behavior). Route test: `_execute_learning_cycle` threads the request flag
  into `initial_state`.
- **Frontend:** quick-cycle sends a 30-day `time_range_start`; empty-state
  copy renders.
- **Script:** unit test for question loading + payload construction;
  `--dry-run` covered; no live-network test in CI.
- **Gates:** ruff `check` **and** `format --check` on every touched Python
  file; frontend lint/tests per CI; targeted pytest locally (CI runs the full
  suite — no whole-tree runs on the droplet).
- **Live verification (after deploy + W2 + W3):** page at
  eznomics.site/feedback-learning shows the backfill batch and the replay
  batch with real counts; Patterns/Updates tabs populated; updates in
  `proposed` status until manually applied; DB row counts and
  `goldset-replay-` provenance confirmed via psql.

## Sequencing

1. One PR: W1 (auto_apply gate) + W3 script + W4 frontend, with tests.
2. Merge (no squash) → deploy → verify deploy converged.
3. W2 backfill cycle (operation) → verify page shows the batch.
4. W3 smoke (`--limit 2`) → verify signals land → full replay → second cycle
   → live-verify tabs.

## Out of scope

- Copilot/frontend chat signal wiring — issue #1240.
- Any change to the synthetic-exclusion filter in
  `LearningSignalsFeedbackStore` — it is the honesty guardrail and stays.
- RAGAS metric evaluation (manual-only per incident #504) — only the golden
  dataset's questions are reused.
- Backfilling synthetic signals or fabricating patterns/updates in any form.
