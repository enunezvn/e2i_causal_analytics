# DSPy Loop — Premise Investigation (cheapest-disproof, read-only)

**Date:** 2026-06-08
**Method:** read-only source reads + read-only `docker exec supabase-db psql` SELECTs against the live DB. No mocks, no LM spend, no writes.
**Question:** does the merged DSPy self-improvement loop have any real fuel to optimize on?

---

## Verdict: the learner-side premise is DISPROVEN — the loop has no fuel.

Building Gap A (persist fix + generation beat + threshold) is premature: there is no real
feedback to turn into `feedback_learner` signals, and zero such signals exist.

## Evidence

### 1. The feedback-collection input path has no real source wired
`FeedbackCollectorNode._collect_user_feedback` (`nodes/feedback_collector.py:98`) calls
`feedback_store.get_feedback(start_time=, end_time=, agents=)`. **No production store implements
`get_feedback`** — only the in-`agent.py` `MockStore` (`:261`). The only bridge,
`dspy_receiver.get_feedback_items_from_signals` (`:413`), reads the **in-memory** receiver deque
(ephemeral, lost on restart — audit §3). So in production `feedback_store=None` →
`_collect_user_feedback` returns `[]`.

### 2. Zero `feedback_learner` signals exist; the table is stale
`SELECT source_agent, count(*), avg(reward), max(created_at) FROM dspy_agent_training_signals GROUP BY 1`:

| source_agent | count | avg_reward | latest |
|---|---|---|---|
| `response` | 189 | 0.800 | 2026-01-26 |
| `mipro_test_147b69ee_summarizer` | 55 | 0.828 | 2025-12-21 |

- **No `feedback_learner` rows** → `get_feedback_learner_training_signals` (filters `source_agent='feedback_learner'`) reads **0** → optimizer always `skipped_insufficient_signals`.
- **No rows from the real senders** (`causal_impact`, `gap_analyzer`, `heterogeneous_optimizer`, `prediction_synthesizer`) despite those emitting in code (audit §6) — so emissions are not reaching this table with the expected `source_agent`, or those agents have not run since the table was last populated.
- Latest signal is **Jan 2026** — the loop has been unfed for ~4.5 months.

### 3. There is essentially no real user feedback to derive signals from
| table | rows | latest |
|---|---|---|
| `feedback_items` | **0** | — |
| `validation_outcomes` | **0** | — |
| `feedback_patterns` | **0** | — |
| `chatbot_message_feedback` | 2 | 2026-01-14 |
| `feedback_learning_batches` | 2 | 2026-06-08 |
| `ml_predictions` | 4364 | 2026-01-01 (stale; predictions, not rating/correction feedback) |

`feedback_items` schema = `feedback_id, source_agent, payload, created_at` (mappable to the
collector's shape) — but it is empty.

## Implications for the build

- **Gap A (learner side) is blocked on a missing prerequisite: a real feedback source.** The true
  blocker is not the loop plumbing (PR #792 fixed that) but that **no feedback data flows into the
  system**. A generation beat over empty feedback windows yields content-empty signals → GEPA
  degenerate. Wiring the collector to a real source (e.g., an adapter over `feedback_items` /
  `chatbot_message_feedback`) is itself moot while those tables are empty.
- **Gap B (recipient side) uses different fuel** — recipients self-emit from their **own runtime
  outputs** + heuristic reward, independent of user feedback. Gap B can produce real results **iff
  the recipient agents are actually invoked in production**. That invocation question is unverified
  and is the next cheapest thing to check.

## Why the real-LM harness was NOT run
The cheap read-only experiment already answers the core question (no fuel). Spending real Anthropic
LM budget to run `learn()`×6 → GEPA would only confirm "content-empty signals → degenerate," which
is already established. Per CLAUDE.md cheapest-disproof-first, stop at the cheapest decisive
evidence and surface it.

## Recommended next cheapest experiment (before any build)
Verify whether real production traffic invokes the agents at all (senders + recipients): inspect
recent agent runs / audit_chain / orchestrator logs and whether the 4 senders ever wrote signals.
This decides whether the problem is "no feedback capture" (fixable wiring) or "no traffic"
(nothing to learn from yet), and whether Gap B has fuel.
