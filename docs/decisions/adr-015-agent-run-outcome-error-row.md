# ADR-015: An agent run has failed only when a node left a `<node>_error` audit row

**Date**: 2026-09-06 | **Status**: Accepted | **Implemented by**: PR #1902 (merge 46a88558f)

## Context

`audit_chain_entries` is the platform's execution ledger, read by the agent-health card (`/health-score` → /system-health) and by the analytics readers (`/analytics`, `/analytics/agents/{name}`). Those readers disagreed with the ledger about what "a run failed" means.

The audited-node wrapper records under `validation_passed` whatever a node returns as `validation_passed` / `overall_robust` — the heterogeneous optimizer's EconML↔CausalML cross-library agreement, `causal_impact`'s refutation verdict. Those are **scientific results about the data**, and a downstream node returning `{**state}` re-records the same verdict once per node. Counting them as failed invocations made /system-health warn *"heterogeneous_optimizer has low success rate (89.7%)"* over a 30-day window in which **no node of any agent raised**.

The mirror-image defect sat on the writer side: several fail-closed paths returned an error result **without writing any error row at all**, so a run that failed closed read back as a successful invocation.

Separately, `/feedback-learning` pinned a 2026-08-08 goldset-replay detection ("cognitive_investigator has high negative feedback rate") at the top of the page, sorted by severity, with no signal path that could ever confirm or clear it.

## Decision

1. **Execution failure has exactly one marker: an `action_type = "<node>_error"` row** (`src/api/utils/audit_outcomes.py`, `is_execution_failure`). This is the single definition shared by the health reader and both analytics readers; each one imports it rather than re-deriving it.
2. **`validation_passed` is reserved for verdicts** and is never read as a run outcome. Verdicts surface where they belong — the Library Validation and Refutation cards on the analysis pages.
3. **The unit of an "invocation" is the workflow run (`workflow_id`), not the row.** A run writes one genesis row plus one row per node; `WorkflowOutcomeTally` groups rows into runs, and a run counts as failed once however many error rows it wrote. Legacy rows with no `workflow_id` have no other unit and count as one run each. `success_rate` over zero runs is **None** (unmeasured), never 0.
4. **A failed run is not spelled as N error rows.** Only the node that flips the status to failed *and* carries a failure payload (`errors` / `error` / `error_message`) is the failure: a downstream node that passes an already-failed state through did not fail, and a node that sets status failed carrying only `warnings` reported a verdict (`resource_optimizer` on an infeasible solve: "no allocation satisfies the constraints"), not a crash.
5. **Corollary — every fail-closed writer must leave the marker.** A reader definition is only honest if the writers uphold it. PR #1902 fixed both audited-node wrappers, `causal_impact`'s own wrapper, and the tool composer's total-tool-failure gate (`<phase>_error`, `compose_error` when no phase is known) to write the row on every fail-closed path. Audit writes stay best-effort — the row is telemetry, never a barrier to returning the error to the caller.
6. **Detected feedback patterns age out after 30 days** (`PATTERN_MAX_AGE_DAYS`, `recent_patterns`). Patterns are persisted forever and the learning cycle re-detects any that still exist (24 h window, roughly every 6 h), so one not re-detected for a month is history, not an open finding; `include_stale=true` still returns them. The window matches the /system-health agent card's telemetry window. A pattern with **no** `detected_at` is kept — an unknown age is not evidence of staleness.

## Consequences

- (+) /system-health's 89.7% warning resolves to a measured rate over the same window, and both dashboards now answer the same question with the same arithmetic.
- (+) An operator reading "success rate" gets execution health; an analyst reading a refutation card gets the science. Neither number can be mistaken for the other.
- (−) The success rate is only as honest as the writers: any **new** fail-closed path must write its `<node>_error` row or it will silently read as a success. This is the recurring failure mode, not a one-time fix — new error paths need the marker as part of the change.
- (−) Historical rows are not rewritten. Runs from before #1902 that failed closed without a marker still read as successes; the 30-day windows age that out rather than a backfill.
- (−) `output_data` is persisted only as a hash, so `action_type` is the only readable outcome on a row — the marker cannot be reconstructed after the fact from the payload.

## References

- `src/api/utils/audit_outcomes.py` — `is_execution_failure`, `WorkflowOutcomeTally`
- `src/agents/base/audit_chain_mixin.py` — which node's failure writes the row; `src/agents/tool_composer/composer.py` — the composer gate
- `src/api/routes/health_score.py`, `src/api/routes/analytics.py`, `src/api/routes/agents.py` — the three readers; `src/api/routes/feedback.py` — `PATTERN_MAX_AGE_DAYS`
