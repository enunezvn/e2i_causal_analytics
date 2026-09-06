"""Execution outcome of audit-chain rows — shared by /health-score and /analytics.

Why this exists: both readers counted ANY ``validation_passed = false`` row as
a failed invocation. That column also carries SCIENTIFIC verdicts — the
heterogeneous optimizer's EconML<->CausalML agreement, causal_impact's
refutation ``overall_robust`` — recorded by the audited-node wrapper and then
re-recorded by every downstream node that returns ``{**state}``. /system-health
therefore warned "heterogeneous_optimizer has low success rate (89.7%)" over a
30-day window containing ZERO node errors (2026-09-06). Execution failure has
exactly one marker: the ``<node>_error`` row the wrapper writes when a node
raises. A run is the unit ("invocation"), not the row.
"""

from src.api.utils.audit_outcomes import WorkflowOutcomeTally, is_execution_failure


class TestIsExecutionFailure:
    def test_error_action_type_is_a_failure(self):
        assert is_execution_failure(
            {"action_type": "learn_policy_error", "validation_passed": False}
        )

    def test_failed_validation_verdict_is_not_a_failure(self):
        # The cross-library agreement verdict: the analysis ran fine and
        # honestly reported that the two libraries disagreed.
        assert not is_execution_failure(
            {"action_type": "uplift_analysis", "validation_passed": False}
        )

    def test_failed_refutation_is_not_a_failure(self):
        assert not is_execution_failure({"action_type": "refutation", "validation_passed": False})

    def test_missing_action_type_is_not_a_failure(self):
        assert not is_execution_failure({"validation_passed": None})
        assert not is_execution_failure({"action_type": None})


class TestWorkflowOutcomeTally:
    def test_one_workflow_many_rows_counts_once(self):
        tally = WorkflowOutcomeTally()
        for node in ("workflow_start", "uplift_analysis", "learn_policy", "generate_profiles"):
            tally.add({"workflow_id": "w1", "action_type": node, "validation_passed": False})
        assert (tally.total, tally.failed, tally.successful) == (1, 0, 1)
        assert tally.success_rate == 1.0

    def test_error_row_fails_its_workflow_once(self):
        tally = WorkflowOutcomeTally()
        tally.add({"workflow_id": "w1", "action_type": "workflow_start"})
        tally.add(
            {"workflow_id": "w1", "action_type": "estimation_error", "validation_passed": False}
        )
        # A retried node writes a second error row: still ONE failed run.
        tally.add(
            {"workflow_id": "w1", "action_type": "estimation_error", "validation_passed": False}
        )
        tally.add({"workflow_id": "w2", "action_type": "workflow_start"})
        tally.add({"workflow_id": "w2", "action_type": "estimation", "validation_passed": True})
        assert (tally.total, tally.failed, tally.successful) == (2, 1, 1)
        assert tally.success_rate == 0.5

    def test_rows_without_workflow_id_count_per_row(self):
        # Legacy pre-instrumentation rows carry no workflow_id: the row is the
        # only unit available, so it is counted as one run.
        tally = WorkflowOutcomeTally()
        tally.add({"action_type": "workflow_start"})
        tally.add({"action_type": "analysis_error"})
        tally.add({"workflow_id": "w1", "action_type": "workflow_start"})
        assert (tally.total, tally.failed) == (3, 1)

    def test_recent_counts_workflows_with_any_row_in_window(self):
        tally = WorkflowOutcomeTally()
        tally.add({"workflow_id": "w1", "action_type": "workflow_start"}, recent=True)
        tally.add({"workflow_id": "w1", "action_type": "estimation"}, recent=True)
        tally.add({"workflow_id": "w2", "action_type": "workflow_start"}, recent=False)
        tally.add({"action_type": "workflow_start"}, recent=True)
        assert tally.recent == 2

    def test_empty_tally_has_no_rate(self):
        tally = WorkflowOutcomeTally()
        assert (tally.total, tally.failed, tally.successful, tally.recent) == (0, 0, 0, 0)
        assert tally.success_rate is None
