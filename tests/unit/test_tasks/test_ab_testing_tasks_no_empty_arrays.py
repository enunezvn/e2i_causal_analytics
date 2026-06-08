"""
Tests for `src/tasks/ab_testing_tasks.py` — F-009 fix (iter-1 codex closure).

Closes #422 (F-009). Before this PR the task passed `control_data=[]` and
`treatment_data=[]` into `ResultsAnalysisService.compute_itt_results` /
`compute_per_protocol_results`, which compute `np.mean([])` and `np.var([])`
producing NaN; the NaN-tainted `ExperimentResults` was persisted to
`ab_experiment_results`.

The #422 no-NaN invariant — preserved under #705 R5:
- The real per-unit outcome feed (`ExperimentOutcomeRepository.load_arrays`,
  assignments ⋈ business_metrics) now supplies the arrays. The task still
  returns `status='insufficient_data'` and NEVER calls
  `compute_itt_results`/`compute_per_protocol_results`/`perform_interim_analysis`
  when an arm has fewer than 2 outcome-bearing units — so empty/degenerate
  arrays never reach the analysis (no NaN persisted). The difference from the
  old behavior is that the bail is now DATA-DRIVEN (empty feed) rather than a
  blanket schema-gap bail; when real assignments + metrics exist, the task
  computes and persists a real result.
- The literal `control_data = []` placeholder must remain absent (pinned below).
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np

# Note: imports below use string-form patch targets so the celery worker
# config does not need to be loaded at test collection time.


class _FakeCeleryRequest:
    """Stub for self.request inside @celery_app.task."""

    id = "test-task-id"


def _fake_self() -> MagicMock:
    """Construct a fake `self` for a bound celery task."""
    s = MagicMock()
    s.request = _FakeCeleryRequest()
    return s


class TestComputeExperimentResultsNoEmptyArrays:
    """`compute_experiment_results` MUST NOT pass empty arrays to compute_itt_results."""

    @staticmethod
    def _found_experiment_client() -> MagicMock:
        client = MagicMock()
        (
            client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "total_rx_count"}]
        return client

    def test_bails_early_with_insufficient_data_status(self) -> None:
        """
        When the real outcome feed yields too few outcome-bearing units, the task
        returns status='insufficient_data' and does NOT call compute_itt_results /
        compute_per_protocol_results (#705 R5 preserves the #422 no-NaN invariant
        via the empty/too-small guard — now data-driven, not a blanket bail).
        """
        from src.tasks.ab_testing_tasks import compute_experiment_results

        mock_service = MagicMock()
        mock_service.compute_itt_results = AsyncMock()
        mock_service.compute_per_protocol_results = AsyncMock()
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(return_value=(np.array([]), np.array([])))

        exp_id = str(uuid4())

        with (
            patch(
                "src.services.results_analysis.ResultsAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "src.repositories.get_supabase_client",
                return_value=self._found_experiment_client(),
            ),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
        ):
            # Celery bound tasks: call via `.run(...)` to bypass `self` injection.
            result: Dict[str, Any] = compute_experiment_results.run(
                experiment_id=exp_id, analysis_type="interim"
            )

        # Empty arrays -> no compute call -> no NaN-tainted DB write.
        mock_service.compute_itt_results.assert_not_called()
        mock_service.compute_per_protocol_results.assert_not_called()
        assert result["status"] == "insufficient_data", (
            f"Expected status='insufficient_data', got {result!r}. "
            "Passing empty arrays to compute_itt_results would produce "
            "NaN-tainted means and corrupt persisted results."
        )
        assert result["experiment_id"] == exp_id

    def test_final_analysis_type_also_bails_no_compute_per_protocol(self) -> None:
        """`analysis_type='final'` must also bail on an empty feed — never call
        `compute_per_protocol_results` with empty/synthetic compliance masks.
        """
        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        mock_service = MagicMock()
        mock_service.compute_itt_results = AsyncMock()
        mock_service.compute_per_protocol_results = AsyncMock()
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(return_value=(np.array([]), np.array([])))

        with (
            patch(
                "src.services.results_analysis.ResultsAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "src.repositories.get_supabase_client",
                return_value=self._found_experiment_client(),
            ),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
        ):
            result = compute_experiment_results.run(experiment_id=exp_id, analysis_type="final")

        mock_service.compute_itt_results.assert_not_called()
        mock_service.compute_per_protocol_results.assert_not_called()
        assert result["status"] == "insufficient_data"
        # Reason string explains WHY (too few outcome-bearing units in an arm).
        assert "outcome-bearing units" in result["reason"]


class TestScheduledInterimAnalysisNoEmptyArrays:
    """`scheduled_interim_analysis` MUST NOT pass empty arrays to perform_interim_analysis."""

    def test_skips_or_insufficient_data_when_no_metrics(self) -> None:
        """
        With a found experiment but an EMPTY real outcome feed, the interim task
        returns insufficient_data and NEVER calls perform_interim_analysis — the
        #422 no-empty-arrays invariant, preserved under the #705 R5 data-driven feed.
        """
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        exp_id = str(uuid4())

        # Enrollment stats: enough to pass the milestone gate (force=True anyway).
        mock_enrollment_stats = MagicMock()
        mock_enrollment_stats.total_enrolled = 500
        mock_enrollment_stats.total_assigned = 1000

        mock_enrollment_service = MagicMock()
        mock_enrollment_service.get_enrollment_stats = AsyncMock(return_value=mock_enrollment_stats)

        # exp_repo.client resolves the experiment (brand + prediction_target).
        mock_exp_repo = MagicMock()
        mock_exp_repo.get_interim_analyses = AsyncMock(return_value=[])
        (
            mock_exp_repo.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "total_rx_count"}]

        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(return_value=(np.array([]), np.array([])))

        mock_interim_service = MagicMock()
        mock_interim_service.perform_interim_analysis = AsyncMock()

        with (
            patch(
                "src.repositories.ab_experiment.ABExperimentRepository",
                return_value=mock_exp_repo,
            ),
            patch(
                "src.services.enrollment.EnrollmentService",
                return_value=mock_enrollment_service,
            ),
            patch(
                "src.services.interim_analysis.InterimAnalysisService",
                return_value=mock_interim_service,
            ),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
        ):
            result = scheduled_interim_analysis.run(experiment_id=exp_id, force=True)

        # Empty feed -> never call perform_interim_analysis -> no NaN.
        mock_interim_service.perform_interim_analysis.assert_not_called()
        assert result["status"] == "insufficient_data", (
            f"Expected status='insufficient_data', got {result.get('status')!r}"
        )


class TestNoPlaceholderInTaskSource:
    """Regression pin: source must not contain the `control_data = []` placeholder."""

    def test_task_source_has_no_empty_array_placeholder(self) -> None:
        """
        Pin the absence of `control_data = []` (the literal placeholder).

        See #422 / F-009.
        """
        from pathlib import Path

        task_path = Path(__file__).resolve().parents[3] / "src" / "tasks" / "ab_testing_tasks.py"
        source = task_path.read_text()
        # Forbid the exact placeholder pattern.
        forbidden = (
            "control_data = []\n            treatment_data = []",
            "# Placeholder: Get control and treatment data",
        )
        for marker in forbidden:
            assert marker not in source, (
                f"Detected re-introduction of empty-array placeholder: {marker!r}. "
                "See #422 / F-009."
            )
