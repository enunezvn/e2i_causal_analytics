"""
Tests for `src/tasks/ab_testing_tasks.py` — F-009 fix (iter-1 codex closure).

Closes #422 (F-009). Before this PR the task passed `control_data=[]` and
`treatment_data=[]` into `ResultsAnalysisService.compute_itt_results` /
`compute_per_protocol_results`, which compute `np.mean([])` and `np.var([])`
producing NaN; the NaN-tainted `ExperimentResults` was persisted to
`ab_experiment_results`.

After the fix:
- The task bails in-line with `status='insufficient_data'` and `reason`
  citing the schema gap (no per-unit A/B metric-observation storage exists).
- It NEVER calls `compute_itt_results`/`compute_per_protocol_results`/
  `perform_interim_analysis` from these celery task paths under the current
  schema. When the storage schema lands (#422 follow-up), the bail is
  removed and real arrays are passed in.

Iter-1 codex feedback addressed:
- HIGH #1: removed `_load_experiment_metric_arrays` helper that always
  returned `None` (a relabeling pattern). The bail is now in-line with the
  honest reason — no helper pretending to load data.
- HIGH #2: removed `primary_metric = "conversion_rate"` hardcoded value
  and `np.ones(...)` "all-compliant" mask placeholders. Those were dead
  code after the bail anyway.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

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

    def test_bails_early_with_insufficient_data_status(self) -> None:
        """
        With no metric data available, the task returns status='insufficient_data'
        and does NOT call compute_itt_results / compute_per_protocol_results.
        """
        from src.tasks.ab_testing_tasks import compute_experiment_results

        mock_service = MagicMock()
        mock_service.compute_itt_results = AsyncMock()
        mock_service.compute_per_protocol_results = AsyncMock()
        mock_repo = MagicMock()

        exp_id = str(uuid4())

        with (
            patch(
                "src.services.results_analysis.ResultsAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "src.repositories.ab_results.ABResultsRepository",
                return_value=mock_repo,
            ),
        ):
            # Celery bound tasks: call via `.run(...)` to bypass `self` injection
            # (the `bind=True` decorator wraps the function so `self` is the
            # Task instance, not a positional arg the caller supplies).
            result: Dict[str, Any] = compute_experiment_results.run(
                experiment_id=exp_id, analysis_type="interim"
            )

        # Must bail early — no compute calls — no DB write.
        mock_service.compute_itt_results.assert_not_called()
        mock_service.compute_per_protocol_results.assert_not_called()

        # Must surface a structured insufficient_data status (not 'completed'
        # with NaN-tainted means).
        assert result["status"] == "insufficient_data", (
            f"Expected status='insufficient_data', got {result!r}. "
            "Passing empty arrays to compute_itt_results would produce "
            "NaN-tainted means and corrupt persisted results."
        )
        assert result["experiment_id"] == exp_id

    def test_final_analysis_type_also_bails_no_compute_per_protocol(self) -> None:
        """`analysis_type='final'` must also bail — never call
        `compute_per_protocol_results` with empty/synthetic compliance masks.
        """
        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        mock_service = MagicMock()
        mock_service.compute_itt_results = AsyncMock()
        mock_service.compute_per_protocol_results = AsyncMock()

        with (
            patch(
                "src.services.results_analysis.ResultsAnalysisService",
                return_value=mock_service,
            ),
            patch(
                "src.repositories.ab_results.ABResultsRepository",
                return_value=MagicMock(),
            ),
        ):
            result = compute_experiment_results.run(experiment_id=exp_id, analysis_type="final")

        mock_service.compute_itt_results.assert_not_called()
        mock_service.compute_per_protocol_results.assert_not_called()
        assert result["status"] == "insufficient_data"
        # Reason string must mention the schema gap (so users know WHY).
        assert "schema" in result["reason"].lower() or "#422" in result["reason"]


class TestScheduledInterimAnalysisNoEmptyArrays:
    """`scheduled_interim_analysis` MUST NOT pass empty arrays to perform_interim_analysis."""

    def test_skips_or_insufficient_data_when_no_metrics(self) -> None:
        """
        When the experiment has enrollment but no metric data, the task must
        either skip or return insufficient_data — NEVER call
        perform_interim_analysis(metric_data={'control': [], 'treatment': []}).
        """
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        exp_id = str(uuid4())

        # Mock enrollment stats: enough enrollment to trigger analysis under the
        # default schedule (information_fraction >= 0.25).
        mock_enrollment_stats = MagicMock()
        mock_enrollment_stats.total_enrolled = 500
        mock_enrollment_stats.target_sample_size = 1000

        mock_enrollment_service = MagicMock()
        mock_enrollment_service.get_enrollment_stats = AsyncMock(return_value=mock_enrollment_stats)

        mock_exp_repo = MagicMock()
        mock_exp_repo.get_interim_analyses = AsyncMock(return_value=[])

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
                "src.services.results_analysis.ResultsAnalysisService",
                return_value=MagicMock(),
            ),
        ):
            result = scheduled_interim_analysis.run(experiment_id=exp_id, force=True)

        # Must NOT have called perform_interim_analysis with empty arrays.
        if mock_interim_service.perform_interim_analysis.call_args is not None:
            kwargs = mock_interim_service.perform_interim_analysis.call_args.kwargs
            metric_data = kwargs.get("metric_data")
            if metric_data is not None:
                assert metric_data.get("control") != [], (
                    "perform_interim_analysis called with control=[] — F-009 regressed."
                )
                assert metric_data.get("treatment") != [], (
                    "perform_interim_analysis called with treatment=[] — F-009 regressed."
                )

        assert result["status"] in {"insufficient_data", "skipped", "failed"}, (
            f"Expected status in {{insufficient_data, skipped, failed}}, "
            f"got {result.get('status')!r}"
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
