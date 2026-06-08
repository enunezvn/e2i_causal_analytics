"""R4b/H9 regression: fidelity_tracking_update must route through the R1
compare_experiment_to_twin convenience method (fixing the unbound-vars /
predicted_ci= signature / non-existent-column bugs and the
confidence_interval_coverage attr read), and compute_experiment_results must
enqueue it as the post-experiment producer on a FINAL analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np


def _real_fidelity_comparison(exp_id, sim_id):
    from src.services.results_analysis import FidelityComparison

    return FidelityComparison(
        experiment_id=exp_id,
        twin_simulation_id=sim_id,
        comparison_timestamp=datetime.now(timezone.utc),
        predicted_effect=0.05,
        actual_effect=0.06,
        prediction_error=0.01,
        prediction_error_percent=20.0,
        predicted_ci_lower=0.02,
        predicted_ci_upper=0.08,
        ci_coverage=True,
        fidelity_score=0.9,
        fidelity_grade="A",
    )


class TestFidelityTrackingUpdate:
    def test_completed_routes_through_compare_experiment_to_twin(self):
        from src.tasks.ab_testing_tasks import fidelity_tracking_update

        exp_id = uuid4()
        sim_id = uuid4()
        fc = _real_fidelity_comparison(exp_id, sim_id)

        svc = MagicMock()
        svc.compare_experiment_to_twin = AsyncMock(return_value=fc)
        with patch("src.services.results_analysis.ResultsAnalysisService", return_value=svc):
            result = fidelity_tracking_update.run(
                experiment_id=str(exp_id), twin_simulation_id=str(sim_id)
            )

        assert result["status"] == "completed"
        assert result["prediction_error"] == 0.01
        assert result["fidelity_score"] == 0.9
        assert result["ci_coverage"] is True
        svc.compare_experiment_to_twin.assert_awaited_once()

    def test_skipped_when_no_results_or_sim(self):
        from src.tasks.ab_testing_tasks import fidelity_tracking_update

        exp_id = uuid4()
        svc = MagicMock()
        svc.compare_experiment_to_twin = AsyncMock(
            side_effect=ValueError("No computed results for experiment")
        )
        with patch("src.services.results_analysis.ResultsAnalysisService", return_value=svc):
            result = fidelity_tracking_update.run(
                experiment_id=str(exp_id), twin_simulation_id=str(uuid4())
            )

        assert result["status"] == "skipped"
        assert "results" in result["reason"].lower()


class TestFidelityProducer:
    """The post-results producer fires only on a FINAL analysis. After the R5
    rewire, ``compute_experiment_results`` first resolves the experiment + loads
    the real outcome feed; the producer fires on the insufficient_data branch
    (empty arrays here) AND on the completed branch — but never for interim."""

    @staticmethod
    def _found_experiment_client():
        client = MagicMock()
        (
            client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "total_rx_count"}]
        return client

    def test_final_analysis_enqueues_fidelity_tracking_update(self):
        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(return_value=(np.array([]), np.array([])))
        with (
            patch("src.tasks.ab_testing_tasks.celery_app.send_task") as mock_send,
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

        # Empty arrays -> honest insufficient_data bail, but the FINAL producer
        # still fires (it self-skips downstream if no twin sim is linked).
        assert result["status"] == "insufficient_data"
        names = [c.args[0] if c.args else c.kwargs.get("name") for c in mock_send.call_args_list]
        assert "src.tasks.fidelity_tracking_update" in names

    def test_interim_analysis_does_not_enqueue_fidelity(self):
        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(return_value=(np.array([]), np.array([])))
        with (
            patch("src.tasks.ab_testing_tasks.celery_app.send_task") as mock_send,
            patch(
                "src.repositories.get_supabase_client",
                return_value=self._found_experiment_client(),
            ),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
        ):
            compute_experiment_results.run(experiment_id=exp_id, analysis_type="interim")

        names = [c.args[0] if c.args else c.kwargs.get("name") for c in mock_send.call_args_list]
        assert "src.tasks.fidelity_tracking_update" not in names
