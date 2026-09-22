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
        ).data = [{"brand": "Fabhalta", "prediction_target": "triggers_total_count"}]
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


# =============================================================================
# #2206 — the fidelity loop must be consumed by a RUNNING worker and have a
# LIVE producer; a completed comparison must roll up into the model's fidelity.
# =============================================================================


class TestFidelityLoopIsLive:
    def test_fidelity_tracking_update_routes_to_a_queue_a_running_tier_consumes(self):
        """`twins` is consumed only by worker_heavy (replicas 0 by #705 owner
        decision) — a task routed there never runs. worker_medium consumes
        `analytics`, where the producer (compute_experiment_results) already runs."""
        from src.workers.celery_app import celery_app, get_worker_info

        route = celery_app.conf.task_routes["src.tasks.fidelity_tracking_update"]
        assert route == {"queue": "analytics"}
        assert route["queue"] not in ("twins", "shap", "causal", "ml", "forecast")
        with patch.dict("os.environ", {"WORKER_TYPE": "medium"}):
            assert route["queue"] in get_worker_info()["queues"]

    def test_final_analysis_producer_does_not_pin_the_dark_twins_queue(self):
        """The producer used to override routing with queue='twins'. It must let the
        routing table decide (or name the consumed queue) — never the dark one."""
        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(return_value=(np.array([]), np.array([])))
        client = MagicMock()
        (
            client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "triggers_total_count"}]
        with (
            patch("src.tasks.ab_testing_tasks.celery_app.send_task") as mock_send,
            patch("src.repositories.get_supabase_client", return_value=client),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
        ):
            compute_experiment_results.run(experiment_id=exp_id, analysis_type="final")

        calls = [
            c for c in mock_send.call_args_list if c.args[0] == "src.tasks.fidelity_tracking_update"
        ]
        assert len(calls) == 1
        assert calls[0].kwargs.get("queue") in (None, "analytics")

    @staticmethod
    def _interim_patches(decision_value: str, existing_final_rows):
        from src.services.interim_analysis import StoppingDecision

        mock_enrollment_stats = MagicMock()
        mock_enrollment_stats.total_enrolled = 500
        mock_enrollment_stats.total_assigned = 1000
        mock_enrollment_service = MagicMock()
        mock_enrollment_service.get_enrollment_stats = AsyncMock(return_value=mock_enrollment_stats)

        mock_exp_repo = MagicMock()
        mock_exp_repo.get_interim_analyses = AsyncMock(return_value=[])
        (
            mock_exp_repo.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "triggers_total_count"}]

        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(
            return_value=(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        )

        interim_result = MagicMock()
        interim_result.analysis_number = 1
        interim_result.information_fraction = 0.5
        interim_result.effect_estimate = 1.0
        interim_result.p_value = 0.01
        interim_result.decision = StoppingDecision(decision_value)
        mock_interim_service = MagicMock()
        mock_interim_service.perform_interim_analysis = AsyncMock(return_value=interim_result)

        results_repo = MagicMock()
        results_repo.get_results = AsyncMock(return_value=existing_final_rows)

        return (
            patch(
                "src.repositories.ab_experiment.ABExperimentRepository", return_value=mock_exp_repo
            ),
            patch(
                "src.services.enrollment.EnrollmentService", return_value=mock_enrollment_service
            ),
            patch(
                "src.services.interim_analysis.InterimAnalysisService",
                return_value=mock_interim_service,
            ),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
            patch("src.repositories.ab_results.ABResultsRepository", return_value=results_repo),
            patch("src.tasks.ab_testing_tasks.celery_app.send_task"),
        )

    def _run_interim(self, decision_value: str, existing_final_rows=()):
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        exp_id = str(uuid4())
        p1, p2, p3, p4, p5, p6 = self._interim_patches(decision_value, list(existing_final_rows))
        with p1, p2, p3, p4, p5, p6 as mock_send:
            result = scheduled_interim_analysis.run(experiment_id=exp_id, force=True)
        finals = [
            c
            for c in mock_send.call_args_list
            if c.args and c.args[0] == "src.tasks.compute_experiment_results"
        ]
        return exp_id, result, finals

    def test_a_stopping_decision_enqueues_the_final_analysis(self):
        """compute_experiment_results(final) had NO producer anywhere (no beat entry,
        no send_task, no route) — the whole fidelity chain was dark at its root. The
        sequential test's stopping decision is the product moment that makes an
        analysis final, so it is the producer."""
        exp_id, result, finals = self._run_interim("stop_efficacy")
        assert result["status"] == "completed"
        assert len(finals) == 1
        assert finals[0].kwargs.get("args") == [exp_id, "final"] or finals[0].args[1:] == (
            [exp_id, "final"],
        )
        assert result.get("final_analysis_enqueued") is True

    def test_continue_does_not_enqueue_a_final_analysis(self):
        _, result, finals = self._run_interim("continue")
        assert result["status"] == "completed"
        assert finals == []
        assert result.get("final_analysis_enqueued") is False

    def test_an_existing_final_row_is_not_recomputed(self):
        _, result, finals = self._run_interim("stop_futility", existing_final_rows=[MagicMock()])
        assert finals == []
        assert result.get("final_analysis_enqueued") is False

    def test_completed_comparison_rolls_up_into_the_model_fidelity(self):
        """ab_fidelity_comparisons alone never changes digital_twin_models.fidelity_score
        (its only writer, update_fidelity_score, had zero callers) — the gate would stay
        'unvalidated' forever. The task now refreshes the model's fidelity from every
        comparison of that model's simulations."""
        from src.tasks.ab_testing_tasks import fidelity_tracking_update

        exp_id = uuid4()
        sim_id = uuid4()
        model_id = uuid4()
        fc = _real_fidelity_comparison(exp_id, sim_id)

        svc = MagicMock()
        svc.compare_experiment_to_twin = AsyncMock(return_value=fc)
        repo = MagicMock()
        repo.refresh_model_fidelity_from_comparisons = AsyncMock(
            return_value={"model_id": str(model_id), "fidelity_score": 0.9, "sample_count": 1}
        )
        repo.get_simulation = AsyncMock(
            return_value={"simulation_id": str(sim_id), "model_id": str(model_id)}
        )
        with (
            patch("src.services.results_analysis.ResultsAnalysisService", return_value=svc),
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=repo),
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                new=AsyncMock(return_value=MagicMock()),
            ),
        ):
            result = fidelity_tracking_update.run(
                experiment_id=str(exp_id), twin_simulation_id=str(sim_id)
            )

        assert result["status"] == "completed"
        repo.refresh_model_fidelity_from_comparisons.assert_awaited_once_with(model_id)
        assert result["model_fidelity"] == {
            "model_id": str(model_id),
            "fidelity_score": 0.9,
            "sample_count": 1,
        }


class TestProducerPathResolvesTheLinkedSimulation:
    def test_no_explicit_simulation_id_resolves_experiment_scoped(self):
        """The producer enqueues only the experiment id (codex r1 #1): the task must
        hand None through so compare_experiment_to_twin resolves the simulation via
        twin_simulations.experiment_design_id — the link /simulate now writes."""
        from src.tasks.ab_testing_tasks import fidelity_tracking_update

        exp_id = uuid4()
        sim_id = uuid4()
        fc = _real_fidelity_comparison(exp_id, sim_id)
        svc = MagicMock()
        svc.compare_experiment_to_twin = AsyncMock(return_value=fc)
        repo = MagicMock()
        repo.get_simulation = AsyncMock(
            return_value={"simulation_id": str(sim_id), "model_id": str(uuid4())}
        )
        repo.refresh_model_fidelity_from_comparisons = AsyncMock(return_value=None)
        with (
            patch("src.services.results_analysis.ResultsAnalysisService", return_value=svc),
            patch("src.digital_twin.twin_repository.TwinRepository", return_value=repo),
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                new=AsyncMock(return_value=MagicMock()),
            ),
        ):
            result = fidelity_tracking_update.run(experiment_id=str(exp_id))

        assert result["status"] == "completed"
        svc.compare_experiment_to_twin.assert_awaited_once_with(
            experiment_id=exp_id, twin_simulation_id=None, analysis_type="final"
        )
        assert result["twin_simulation_id"] == str(sim_id)


class TestFinalResultsAreIdempotent:
    def test_a_redelivered_final_task_does_not_recompute_but_still_fires_the_producer(self):
        """codex r2 #4: Celery late-acks; a worker lost after persisting the final row
        redelivers compute_experiment_results(final). The task checks for an existing
        final row itself (the interim producer's check does not cover redelivery) and
        skips the recompute. The fidelity enqueue is idempotent downstream (the
        comparison is an upsert on experiment+simulation+type), so it still fires."""
        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(
            return_value=(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
        )
        client = MagicMock()
        (
            client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "triggers_total_count"}]
        results_repo = MagicMock()
        results_repo.get_results = AsyncMock(return_value=[MagicMock()])
        svc = MagicMock()
        svc.compute_itt_results = AsyncMock()
        with (
            patch("src.tasks.ab_testing_tasks.celery_app.send_task") as mock_send,
            patch("src.repositories.get_supabase_client", return_value=client),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
            patch("src.repositories.ab_results.ABResultsRepository", return_value=results_repo),
            patch("src.services.results_analysis.ResultsAnalysisService", return_value=svc),
        ):
            result = compute_experiment_results.run(experiment_id=exp_id, analysis_type="final")

        assert result["status"] == "skipped"
        assert "already" in result["reason"]
        svc.compute_itt_results.assert_not_called()
        outcome_repo.load_arrays.assert_not_called()
        names = [c.args[0] for c in mock_send.call_args_list if c.args]
        assert names.count("src.tasks.fidelity_tracking_update") == 1
