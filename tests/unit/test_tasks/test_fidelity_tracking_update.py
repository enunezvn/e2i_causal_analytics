"""R4b/H9 regression: fidelity_tracking_update must route through the R1
compare_experiment_to_twin convenience method (fixing the unbound-vars /
predicted_ci= signature / non-existent-column bugs and the
confidence_interval_coverage attr read), and compute_experiment_results must
enqueue it as the post-experiment producer on a FINAL analysis."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

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

    @staticmethod
    def _race_patches(exp_id, *, compute_raises):
        """Two deliveries racing within one round-trip: the pre-check sees no final
        row (both pass it), the loser's INSERT then trips ml/046's partial unique
        index inside compute_itt_results."""
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(
            return_value=(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
        )
        client = MagicMock()
        (
            client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "triggers_total_count"}]
        results_repo = MagicMock()
        results_repo.get_results = AsyncMock(return_value=[])
        svc = MagicMock()
        svc.compute_itt_results = AsyncMock(side_effect=compute_raises)
        send = MagicMock()
        stack = ExitStack()
        for p in (
            patch("src.repositories.get_supabase_client", return_value=client),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
            patch("src.repositories.ab_results.ABResultsRepository", return_value=results_repo),
            patch("src.services.results_analysis.ResultsAnalysisService", return_value=svc),
            patch("src.tasks.ab_testing_tasks.celery_app.send_task", send),
        ):
            stack.enter_context(p)
        return stack, outcome_repo, svc, send

    def test_a_delivery_that_loses_the_insert_race_skips_and_still_fires_the_producer(self):
        """Owner fix (#2206): ml/046 makes the FINAL claim atomic. The delivery whose
        INSERT loses is told so by the repository (FinalResultAlreadyPersisted with the
        winner's row) and must report ``skipped`` — not ``failed`` — and still enqueue
        fidelity tracking, exactly like the pre-check branch above."""
        from src.repositories.ab_results import ExperimentResultRecord, FinalResultAlreadyPersisted
        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        winner = ExperimentResultRecord(
            id=uuid4(),
            experiment_id=UUID(exp_id),
            analysis_type="final",
            analysis_method="itt",
            computed_at=datetime.now(timezone.utc),
            primary_metric="triggers_total_count",
            control_mean=1.0,
            treatment_mean=1.2,
            effect_estimate=0.2,
            effect_ci_lower=0.1,
            effect_ci_upper=0.3,
            p_value=0.01,
            sample_size_control=2,
            sample_size_treatment=2,
            statistical_power=0.8,
            is_significant=True,
        )
        stack, outcome_repo, svc, mock_send = self._race_patches(
            exp_id, compute_raises=FinalResultAlreadyPersisted(UUID(exp_id), winner)
        )
        with stack:
            result = compute_experiment_results.run(experiment_id=exp_id, analysis_type="final")

        assert result["status"] == "skipped"
        assert "concurrent" in result["reason"]
        assert result["results_id"] == str(winner.id)
        # It got past the pre-check (the race is real) and only lost at the INSERT.
        outcome_repo.load_arrays.assert_awaited_once()
        svc.compute_itt_results.assert_awaited_once()
        names = [c.args[0] for c in mock_send.call_args_list if c.args]
        assert names.count("src.tasks.fidelity_tracking_update") == 1

    def test_an_interim_unique_violation_is_still_a_failure(self):
        """Any other 23505 (an interim never hits the partial index) propagates as
        today: the task reports ``failed`` and never fires the fidelity producer."""
        from postgrest.exceptions import APIError

        from src.tasks.ab_testing_tasks import compute_experiment_results

        exp_id = str(uuid4())
        stack, _outcome_repo, _svc, mock_send = self._race_patches(
            exp_id,
            compute_raises=APIError(
                {
                    "message": 'duplicate key value violates unique constraint "x"',
                    "code": "23505",
                    "hint": None,
                    "details": None,
                }
            ),
        )
        with stack:
            result = compute_experiment_results.run(experiment_id=exp_id, analysis_type="interim")

        assert result["status"] == "failed"
        assert "duplicate key" in result["error"]
        assert not mock_send.called


class TestFinalAnalysisIsReconciledIndependentlyOfMilestones:
    """codex r7 #1: a broker failure while enqueueing compute_experiment_results(final)
    must not leave the loop permanently dark — later sweeps return "No new milestone
    reached" and would never retry. The persisted stopping decision is the durable
    record; every sweep reconciles it against the final-results table."""

    @staticmethod
    def _patches(previous, *, enqueue_raises=False, force=True):
        from src.services.interim_analysis import StoppingDecision

        stats = MagicMock()
        stats.total_enrolled = 500
        stats.total_assigned = 1000
        enrollment = MagicMock()
        enrollment.get_enrollment_stats = AsyncMock(return_value=stats)
        exp_repo = MagicMock()
        exp_repo.get_interim_analyses = AsyncMock(return_value=previous)
        (
            exp_repo.client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value
        ).data = [{"brand": "Fabhalta", "prediction_target": "triggers_total_count"}]
        outcome_repo = MagicMock()
        outcome_repo.load_arrays = AsyncMock(
            return_value=(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        )
        interim_result = MagicMock()
        interim_result.analysis_number = len(previous) + 1
        interim_result.information_fraction = 0.5
        interim_result.effect_estimate = 1.0
        interim_result.p_value = 0.01
        interim_result.decision = StoppingDecision.STOP_EFFICACY
        interim_service = MagicMock()
        interim_service.perform_interim_analysis = AsyncMock(return_value=interim_result)
        results_repo = MagicMock()
        results_repo.get_results = AsyncMock(return_value=[])
        send = MagicMock(side_effect=RuntimeError("broker down") if enqueue_raises else None)
        return (
            patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=exp_repo),
            patch("src.services.enrollment.EnrollmentService", return_value=enrollment),
            patch(
                "src.services.interim_analysis.InterimAnalysisService", return_value=interim_service
            ),
            patch(
                "src.repositories.experiment_outcome.ExperimentOutcomeRepository",
                return_value=outcome_repo,
            ),
            patch("src.repositories.ab_results.ABResultsRepository", return_value=results_repo),
            patch("src.tasks.ab_testing_tasks.celery_app.send_task", send),
        ), send

    def test_a_broker_failure_is_reported_not_raised_and_the_next_sweep_retries(self):
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        exp_id = str(uuid4())
        patches, send = self._patches([], enqueue_raises=True)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            first = scheduled_interim_analysis.run(experiment_id=exp_id, force=True)
        assert first["status"] == "completed"
        assert first["final_analysis_enqueued"] is False
        assert send.call_count == 1

        # Next daily sweep: the stopping decision is persisted, no new milestone is
        # reached (force=False; 0.25 and 0.5 already analysed at fraction 0.5) — the
        # final enqueue is retried.
        earlier = MagicMock()
        earlier.information_fraction = 0.25
        earlier.decision = "continue"
        stopped = MagicMock()
        stopped.information_fraction = 0.5
        stopped.decision = "stop_efficacy"
        patches, send = self._patches([earlier, stopped], force=False)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            second = scheduled_interim_analysis.run(experiment_id=exp_id, force=False)
        assert second["status"] == "skipped"
        assert second["reason"] == "No new milestone reached"
        assert second["final_analysis_enqueued"] is True
        finals = [
            c for c in send.call_args_list if c.args[0] == "src.tasks.compute_experiment_results"
        ]
        assert len(finals) == 1

    def test_reconciliation_does_nothing_without_a_stopping_decision(self):
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        earlier = MagicMock()
        earlier.information_fraction = 0.25
        earlier.decision = "continue"
        continuing = MagicMock()
        continuing.information_fraction = 0.5
        continuing.decision = "continue"
        patches, send = self._patches([earlier, continuing], force=False)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = scheduled_interim_analysis.run(experiment_id=str(uuid4()), force=False)
        assert result["status"] == "skipped"
        assert result["final_analysis_enqueued"] is False
        assert send.call_count == 0


class TestFidelityHopIsReconciledToo:
    """codex r8: the final-results → fidelity hop was one-shot. The sweep now also
    detects a FINAL row with no fidelity comparison for the experiment and
    enqueues fidelity_tracking_update directly (idempotent; it skips until a
    simulation is linked, and the next sweep tries again)."""

    @staticmethod
    def _sweep(previous, *, final_rows, comparisons, fidelity_enqueue_raises=False):
        stats = MagicMock()
        stats.total_enrolled = 500
        stats.total_assigned = 1000
        enrollment = MagicMock()
        enrollment.get_enrollment_stats = AsyncMock(return_value=stats)
        exp_repo = MagicMock()
        exp_repo.get_interim_analyses = AsyncMock(return_value=previous)
        results_repo = MagicMock()
        results_repo.get_results = AsyncMock(return_value=final_rows)
        results_repo.get_fidelity_comparisons = AsyncMock(return_value=comparisons)

        def _send(name, *a, **k):
            if fidelity_enqueue_raises and name == "src.tasks.fidelity_tracking_update":
                raise RuntimeError("broker down")

        send = MagicMock(side_effect=_send)
        from src.tasks.ab_testing_tasks import scheduled_interim_analysis

        with (
            patch("src.repositories.ab_experiment.ABExperimentRepository", return_value=exp_repo),
            patch("src.services.enrollment.EnrollmentService", return_value=enrollment),
            patch("src.repositories.ab_results.ABResultsRepository", return_value=results_repo),
            patch("src.tasks.ab_testing_tasks.celery_app.send_task", send),
        ):
            result = scheduled_interim_analysis.run(experiment_id=str(uuid4()), force=False)
        names = [c.args[0] for c in send.call_args_list if c.args]
        return result, names

    @staticmethod
    def _stopped_history():
        earlier = MagicMock()
        earlier.information_fraction = 0.25
        earlier.decision = "continue"
        stopped = MagicMock()
        stopped.information_fraction = 0.5
        stopped.decision = "stop_futility"
        return [earlier, stopped]

    def test_a_final_row_without_a_comparison_enqueues_fidelity_directly(self):
        result, names = self._sweep(
            self._stopped_history(), final_rows=[MagicMock()], comparisons=[]
        )
        assert result["reason"] == "No new milestone reached"
        assert names == ["src.tasks.fidelity_tracking_update"]
        assert result["final_analysis_enqueued"] is False
        assert result["fidelity_tracking_enqueued"] is True

    def test_an_existing_comparison_means_the_loop_is_closed_for_that_experiment(self):
        result, names = self._sweep(
            self._stopped_history(), final_rows=[MagicMock()], comparisons=[MagicMock()]
        )
        assert names == []
        assert result["fidelity_tracking_enqueued"] is False

    def test_a_broker_failure_on_the_fidelity_hop_is_reported_and_retried_next_sweep(self):
        result, names = self._sweep(
            self._stopped_history(),
            final_rows=[MagicMock()],
            comparisons=[],
            fidelity_enqueue_raises=True,
        )
        assert result["status"] == "skipped"
        assert result["fidelity_tracking_enqueued"] is False
        result2, names2 = self._sweep(
            self._stopped_history(), final_rows=[MagicMock()], comparisons=[]
        )
        assert result2["fidelity_tracking_enqueued"] is True
        assert names2 == ["src.tasks.fidelity_tracking_update"]
