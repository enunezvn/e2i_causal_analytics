"""#548: execute_twin_retraining runs the REAL TwinGenerator and fails closed.

The bug: ``TwinRetrainingService.trigger_retraining`` imports
``execute_twin_retraining`` from ``src.tasks.ab_testing_tasks`` and queues it,
but no such task existed — the ImportError was caught and logged as the
MISLEADING "Celery tasks not available" (Celery IS available; only this one
task was missing), so the live auto-retraining path silently no-oped.

The fix (mirroring #545's ``_execute_real_retraining``):
  - ``_twin_training_data_from_config``: resolves the twin's training data from
    the retraining contract; fails loud (ValueError) when the data source is
    missing — the system has NO live cohort feed, so a degradation auto-trigger
    today carries no data_source and MUST fail closed, never fabricate.
  - ``_extract_validation_r2``: pulls the real finite validation R² from the
    ``TwinModelMetrics`` ``TwinGenerator.train`` returns; ``None`` when absent.
  - ``_execute_real_twin_retraining``: a real train run writes the real metric
    via ``complete_retraining``; every failure mode fails closed (status=failed,
    complete_retraining NOT called, no metric written).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Bug repro: the task must exist and be importable from the path the service
# already imports it from.
# --------------------------------------------------------------------------- #
def test_execute_twin_retraining_exists_and_is_importable() -> None:
    from src.tasks.ab_testing_tasks import execute_twin_retraining

    # It is a registered Celery task with the conventional name.
    assert execute_twin_retraining.name == "src.tasks.execute_twin_retraining"


def test_execute_twin_retraining_routed_to_ml_queue() -> None:
    import src.tasks.ab_testing_tasks  # noqa: F401 — register tasks
    from src.workers.celery_app import celery_app

    routes = celery_app.conf.task_routes
    assert routes.get("src.tasks.execute_twin_retraining") == {"queue": "ml"}


# --------------------------------------------------------------------------- #
# _twin_training_data_from_config
# --------------------------------------------------------------------------- #
def test_training_data_resolves_from_data_source(tmp_path: Any) -> None:
    from src.tasks.ab_testing_tasks import _twin_training_data_from_config

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"a": [1, 2, 3], "y": [0.1, 0.2, 0.3]}).to_csv(csv, index=False)

    df, target = _twin_training_data_from_config({"data_source": str(csv), "target_column": "y"})
    assert list(df.columns) == ["a", "y"]
    assert target == "y"


def test_training_data_missing_source_fails_loud() -> None:
    from src.tasks.ab_testing_tasks import _twin_training_data_from_config

    with pytest.raises(ValueError, match="data_source"):
        _twin_training_data_from_config({"target_column": "y"})


def test_training_data_missing_target_fails_loud(tmp_path: Any) -> None:
    from src.tasks.ab_testing_tasks import _twin_training_data_from_config

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="target_column"):
        _twin_training_data_from_config({"data_source": str(csv)})


# --------------------------------------------------------------------------- #
# _extract_validation_r2
# --------------------------------------------------------------------------- #
def test_extract_r2_reads_finite_metric() -> None:
    from src.tasks.ab_testing_tasks import _extract_validation_r2

    metrics = MagicMock()
    metrics.r2_score = 0.83
    assert _extract_validation_r2(metrics) == 0.83


def test_extract_r2_returns_none_when_absent_or_nonfinite() -> None:
    from src.tasks.ab_testing_tasks import _extract_validation_r2

    none_metric = MagicMock()
    none_metric.r2_score = None
    assert _extract_validation_r2(none_metric) is None

    nan_metric = MagicMock()
    nan_metric.r2_score = float("nan")
    assert _extract_validation_r2(nan_metric) is None

    inf_metric = MagicMock()
    inf_metric.r2_score = float("inf")
    assert _extract_validation_r2(inf_metric) is None


# --------------------------------------------------------------------------- #
# _execute_real_twin_retraining
# --------------------------------------------------------------------------- #
def _model_row() -> dict:
    return {
        "model_id": str(uuid4()),
        "twin_type": "hcp",
        "brand": "Kisqali",
        "feature_columns": ["decile", "digital_engagement_score"],
        "target_columns": ["prescribing_change"],
        "training_config": {"algorithm": "random_forest"},
    }


def _patch_repo_and_service(model_row: Any):
    """Patch the TwinRepository (model lookup) and TwinRetrainingService.

    By default ``complete_retraining`` returns a truthy job object — the
    same-process / eager case where the job IS present in the service's
    in-process ``_pending_jobs`` and the completion was genuinely recorded.
    The cross-process case (job absent → ``complete_retraining`` returns None)
    is exercised explicitly in its own test.
    """
    repo = MagicMock()
    repo.get_model = AsyncMock(return_value=model_row)
    repo_patch = patch("src.digital_twin.twin_repository.TwinModelRepository", return_value=repo)

    service = MagicMock()
    # Non-None return => the job was found and completion was recorded.
    service.complete_retraining = AsyncMock(return_value=MagicMock(name="recorded_job"))
    service.fail_retraining = AsyncMock(return_value=MagicMock(name="failed_job"))
    return repo_patch, repo, service


@pytest.mark.asyncio
async def test_twin_retraining_records_real_metric(tmp_path: Any) -> None:
    """A real TwinGenerator.train returning a finite R² → job completed with
    that EXACT metric persisted via complete_retraining."""
    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    model_row = _model_row()
    repo_patch, _repo, service = _patch_repo_and_service(model_row)

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"decile": [1, 2], "digital_engagement_score": [0.1, 0.2]}).to_csv(
        csv, index=False
    )

    real_metrics = MagicMock()
    real_metrics.r2_score = 0.741
    real_metrics.model_id = uuid4()
    mock_gen = MagicMock()
    mock_gen.train = MagicMock(return_value=real_metrics)
    gen_patch = patch("src.tasks.ab_testing_tasks.TwinGenerator", MagicMock(return_value=mock_gen))

    cfg = {"data_source": str(csv), "target_column": "prescribing_change"}

    with repo_patch, gen_patch:
        out = await _execute_real_twin_retraining(
            retraining_job_id="twin-1",
            model_id=model_row["model_id"],
            training_config=cfg,
            service=service,
        )

    assert out["status"] == "completed"
    assert out["validation_r2"] == 0.741
    # Real metric persisted (exact value, not a fabricated default).
    service.complete_retraining.assert_awaited_once()
    assert service.complete_retraining.await_args.kwargs["fidelity_after"] == 0.741
    assert service.complete_retraining.await_args.kwargs["success"] is True
    # The real trainer was actually invoked.
    mock_gen.train.assert_called_once()


@pytest.mark.asyncio
async def test_twin_retraining_cross_process_completion_not_recorded(tmp_path: Any) -> None:
    """codex HIGH (#548): in a real Celery worker the service is a FRESH instance
    with an EMPTY _pending_jobs (jobs are in-process only; no DB), so
    complete_retraining returns None — the completion was NOT recorded. The task
    must NOT report 'completed' or a real-looking metric in that case; it returns
    an honest non-success status and writes no fabricated completion."""
    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    model_row = _model_row()
    repo_patch, _repo, service = _patch_repo_and_service(model_row)
    # Cross-process: the job is not in THIS worker's _pending_jobs.
    service.complete_retraining = AsyncMock(return_value=None)

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"decile": [1, 2], "digital_engagement_score": [0.1, 0.2]}).to_csv(
        csv, index=False
    )

    real_metrics = MagicMock()
    real_metrics.r2_score = 0.741
    real_metrics.model_id = uuid4()
    mock_gen = MagicMock()
    mock_gen.train = MagicMock(return_value=real_metrics)
    gen_patch = patch("src.tasks.ab_testing_tasks.TwinGenerator", MagicMock(return_value=mock_gen))

    cfg = {"data_source": str(csv), "target_column": "prescribing_change"}

    with repo_patch, gen_patch:
        out = await _execute_real_twin_retraining(
            retraining_job_id="twin-xproc",
            model_id=model_row["model_id"],
            training_config=cfg,
            service=service,
        )

    # NOT a success — the completion could not be recorded across the process
    # boundary, so claiming "completed" would be a false success.
    assert out["status"] != "completed"
    # No fabricated completion metric is surfaced as a recorded result.
    assert out.get("validation_r2") is None
    # The honest message names the real cause (cross-process job-store gap).
    assert "could not be recorded" in out["error"]
    assert "twin-xproc" in out["error"]
    # complete_retraining WAS attempted (the run really happened); it just
    # couldn't persist because the job isn't in this process's store.
    service.complete_retraining.assert_awaited_once()


@pytest.mark.asyncio
async def test_twin_retraining_end_to_end_real_trainer(tmp_path: Any) -> None:
    """Anti-mocking: run the ACTUAL TwinGenerator.train (real sklearn) on a
    synthetic cohort and persist the REAL held-out R² it produces — no mocked
    trainer, no fabricated metric."""
    import numpy as np

    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    rng = np.random.default_rng(0)
    n = 1200  # >= TwinGenerator.MIN_TRAINING_SAMPLES
    decile = rng.integers(1, 11, size=n).astype(float)
    engagement = rng.random(size=n)
    # A learnable signal so R² is a real finite number.
    y = 0.5 * decile + 2.0 * engagement + rng.normal(0, 0.1, size=n)

    csv = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "decile": decile,
            "digital_engagement_score": engagement,
            "prescribing_change": y,
        }
    ).to_csv(csv, index=False)

    model_row = _model_row()
    repo_patch, _repo, service = _patch_repo_and_service(model_row)
    cfg = {"data_source": str(csv), "target_column": "prescribing_change"}

    with repo_patch:  # TwinGenerator NOT patched — the real trainer runs.
        out = await _execute_real_twin_retraining(
            retraining_job_id="twin-e2e",
            model_id=model_row["model_id"],
            training_config=cfg,
            service=service,
        )

    assert out["status"] == "completed"
    # A real, finite R² was produced and persisted (not a placeholder).
    r2 = out["validation_r2"]
    assert isinstance(r2, float) and -1.0 <= r2 <= 1.0
    service.complete_retraining.assert_awaited_once()
    assert service.complete_retraining.await_args.kwargs["fidelity_after"] == r2


@pytest.mark.asyncio
async def test_twin_retraining_fail_closed_on_train_failure(tmp_path: Any) -> None:
    """TwinGenerator.train raises → job FAILED, complete_retraining NOT called,
    NO metric written."""
    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    model_row = _model_row()
    repo_patch, _repo, service = _patch_repo_and_service(model_row)

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"decile": [1, 2]}).to_csv(csv, index=False)

    mock_gen = MagicMock()
    mock_gen.train = MagicMock(side_effect=ValueError("Insufficient training data"))
    gen_patch = patch("src.tasks.ab_testing_tasks.TwinGenerator", MagicMock(return_value=mock_gen))

    cfg = {"data_source": str(csv), "target_column": "prescribing_change"}

    with repo_patch, gen_patch:
        out = await _execute_real_twin_retraining(
            retraining_job_id="twin-2",
            model_id=model_row["model_id"],
            training_config=cfg,
            service=service,
        )

    assert out["status"] == "failed"
    # Anti-mocking: NO metric persisted; job marked FAILED instead.
    service.complete_retraining.assert_not_awaited()
    service.fail_retraining.assert_awaited_once()
    assert out.get("validation_r2") is None


@pytest.mark.asyncio
async def test_twin_retraining_fail_closed_on_nonfinite_metric(tmp_path: Any) -> None:
    """train returns metrics with no certifiable (finite) R² → fail closed,
    write no metric rather than a placeholder."""
    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    model_row = _model_row()
    repo_patch, _repo, service = _patch_repo_and_service(model_row)

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"decile": [1, 2]}).to_csv(csv, index=False)

    metrics = MagicMock()
    metrics.r2_score = None  # no certifiable metric
    metrics.model_id = uuid4()
    mock_gen = MagicMock()
    mock_gen.train = MagicMock(return_value=metrics)
    gen_patch = patch("src.tasks.ab_testing_tasks.TwinGenerator", MagicMock(return_value=mock_gen))

    cfg = {"data_source": str(csv), "target_column": "prescribing_change"}

    with repo_patch, gen_patch:
        out = await _execute_real_twin_retraining(
            retraining_job_id="twin-3",
            model_id=model_row["model_id"],
            training_config=cfg,
            service=service,
        )

    assert out["status"] == "failed"
    service.complete_retraining.assert_not_awaited()
    service.fail_retraining.assert_awaited_once()


@pytest.mark.asyncio
async def test_twin_retraining_fail_closed_on_missing_data_source() -> None:
    """A degradation auto-trigger today carries no data_source (the system has
    no live cohort feed) → fail closed, trainer never invoked, no metric."""
    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    model_row = _model_row()
    repo_patch, _repo, service = _patch_repo_and_service(model_row)

    mock_gen = MagicMock()
    gen_patch = patch("src.tasks.ab_testing_tasks.TwinGenerator", MagicMock(return_value=mock_gen))

    # legacy _build_training_config-style payload: no cohort identity.
    cfg = {"min_samples": 1000, "cv_folds": 5, "hyperparameter_tuning": True}

    with repo_patch, gen_patch:
        out = await _execute_real_twin_retraining(
            retraining_job_id="twin-4",
            model_id=model_row["model_id"],
            training_config=cfg,
            service=service,
        )

    assert out["status"] == "failed"
    mock_gen.train.assert_not_called()
    service.complete_retraining.assert_not_awaited()
    service.fail_retraining.assert_awaited_once()
    service.fail_retraining.assert_awaited_once()


@pytest.mark.asyncio
async def test_twin_retraining_fail_closed_on_missing_model_row(tmp_path: Any) -> None:
    """Model row cannot be located → fail closed (cannot rebuild the trainer)."""
    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    repo = MagicMock()
    repo.get_model = AsyncMock(return_value=None)
    repo_patch = patch("src.digital_twin.twin_repository.TwinModelRepository", return_value=repo)
    service = MagicMock()
    service.complete_retraining = AsyncMock()
    service.fail_retraining = AsyncMock()

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"decile": [1, 2]}).to_csv(csv, index=False)
    cfg = {"data_source": str(csv), "target_column": "prescribing_change"}

    mock_gen = MagicMock()
    gen_patch = patch("src.tasks.ab_testing_tasks.TwinGenerator", MagicMock(return_value=mock_gen))

    with repo_patch, gen_patch:
        out = await _execute_real_twin_retraining(
            retraining_job_id="twin-5",
            model_id=str(uuid4()),
            training_config=cfg,
            service=service,
        )

    assert out["status"] == "failed"
    mock_gen.train.assert_not_called()
    service.complete_retraining.assert_not_awaited()
    service.fail_retraining.assert_awaited_once()
