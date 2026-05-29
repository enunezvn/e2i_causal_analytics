"""#549: TwinRetrainingService durable write-through across the worker boundary.

The fix: when a ``TwinRetrainingService`` is given a ``job_repository`` it
persists the job at trigger time and reads/updates it through that durable,
process-shared store — so a Celery worker (a fresh service instance with an
EMPTY ``_pending_jobs``) finds the job the API created, records completion with
the real metric, and the API re-reads it. With NO ``job_repository`` the service
is byte-for-byte the legacy in-process-dict behavior (covered by
``test_retraining_service.py``).

These tests inject ONE ``FakeSupabaseClient`` (see ``conftest.py``) into two
separate repo+service pairs to reproduce the worker boundary: separate
instances, SAME database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from src.digital_twin.retraining_service import (
    TwinRetrainingService,
    TwinRetrainingStatus,
    TwinTriggerReason,
    get_twin_retraining_service,
)
from src.digital_twin.twin_repository import TwinRetrainingJobRepository


# --------------------------------------------------------------------------- #
# Factory auto-wiring — the API + worker BOTH get the shared durable store
# with no explicit wiring, which is what makes #549 work in production.
# --------------------------------------------------------------------------- #
def test_factory_auto_wires_durable_job_repository() -> None:
    service = get_twin_retraining_service()
    assert isinstance(service.job_repository, TwinRetrainingJobRepository)


def test_factory_respects_explicit_job_repository() -> None:
    sentinel = object()
    service = get_twin_retraining_service(job_repository=sentinel)
    assert service.job_repository is sentinel


@pytest.fixture(autouse=True)
def _no_broker_enqueue():
    """Keep trigger_retraining off the live Celery broker (mirrors
    test_retraining_service.py): patch the real task's .delay so these tests
    exercise the service's own job-store logic, not Redis I/O."""
    with patch("src.tasks.ab_testing_tasks.execute_twin_retraining") as mock_task:
        mock_task.delay = MagicMock(return_value=MagicMock(id="test-task-id"))
        yield mock_task


def _service(job_repo: TwinRetrainingJobRepository) -> TwinRetrainingService:
    return TwinRetrainingService(job_repository=job_repo)


@pytest.mark.asyncio
async def test_trigger_persists_job_to_durable_store(fake_supabase) -> None:
    """trigger_retraining writes the job to the durable store, visible to a
    fresh repo instance bound to the same DB."""
    service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))
    model_id = uuid4()

    job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    # A DIFFERENT repo instance (another process) sees the persisted job.
    other = TwinRetrainingJobRepository(supabase_client=fake_supabase)
    record = await other.get_job(job.job_id)
    assert record is not None
    assert record.id == job.job_id
    assert record.model_id == str(model_id)
    assert record.status == "pending"
    assert record.trigger_reason == "manual"


@pytest.mark.asyncio
async def test_cross_process_complete_records_via_shared_store(fake_supabase) -> None:
    """THE #549 fix: the API-process service triggers; a fresh worker-process
    service (empty _pending_jobs, same durable store) records completion and
    gets a truthy job back — NOT None — and the API re-reads 'completed'."""
    api_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))
    worker_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))

    model_id = uuid4()
    job = await api_service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    # The worker process never saw the API's in-process dict.
    assert worker_service._pending_jobs == {}

    recorded = await worker_service.complete_retraining(
        job_id=job.job_id,
        new_model_id="twin-model-new",
        fidelity_after=0.77,
        success=True,
    )

    # Pre-#549 this returned None (job absent from the worker's dict). Now the
    # durable store carries it, so the completion IS recorded.
    assert recorded is not None
    assert recorded.status == TwinRetrainingStatus.COMPLETED
    assert recorded.new_model_id == "twin-model-new"
    assert recorded.fidelity_after == 0.77

    # The API process re-reads the worker's durable completion.
    seen_by_api = await api_service.get_job_status(job.job_id)
    assert seen_by_api is not None
    assert seen_by_api.status == TwinRetrainingStatus.COMPLETED
    assert seen_by_api.fidelity_after == 0.77


@pytest.mark.asyncio
async def test_complete_returns_none_when_job_truly_unknown(fake_supabase) -> None:
    """A job id that was never created is absent from BOTH the in-process dict
    and the durable store → complete_retraining returns None (honest: nothing to
    record), preserving fail-closed for genuinely-unknown jobs."""
    service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))

    result = await service.complete_retraining(
        job_id="never-created",
        new_model_id="x",
        fidelity_after=0.9,
        success=True,
    )
    assert result is None


@pytest.mark.asyncio
async def test_complete_fails_closed_when_durable_store_inert() -> None:
    """#549 codex HIGH #2: when a durable job store IS wired but the durable write
    does not record it (inert / unconfigured store), complete_retraining must NOT
    mask the failure with the in-process job — it returns None (fail closed), so a
    worker never reports a false 'completed' without a durable record."""
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    async def _raise():
        raise RuntimeError("Supabase not configured")

    with patch("src.memory.services.factories.get_async_supabase_client", _raise):
        service = _service(TwinRetrainingJobRepository())  # wired but client never resolves
        model_id = uuid4()
        job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)
        # The job IS in THIS instance's in-process dict...
        assert job.job_id in service._pending_jobs
        # ...but the durable store could not record it → success is NOT reported.
        result = await service.complete_retraining(
            job.job_id, new_model_id="m", fidelity_after=0.9, success=True
        )
        assert result is None

        # codex iter-1 HIGH: the status API must NOT surface a false 'completed'
        # via the in-process dict fallback after a failed durable write — the
        # in-process copy was forced to a non-success state with no metric.
        seen = await service.get_job_status(job.job_id)
        assert seen is not None
        assert seen.status != TwinRetrainingStatus.COMPLETED
        assert seen.fidelity_after is None


@pytest.mark.asyncio
async def test_complete_with_success_false_writes_no_metric(fake_supabase) -> None:
    """#549 codex HIGH #3: complete_retraining(success=False) records the failed
    status but writes NO fidelity_after / new_model_id — the #548 invariant (a
    non-success completion must never surface a metric), enforced in-memory AND
    in the durable record."""
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    repo = TwinRetrainingJobRepository(supabase_client=fake_supabase)
    service = _service(repo)
    model_id = uuid4()
    job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    result = await service.complete_retraining(
        job.job_id, new_model_id="should-not-persist", fidelity_after=0.99, success=False
    )
    assert result is not None
    assert result.status == TwinRetrainingStatus.FAILED
    assert result.fidelity_after is None
    assert result.new_model_id is None

    durable = await repo.get_job(job.job_id)
    assert durable is not None
    assert durable.status == "failed"
    assert durable.fidelity_after is None
    assert durable.new_model_id is None


@pytest.mark.asyncio
async def test_complete_then_fail_clears_stale_metric(fake_supabase) -> None:
    """#549 codex iter-1 MEDIUM: a complete->fail transition must leave NO metric.
    fail_retraining clears any fidelity_after / new_model_id from a prior
    completion, so the #548 invariant holds across state transitions, not just for
    fresh pending jobs — in BOTH the returned job and the durable record."""
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    repo = TwinRetrainingJobRepository(supabase_client=fake_supabase)
    service = _service(repo)
    model_id = uuid4()
    job = await service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    completed = await service.complete_retraining(
        job.job_id, new_model_id="m-1", fidelity_after=0.8, success=True
    )
    assert completed is not None and completed.fidelity_after == 0.8

    # A later failure must clear the stale metric.
    failed = await service.fail_retraining(job.job_id, "post-hoc invalidation")
    assert failed is not None
    assert failed.status == TwinRetrainingStatus.FAILED
    assert failed.fidelity_after is None
    assert failed.new_model_id is None

    durable = await repo.get_job(job.job_id)
    assert durable is not None
    assert durable.status == "failed"
    assert durable.fidelity_after is None
    assert durable.new_model_id is None


@pytest.mark.asyncio
async def test_fail_retraining_records_durably_without_metric(fake_supabase) -> None:
    """fail_retraining persists status=failed + reason across the boundary and
    writes NO fidelity metric (the #548 fail-closed invariant)."""
    api_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))
    worker_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))

    model_id = uuid4()
    job = await api_service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    failed = await worker_service.fail_retraining(job.job_id, "train raised ValueError")
    assert failed is not None
    assert failed.status == TwinRetrainingStatus.FAILED
    assert failed.error_message == "train raised ValueError"
    assert failed.fidelity_after is None

    seen_by_api = await api_service.get_job_status(job.job_id)
    assert seen_by_api is not None
    assert seen_by_api.status == TwinRetrainingStatus.FAILED
    assert seen_by_api.fidelity_after is None


@pytest.mark.asyncio
async def test_cancel_via_durable_store(fake_supabase) -> None:
    """A PENDING job created by the API process can be cancelled through the
    durable store by another instance, and the cancellation is re-readable."""
    api_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))
    worker_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))

    model_id = uuid4()
    job = await api_service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    cancelled = await worker_service.cancel_retraining(job.job_id, "no longer needed")
    assert cancelled is not None
    assert cancelled.status == TwinRetrainingStatus.CANCELLED

    seen_by_api = await api_service.get_job_status(job.job_id)
    assert seen_by_api is not None
    assert seen_by_api.status == TwinRetrainingStatus.CANCELLED


def test_fidelity_tracker_auto_trigger_wires_durable_service() -> None:
    """#549: enabling auto-retraining without an injected service wires the
    DURABLE service so auto-triggered jobs are persisted to the shared store and
    survive the Celery worker boundary (otherwise the worker fails closed)."""
    from src.digital_twin.fidelity_tracker import FidelityTracker

    tracker = FidelityTracker(auto_trigger_retraining=True)
    assert isinstance(tracker._retraining_service, TwinRetrainingService)
    assert tracker._retraining_service.job_repository is not None


def test_fidelity_tracker_default_has_no_retraining_service() -> None:
    """Backward-compat: with auto-retraining off, no service is wired (unchanged)."""
    from src.digital_twin.fidelity_tracker import FidelityTracker

    tracker = FidelityTracker()
    assert tracker._retraining_service is None


def test_fidelity_tracker_respects_injected_service() -> None:
    """An explicitly injected service is never overridden by the durable default."""
    from src.digital_twin.fidelity_tracker import FidelityTracker

    injected = TwinRetrainingService()
    tracker = FidelityTracker(retraining_service=injected, auto_trigger_retraining=True)
    assert tracker._retraining_service is injected


@pytest.mark.asyncio
async def test_worker_records_completion_across_boundary_via_durable_store(
    tmp_path, fake_supabase
) -> None:
    """End-to-end #549 across the worker boundary (no live DB/Celery): the API
    service triggers a job into the shared durable store; a FRESH worker-process
    service bound to the SAME store runs the real worker entrypoint
    (_execute_real_twin_retraining), persists the model, records the real metric,
    and the API process re-reads 'completed'. Pre-#549 this returned a non-success
    status because the worker's in-process store was empty."""
    from src.tasks.ab_testing_tasks import _execute_real_twin_retraining

    # Two services backed by the SAME durable store = the worker boundary.
    api_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))
    worker_service = _service(TwinRetrainingJobRepository(supabase_client=fake_supabase))

    model_id = uuid4()
    job = await api_service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)
    assert worker_service._pending_jobs == {}  # worker never saw the API's dict

    # Patch the model lookup + durable model persistence + the trainer (the focus
    # here is the cross-process job recording, not sklearn).
    model_row = {
        "model_id": str(model_id),
        "model_name": "kisqali_hcp",
        "twin_type": "hcp",
        "brand": "Kisqali",
        "feature_columns": ["decile", "digital_engagement_score"],
        "target_columns": ["prescribing_change"],
        "training_config": {"algorithm": "random_forest"},
    }
    repo = MagicMock()
    repo.get_model = AsyncMock(return_value=model_row)
    repo.save_model = AsyncMock(return_value=uuid4())
    real_metrics = MagicMock()
    real_metrics.r2_score = 0.73
    real_metrics.model_id = uuid4()
    gen = MagicMock()
    gen.train = MagicMock(return_value=real_metrics)
    gen.model = object()

    csv = tmp_path / "cohort.csv"
    pd.DataFrame({"decile": [1, 2], "digital_engagement_score": [0.1, 0.2]}).to_csv(
        csv, index=False
    )
    cfg = {"data_source": str(csv), "target_column": "prescribing_change"}

    with (
        patch("src.digital_twin.twin_repository.TwinModelRepository", return_value=repo),
        patch("src.tasks.ab_testing_tasks.TwinGenerator", MagicMock(return_value=gen)),
    ):
        out = await _execute_real_twin_retraining(
            retraining_job_id=job.job_id,
            model_id=str(model_id),
            training_config=cfg,
            service=worker_service,
        )

    assert out["status"] == "completed"
    assert out["validation_r2"] == 0.73

    # The API process re-reads the worker's durable completion.
    seen_by_api = await api_service.get_job_status(job.job_id)
    assert seen_by_api is not None
    assert seen_by_api.status == TwinRetrainingStatus.COMPLETED
    assert seen_by_api.fidelity_after == 0.73


@pytest.mark.asyncio
async def test_no_job_repository_is_legacy_dict_behavior() -> None:
    """Backward-compat guard: with no job_repository the service uses ONLY the
    in-process dict — a fresh instance cannot see another's job (the pre-#549
    behavior, intact when durable storage is not wired)."""
    api_service = TwinRetrainingService()  # no job_repository
    worker_service = TwinRetrainingService()

    model_id = uuid4()
    job = await api_service.trigger_retraining(model_id, TwinTriggerReason.MANUAL)

    # Same-process completion still works (dict path).
    same = await api_service.complete_retraining(
        job.job_id, new_model_id="m", fidelity_after=0.8, success=True
    )
    assert same is not None and same.status == TwinRetrainingStatus.COMPLETED

    # Cross-instance is NOT shared without a durable store → None.
    cross = await worker_service.complete_retraining(
        job.job_id, new_model_id="m", fidelity_after=0.8, success=True
    )
    assert cross is None
