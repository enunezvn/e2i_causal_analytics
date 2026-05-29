"""#549: TwinRetrainingJobRepository — durable, shared twin-retraining job store.

Before #549 the twin retraining job lived ONLY in the in-process
``TwinRetrainingService._pending_jobs`` dict, so a Celery worker (a separate
process from the API that queued the job) had an empty store and
``complete_retraining`` returned ``None`` — the completion was never recorded.

This repository is the twin analogue of ``RetrainingHistoryRepository``
(``src/repositories/drift_monitoring.py``): a Supabase-backed CRUD store so a
job created in one process is found + updated in another and re-read by the
first. The faithful in-memory ``FakeSupabaseClient`` (see ``conftest.py``)
exercises the repo's real delegation logic without a live DB; the cross-instance
test reproduces the worker boundary (separate repo instance, SAME backing store).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------- #
# Repository round-trip behavior
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_create_then_get_roundtrip(fake_supabase) -> None:
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    repo = TwinRetrainingJobRepository(supabase_client=fake_supabase)

    await repo.create_job(
        job_id="job-1",
        model_id="model-abc",
        trigger_reason="manual",
        status="pending",
        fidelity_before=0.62,
        training_config={"data_source": "cohort.csv", "target_column": "y"},
    )

    got = await repo.get_job("job-1")
    assert got is not None
    assert got.id == "job-1"
    assert got.model_id == "model-abc"
    assert got.status == "pending"
    assert got.fidelity_before == 0.62
    assert got.training_config["data_source"] == "cohort.csv"
    # No completion metric yet — must be honest None, not a fabricated 0.0.
    assert got.fidelity_after is None


@pytest.mark.asyncio
async def test_complete_job_records_status_and_real_metric(fake_supabase) -> None:
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    repo = TwinRetrainingJobRepository(supabase_client=fake_supabase)
    await repo.create_job(
        job_id="job-2",
        model_id="model-abc",
        trigger_reason="fidelity_degradation",
        status="pending",
        fidelity_before=0.5,
        training_config={},
    )

    await repo.complete_job("job-2", new_model_id="model-new", fidelity_after=0.741, success=True)

    got = await repo.get_job("job-2")
    assert got is not None
    assert got.status == "completed"
    assert got.new_model_id == "model-new"
    assert got.fidelity_after == 0.741
    assert got.completed_at is not None


@pytest.mark.asyncio
async def test_fail_job_records_error_and_writes_no_metric(fake_supabase) -> None:
    """Fail-closed (#548 invariant): a failed job records the reason and leaves
    fidelity_after None rather than a fabricated 0.0 misread as a poor score."""
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    repo = TwinRetrainingJobRepository(supabase_client=fake_supabase)
    await repo.create_job(
        job_id="job-3",
        model_id="model-abc",
        trigger_reason="manual",
        status="pending",
        fidelity_before=0.5,
        training_config={},
    )

    await repo.fail_job("job-3", error_message="train raised ValueError")

    got = await repo.get_job("job-3")
    assert got is not None
    assert got.status == "failed"
    assert got.error_message == "train raised ValueError"
    assert got.fidelity_after is None


@pytest.mark.asyncio
async def test_two_repo_instances_sharing_store_see_same_job(fake_supabase) -> None:
    """THE cross-process semantics: a job created via instance A (the API
    process) is found + completed via instance B (a fresh worker-process repo)
    and the completion is re-readable via A. Same DB, different instances —
    exactly what the Celery worker boundary is."""
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    api_repo = TwinRetrainingJobRepository(supabase_client=fake_supabase)
    worker_repo = TwinRetrainingJobRepository(supabase_client=fake_supabase)

    # API process creates the job.
    await api_repo.create_job(
        job_id="job-xproc",
        model_id="model-abc",
        trigger_reason="manual",
        status="pending",
        fidelity_before=0.5,
        training_config={},
    )

    # Worker process (a DIFFERENT instance) finds it and records completion.
    found = await worker_repo.get_job("job-xproc")
    assert found is not None and found.status == "pending"
    await worker_repo.complete_job(
        "job-xproc", new_model_id="model-new", fidelity_after=0.8, success=True
    )

    # API process re-reads the worker's durable update.
    refetched = await api_repo.get_job("job-xproc")
    assert refetched is not None
    assert refetched.status == "completed"
    assert refetched.new_model_id == "model-new"
    assert refetched.fidelity_after == 0.8


# --------------------------------------------------------------------------- #
# Client wiring — the linchpin that makes cross-process sharing work
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_bare_repo_lazily_binds_async_supabase_client(fake_supabase) -> None:
    """A bare ``TwinRetrainingJobRepository()`` lazily binds the process-shared
    ``get_async_supabase_client()`` (the ASYNC client ``BaseRepository`` awaits)
    on first async use — so the API and a Celery worker share the SAME store with
    no explicit wiring, AND the repo works against the real async-client shape
    (a sync ``get_supabase()`` client would raise on ``await .execute()``)."""
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    async def _fake_async_client():
        return fake_supabase

    with patch("src.memory.services.factories.get_async_supabase_client", _fake_async_client):
        repo = TwinRetrainingJobRepository()
        assert repo.client is None  # not resolved until first async use
        await repo.create_job(
            job_id="j",
            model_id="m",
            trigger_reason="manual",
            status="pending",
            fidelity_before=0.0,
            training_config={},
        )
        assert repo.client is fake_supabase  # lazily bound + usable
        got = await repo.get_job("j")
        assert got is not None and got.id == "j"


@pytest.mark.asyncio
async def test_no_client_is_inert_not_an_error() -> None:
    """Unconfigured env (``get_async_supabase_client`` raises) → inert repo:
    get_job returns None and create_job returns the unsaved record, never raises."""
    from src.digital_twin.twin_repository import TwinRetrainingJobRepository

    async def _raise():
        raise RuntimeError("Supabase not configured")

    with patch("src.memory.services.factories.get_async_supabase_client", _raise):
        repo = TwinRetrainingJobRepository()
        rec = await repo.create_job(
            job_id="job-x",
            model_id="m",
            trigger_reason="manual",
            status="pending",
            fidelity_before=0.0,
            training_config={},
        )
        assert rec.id == "job-x"  # record returned even when not persisted
        assert await repo.get_job("job-x") is None  # inert read
