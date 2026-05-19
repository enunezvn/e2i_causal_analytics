"""Unit tests for sentinel action handlers (#375 item 3 + 4).

The plan (§3.8) specifies four action handlers:
* ``rerun_all_active_cohorts``     — fired by ``data_drop``
* ``notify_and_queue_reanalysis``  — fired by ``staleness_threshold``
* ``flag_for_review``              — fired by ``cohort_drift``
* ``run_full_consolidation``       — fired by ``schedule``

All four publish to the Redis pub/sub channel ``e2i:alerts`` (the
``notify_and_queue_reanalysis`` and ``flag_for_review`` actions explicitly so
per plan; the other two implicitly via this audit-trail-friendly shape).

The actions are Celery tasks registered under ``src.tasks.sentinel_actions``.
They take ``(sentinel_id, brands, trigger_data)`` per plan and return a
small dict summary for observability.
"""

from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tasks.sentinel_actions import (
    ALERTS_CHANNEL,
    flag_for_review,
    notify_and_queue_reanalysis,
    publish_alert,
    rerun_all_active_cohorts,
    run_full_consolidation,
)

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------


def test_alerts_channel_constant():
    """Plan §3.8 wires actions to publish on ``e2i:alerts``."""
    assert ALERTS_CHANNEL == "e2i:alerts"


# ---------------------------------------------------------------------------
# publish_alert helper (shared by all four actions)
# ---------------------------------------------------------------------------


class _CapturingRedis:
    def __init__(self) -> None:
        self.published: List[tuple[str, str]] = []

    async def publish(self, channel: str, payload: str) -> int:
        self.published.append((channel, payload))
        return 1


@pytest.fixture
def fake_redis() -> _CapturingRedis:
    return _CapturingRedis()


@pytest.fixture(autouse=True)
def patch_redis(fake_redis):
    # publish_alert lazy-imports get_redis_client from the factory module,
    # so we patch the canonical source rather than a re-export.
    with patch("src.memory.services.factories.get_redis_client", return_value=fake_redis):
        yield fake_redis


@pytest.mark.asyncio
async def test_publish_alert_writes_json_to_e2i_alerts_channel(
    fake_redis: _CapturingRedis,
):
    """The helper publishes the payload as a JSON string on ``e2i:alerts``."""
    await publish_alert({"type": "test", "brand": "Kisqali", "data": [1, 2, 3]})
    assert len(fake_redis.published) == 1
    channel, raw = fake_redis.published[0]
    assert channel == "e2i:alerts"
    decoded = json.loads(raw)
    assert decoded["type"] == "test"
    assert decoded["brand"] == "Kisqali"
    assert decoded["data"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_publish_alert_swallow_redis_failure(fake_redis: _CapturingRedis):
    """Alert publication is best-effort — a Redis outage MUST NOT break the action."""

    async def boom(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("redis down")

    fake_redis.publish = boom  # type: ignore[method-assign]
    # No exception should leak.
    await publish_alert({"type": "test"})


# ---------------------------------------------------------------------------
# rerun_all_active_cohorts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerun_all_active_cohorts_publishes_per_brand(
    fake_redis: _CapturingRedis,
):
    summary = await rerun_all_active_cohorts(
        sentinel_id="s-1",
        brands=["Kisqali", "Pluvicto"],
        trigger_data={"source": "optum_cdm", "refreshed_at": "2026-05-19T00:00:00Z"},
    )
    assert summary["brands_dispatched"] == 2
    # One alert published containing both brands so subscribers see them atomically.
    assert any(
        '"data_refresh"' in raw and '"Kisqali"' in raw and '"Pluvicto"' in raw
        for _ch, raw in fake_redis.published
    )


@pytest.mark.asyncio
async def test_rerun_all_active_cohorts_empty_brands_noop(fake_redis: _CapturingRedis):
    summary = await rerun_all_active_cohorts(
        sentinel_id="s-1",
        brands=[],
        trigger_data={},
    )
    assert summary["brands_dispatched"] == 0
    assert fake_redis.published == []


# ---------------------------------------------------------------------------
# notify_and_queue_reanalysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_and_queue_reanalysis_publishes_staleness_alert(
    fake_redis: _CapturingRedis,
):
    stale_findings = [
        {"finding_id": "f1", "brand": "Kisqali", "staleness_score": 0.95},
        {"finding_id": "f2", "brand": "Kisqali", "staleness_score": 0.80},
    ]
    with patch("src.tasks.sentinel_actions.celery_app.send_task") as mock_send:
        summary = await notify_and_queue_reanalysis(
            sentinel_id="s-2",
            brands=["Kisqali"],
            trigger_data={"stale_findings": stale_findings},
        )
    assert summary["stale_findings_count"] == 2
    # #378: ``queued_for_reanalysis`` now equals the count of actual Celery
    # ``send_task`` calls — one per finding in the top-5 cap.
    assert summary["notified_for_reanalysis"] == 2
    assert summary["queued_for_reanalysis"] == 2, (
        "queued_for_reanalysis must reflect actual send_task calls per #378"
    )
    # Verify the Celery enqueue contract.
    assert mock_send.call_count == 2
    for call in mock_send.call_args_list:
        args, kwargs = call.args, call.kwargs
        assert args[0] == "src.tasks.insight_lifecycle_tasks.reanalyze_finding"
        send_args = kwargs["args"]
        assert send_args[1] == "Kisqali"  # brand
        assert send_args[0] in ("f1", "f2")  # finding_id
        assert kwargs["kwargs"] == {"triggered_by": "sentinel:staleness"}

    found = False
    for _ch, raw in fake_redis.published:
        if "staleness_alert" in raw:
            found = True
            payload = json.loads(raw)
            assert payload["type"] == "staleness_alert"
            assert payload["brands"] == ["Kisqali"]
            assert len(payload["findings"]) == 2
    assert found, "expected at least one staleness_alert publication"


@pytest.mark.asyncio
async def test_notify_and_queue_reanalysis_caps_reanalysis_at_5(
    fake_redis: _CapturingRedis,
):
    """Plan §3.8 caps re-analysis to top-5 most-stale findings.

    #378: the cap applies to both the notify count AND the queued count —
    we enqueue ``reanalyze_finding`` for each of the top-5, no more.
    """
    stale = [{"finding_id": f"f{i}", "brand": "Kisqali", "staleness_score": 0.9} for i in range(20)]
    with patch("src.tasks.sentinel_actions.celery_app.send_task") as mock_send:
        summary = await notify_and_queue_reanalysis(
            sentinel_id="s-2",
            brands=["Kisqali"],
            trigger_data={"stale_findings": stale},
        )
    assert summary["notified_for_reanalysis"] == 5
    assert summary["queued_for_reanalysis"] == 5
    assert mock_send.call_count == 5


@pytest.mark.asyncio
async def test_notify_and_queue_reanalysis_return_contract_is_honest(
    fake_redis: _CapturingRedis,
):
    """The return contract carries BOTH fields so downstream observers can
    disambiguate "intent to reanalyze" (notified) from "actually enqueued"
    (queued). #378 makes them numerically equal for the happy path; the
    field separation stays for forward compatibility with a future degraded
    path where send_task fails partway through.
    """
    stale = [{"finding_id": f"f{i}", "brand": "Kisqali", "staleness_score": 0.9} for i in range(3)]
    with patch("src.tasks.sentinel_actions.celery_app.send_task"):
        summary = await notify_and_queue_reanalysis(
            sentinel_id="s-honest",
            brands=["Kisqali"],
            trigger_data={"stale_findings": stale},
        )
    # Both fields present so downstream observers can disambiguate
    # "intent to reanalyze" vs "actually enqueued".
    assert "notified_for_reanalysis" in summary
    assert "queued_for_reanalysis" in summary
    assert summary["notified_for_reanalysis"] == 3
    assert summary["queued_for_reanalysis"] == 3


@pytest.mark.asyncio
async def test_notify_and_queue_reanalysis_send_task_failure_keeps_handler_alive(
    fake_redis: _CapturingRedis,
):
    """Broker outage on a per-finding ``send_task`` MUST NOT crash the
    handler — the Redis alert publish is the cross-process audit trail and
    the queued counter reflects only successful enqueues. This mirrors the
    sentinel dispatcher's best-effort send_task pattern (registry.py:680).
    """
    stale = [{"finding_id": f"f{i}", "brand": "Kisqali", "staleness_score": 0.9} for i in range(3)]
    with patch(
        "src.tasks.sentinel_actions.celery_app.send_task",
        side_effect=ConnectionError("broker down"),
    ) as mock_send:
        summary = await notify_and_queue_reanalysis(
            sentinel_id="s-broker-down",
            brands=["Kisqali"],
            trigger_data={"stale_findings": stale},
        )
    # All three were attempted.
    assert mock_send.call_count == 3
    # Notified count reflects what we logged (top-5 of 3 == 3).
    assert summary["notified_for_reanalysis"] == 3
    # Queued count reflects ACTUAL successful enqueues — zero, all failed.
    assert summary["queued_for_reanalysis"] == 0


@pytest.mark.asyncio
async def test_notify_and_queue_reanalysis_uses_finding_brand_when_present(
    fake_redis: _CapturingRedis,
):
    """The per-finding brand (carried in the match dict from the sentinel
    evaluator) is what gets passed to ``reanalyze_finding``. This matters
    when a sentinel watches a brand-agnostic table but the finding row
    itself carries a specific brand.
    """
    stale = [
        {"finding_id": "f1", "brand": "Kisqali", "staleness_score": 0.9},
        {"finding_id": "f2", "brand": "Pluvicto", "staleness_score": 0.8},
    ]
    with patch("src.tasks.sentinel_actions.celery_app.send_task") as mock_send:
        summary = await notify_and_queue_reanalysis(
            sentinel_id="s-multi-brand",
            brands=["all"],
            trigger_data={"stale_findings": stale},
        )
    assert summary["queued_for_reanalysis"] == 2
    brands_seen = {call.kwargs["args"][1] for call in mock_send.call_args_list}
    assert brands_seen == {"Kisqali", "Pluvicto"}


@pytest.mark.asyncio
async def test_notify_and_queue_reanalysis_programming_errors_propagate(
    fake_redis: _CapturingRedis,
):
    """M1 (codex iter-0): the per-finding ``send_task`` catch is narrowed
    to broker/transport exceptions only. Programming errors (``TypeError``,
    ``AttributeError``, ``KeyError``) MUST propagate so they surface in
    error tracking instead of being silently indistinguishable from a
    broker outage.

    Pre-fix: a ``TypeError`` raised by send_task (e.g., bad task-name shape
    or serialization issue) would be swallowed by ``except Exception`` and
    the queued_count would remain 0, leaving operators unable to
    distinguish "broker down" from "programmer broke the dispatch
    contract".
    """
    stale = [{"finding_id": "f1", "brand": "Kisqali", "staleness_score": 0.9}]
    with patch(
        "src.tasks.sentinel_actions.celery_app.send_task",
        side_effect=TypeError("send_task signature mismatch"),
    ):
        with pytest.raises(TypeError, match="signature mismatch"):
            await notify_and_queue_reanalysis(
                sentinel_id="s-bad-shape",
                brands=["Kisqali"],
                trigger_data={"stale_findings": stale},
            )


# ---------------------------------------------------------------------------
# flag_for_review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_for_review_publishes_cohort_drift(fake_redis: _CapturingRedis):
    summary = await flag_for_review(
        sentinel_id="s-3",
        brands=["Pluvicto"],
        trigger_data={"drift_data": {"baseline": 1000, "current": 1080, "shift": 0.08}},
    )
    assert summary["flagged"] is True
    assert any("cohort_drift" in raw for _ch, raw in fake_redis.published)


# ---------------------------------------------------------------------------
# run_full_consolidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_full_consolidation_invokes_consolidator(fake_redis: _CapturingRedis):
    """The action triggers a full consolidator run and publishes a heartbeat."""

    mock_consolidator = AsyncMock()
    fake_result = MagicMock()
    fake_result.promoted_to_semantic = 3
    fake_result.promoted_to_procedural = 1
    fake_result.causal_paths_examined = 17
    fake_result.procedural_examined = 5
    fake_result.errors = []
    fake_result.by_brand = {"Kisqali": {"semantic": 3}}
    mock_consolidator.return_value = fake_result

    with patch(
        "src.memory.lifecycle.consolidator.consolidate_insights",
        new=mock_consolidator,
    ):
        summary = await run_full_consolidation(
            sentinel_id="s-4",
            brands=["all"],
            trigger_data={},
        )
    assert summary["promoted_to_semantic"] == 3
    assert summary["promoted_to_procedural"] == 1
    # Plan §3.8 publishes a heartbeat-style alert so subscribers (CopilotKit)
    # know a full consolidation just happened.
    assert any("full_consolidation_run" in raw for _ch, raw in fake_redis.published)


# ---------------------------------------------------------------------------
# Action handlers are registered as Celery tasks (sanity)
# ---------------------------------------------------------------------------


def test_all_four_actions_are_celery_tasks():
    """The four actions are reachable via ``celery_app.tasks[name]``.

    This is a placeholder check — we don't actually need a worker for unit
    tests, but the registration must be present so ``celery worker`` can
    enqueue them and the dispatcher's ``dispatch_agent → Celery`` bridge
    can route to them by name.
    """
    from src.tasks.sentinel_actions import (  # noqa: F401
        celery_flag_for_review,
        celery_notify_and_queue_reanalysis,
        celery_rerun_all_active_cohorts,
        celery_run_full_consolidation,
    )

    # Each ``celery_*`` is a registered Celery task wrapping the async helper.
    # The wrapping is the side-channel that lets a worker enqueue/execute them.
    for name in (
        "src.tasks.sentinel_actions.rerun_all_active_cohorts",
        "src.tasks.sentinel_actions.notify_and_queue_reanalysis",
        "src.tasks.sentinel_actions.flag_for_review",
        "src.tasks.sentinel_actions.run_full_consolidation",
    ):
        from src.workers.celery_app import celery_app

        assert name in celery_app.tasks, f"Celery task {name} not registered"
