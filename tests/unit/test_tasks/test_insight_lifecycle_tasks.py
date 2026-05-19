"""Unit tests for the insight-lifecycle Celery tasks (#378).

This module covers the ``reanalyze_finding`` task added by #378, which the
``notify_and_queue_reanalysis`` sentinel action handler now enqueues per
finding (top-5, plan §3.8).

Design (per #378 round-1 design step, option (a) honest minimal):
    * The new task is a thin wrapper that publishes a brand-namespaced
      ``reanalysis:e2i:{brand}`` Redis pub/sub signal carrying the
      ``finding_id`` + originating ``triggered_by`` context. Downstream
      orchestrator consumers (cascade re-run, single-finding re-evaluation)
      subscribe to the channel.
    * Honest scope: this does NOT itself re-run a Tier-3 analysis pipeline
      (that surface is still moving under #237 / #373 follow-ups). It
      provides the durable Celery hand-off the sentinel action needs to
      enqueue against, and the canonical signal channel downstream picks up.

The existing tests in ``test_sentinel_actions.py`` are updated separately to
reflect the new ``queued_for_reanalysis`` contract (now equal to the count of
``send_task`` calls, not 0).
"""

from __future__ import annotations

import json
import logging
from typing import Any, List
from unittest.mock import patch

import pytest

from src.tasks.insight_lifecycle_tasks import reanalyze_finding


class _CapturingRedis:
    """Test double for ``redis.asyncio.Redis``; records publish calls."""

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
    with patch("src.memory.services.factories.get_redis_client", return_value=fake_redis):
        yield fake_redis


# ---------------------------------------------------------------------------
# reanalyze_finding registration + signature smoke
# ---------------------------------------------------------------------------


def test_reanalyze_finding_task_smoke():
    """The task is registered with the expected Celery name + signature.

    Contract from ``notify_and_queue_reanalysis`` (sentinel_actions.py):
        celery_app.send_task(
            "src.tasks.insight_lifecycle_tasks.reanalyze_finding",
            args=[finding_id, brand],
            kwargs={"triggered_by": "sentinel:staleness"},
        )

    The task name must match. The callable must accept (finding_id, brand)
    positionally and ``triggered_by`` as a keyword argument.
    """
    # Importing the package triggers worker-discovery registration via
    # src/tasks/__init__.py — the same path Celery boot exercises.
    import src.tasks  # noqa: F401
    from src.workers.celery_app import celery_app

    expected_name = "src.tasks.insight_lifecycle_tasks.reanalyze_finding"
    assert expected_name in celery_app.tasks, (
        f"Celery task {expected_name} not registered. The sentinel action "
        f"handler's send_task call would dead-letter."
    )


def test_reanalyze_finding_callable_with_expected_signature():
    """The task callable accepts (finding_id, brand, *, triggered_by)."""
    assert callable(reanalyze_finding)
    # Smoke: can we call it with the documented signature? It must not raise
    # TypeError for argument mismatch (return value is verified elsewhere).
    result = reanalyze_finding.run(  # type: ignore[attr-defined]
        "f-test-1",
        "Kisqali",
        triggered_by="sentinel:staleness",
    )
    assert isinstance(result, dict)
    assert "finding_id" in result
    assert "brand" in result
    assert "triggered_by" in result


# ---------------------------------------------------------------------------
# reanalyze_finding behaviour
# ---------------------------------------------------------------------------


def test_reanalyze_finding_publishes_reanalysis_signal(
    fake_redis: _CapturingRedis,
):
    """The task publishes a per-brand ``reanalysis:e2i:{brand}`` signal so
    downstream orchestrator/agent consumers can pick up the request.
    """
    result = reanalyze_finding.run(  # type: ignore[attr-defined]
        "f-123",
        "Kisqali",
        triggered_by="sentinel:staleness",
    )

    assert result["finding_id"] == "f-123"
    assert result["brand"] == "Kisqali"
    assert result["triggered_by"] == "sentinel:staleness"
    assert result["signal_published"] is True

    # The publish hit the brand-scoped reanalysis channel.
    channels = {channel for channel, _payload in fake_redis.published}
    assert "reanalysis:e2i:Kisqali" in channels, (
        f"expected publish on 'reanalysis:e2i:Kisqali', got channels={channels}"
    )

    # Payload carries the necessary metadata.
    payload_str = next(
        raw for channel, raw in fake_redis.published if channel == "reanalysis:e2i:Kisqali"
    )
    payload = json.loads(payload_str)
    assert payload["type"] == "reanalysis_requested"
    assert payload["finding_id"] == "f-123"
    assert payload["brand"] == "Kisqali"
    assert payload["triggered_by"] == "sentinel:staleness"


def test_reanalyze_finding_loads_finding_or_raises():
    """The task validates inputs — empty finding_id or brand raises ValueError
    BEFORE any side effects, so the caller (sentinel action handler) gets a
    clear signal that the dispatch is malformed and won't silently no-op.
    """
    with pytest.raises(ValueError, match="finding_id"):
        reanalyze_finding.run(  # type: ignore[attr-defined]
            "",
            "Kisqali",
            triggered_by="sentinel:staleness",
        )

    with pytest.raises(ValueError, match="brand"):
        reanalyze_finding.run(  # type: ignore[attr-defined]
            "f-1",
            "",
            triggered_by="sentinel:staleness",
        )


def test_reanalyze_finding_swallows_redis_outage(
    fake_redis: _CapturingRedis,
    caplog: pytest.LogCaptureFixture,
):
    """Redis publish is best-effort — a broker outage MUST NOT crash the
    task. The result reports ``signal_published=False`` so observability
    can detect a degraded run.
    """

    async def boom(*args: Any, **kwargs: Any) -> int:
        raise ConnectionError("redis down")

    fake_redis.publish = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        result = reanalyze_finding.run(  # type: ignore[attr-defined]
            "f-9",
            "Kisqali",
            triggered_by="sentinel:staleness",
        )
    assert result["signal_published"] is False
    # The failure is logged at WARNING level so SREs can see it.
    assert any("reanaly" in rec.message.lower() for rec in caplog.records)


def test_reanalyze_finding_programming_errors_propagate(
    fake_redis: _CapturingRedis,
):
    """L2 (codex iter-0): the publish path's catch is narrowed to
    ConnectionError + RedisConnectionError only. Programming errors
    (TypeError, AttributeError, KeyError) from an unexpected publish
    failure MUST propagate so they surface in error tracking and the
    Celery task_failure handler can route them to the dead-letter queue.

    Pre-fix: a broad ``except Exception`` fallback swallowed these
    silently and the task returned ``signal_published=False`` —
    indistinguishable from a real broker outage.
    """

    async def boom(*args: Any, **kwargs: Any) -> int:
        # An unexpected exception class — NOT ConnectionError, NOT
        # RuntimeError. A real client-shape mismatch could throw this.
        raise TypeError("unexpected publish shape — programming error")

    fake_redis.publish = boom  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="programming error"):
        reanalyze_finding.run(  # type: ignore[attr-defined]
            "f-99",
            "Kisqali",
            triggered_by="sentinel:staleness",
        )


def test_publish_reanalysis_signal_swallows_redis_py_connection_error(
    fake_redis: _CapturingRedis,
    caplog: pytest.LogCaptureFixture,
):
    """H2 (codex iter-1): ``redis.exceptions.ConnectionError`` does NOT
    inherit from builtin ``ConnectionError``. The iter-0 narrow catch
    ``(ConnectionError, RuntimeError)`` would NOT match a redis-py
    transport error, defeating the broker-outage best-effort contract.

    Pre-fix: a real redis transport outage escaped the catch and
    propagated to the Celery task wrapper, indistinguishable from a
    programming bug.

    Post-fix: the catch tuple is ``(ConnectionError, RedisConnectionError)``
    so redis-py transport errors are correctly classified as broker
    outage. Task returns ``signal_published=False`` without crashing.
    """
    from redis.exceptions import ConnectionError as RedisConnectionError

    async def boom(*args: Any, **kwargs: Any) -> int:
        # redis-py's canonical transport-failure class — what
        # `redis.asyncio.Redis.publish` raises on a real network outage.
        raise RedisConnectionError("redis transport down")

    fake_redis.publish = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        result = reanalyze_finding.run(  # type: ignore[attr-defined]
            "f-redis-down",
            "Kisqali",
            triggered_by="sentinel:staleness",
        )
    # Caught + classified as degraded (best-effort), NOT propagated.
    assert result["signal_published"] is False
    assert any(
        "reanalysis-signal publish failed" in rec.message for rec in caplog.records
    )


def test_publish_reanalysis_signal_does_not_catch_runtime_error(
    fake_redis: _CapturingRedis,
):
    """M3 (codex iter-1): ``RuntimeError`` was dropped from the catch
    tuple — redis-py's only ``raise RuntimeError`` sites are on the
    PubSub-CONSUMER side (subscribe/psubscribe lifecycle gates), not on
    publish. Keeping it would have masked real programming bugs.

    Pre-fix: ``RuntimeError`` in the catch tuple was unjustified
    over-catch. Post-fix: a stray ``RuntimeError`` propagates as a
    programming bug.
    """

    async def boom(*args: Any, **kwargs: Any) -> int:
        # E.g. a real codepath defect that raises RuntimeError —
        # operators want this to surface, not be silently swallowed.
        raise RuntimeError("dispatch logic invariant violated")

    fake_redis.publish = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="invariant violated"):
        reanalyze_finding.run(  # type: ignore[attr-defined]
            "f-runtime",
            "Kisqali",
            triggered_by="sentinel:staleness",
        )
