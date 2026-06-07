"""
InsightSignalBus — Redis Streams fan-out for inter-agent events.

Use cases:
- cohort_constructor publishes ``insights:cohort:rebuilt`` after a rebuild;
  causal_impact subscribes and re-reads the cohort.
- causal_impact publishes ``insights:causal_path:discovered`` when a new
  path lands; gap_analyzer + heterogeneous_optimizer subscribe.

Streams (not pub/sub): consumers have offsets, so a late subscriber can
replay events. Consumer groups give at-least-once delivery without
sharing offsets between agents.

Streams are brand-namespaced for tenancy: ``insights:{topic}:{brand}``.
Subscribers that want all brands can use ``insights:{topic}:all`` as a
side channel, but cross-brand signals must be authored explicitly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from src.memory.services.factories import get_redis_client

logger = logging.getLogger(__name__)

# Stream entries are capped to keep memory bounded; older entries fall off.
DEFAULT_STREAM_MAXLEN = 10_000


@dataclass
class SignalStream:
    """Describes a stream the bus knows about."""

    topic: str  # e.g. "cohort:rebuilt", "causal_path:discovered"
    brand: str  # 'Kisqali', 'Fabhalta', 'Remibrutinib', or 'all'

    @property
    def key(self) -> str:
        return f"insights:{self.topic}:{self.brand}"


class InsightSignalBus:
    """
    Publish/subscribe to brand-scoped agent signals.

    Publishing is fire-and-forget. Consumption uses Redis Streams consumer
    groups so each agent (group) sees each message once, but the same
    message reaches multiple distinct groups (orchestrator, drift_monitor,
    etc.).
    """

    def __init__(self, stream_maxlen: int = DEFAULT_STREAM_MAXLEN):
        self._maxlen = stream_maxlen

    # ------------------------------------------------------------------ publish

    async def publish(
        self,
        topic: str,
        brand: str,
        payload: Dict[str, Any],
    ) -> str:
        """
        Append a message to ``insights:{topic}:{brand}``.

        Returns the stream entry ID (Redis-assigned).
        """
        if not brand:
            raise ValueError("brand is required (use 'all' for explicit cross-brand)")
        stream = SignalStream(topic=topic, brand=brand)
        redis = get_redis_client()
        # XADD with MAXLEN ~ approximate trim — cheap and bounds the stream.
        entry_id = await redis.xadd(
            stream.key,
            {"payload": json.dumps(payload), "brand": brand, "topic": topic},
            maxlen=self._maxlen,
            approximate=True,
        )
        logger.debug(f"published {stream.key} entry={entry_id} brand={brand}")
        return str(entry_id)

    # --------------------------------------------------------------- subscribe

    async def ensure_group(
        self,
        topic: str,
        brand: str,
        group: str,
    ) -> None:
        """
        Idempotently create a consumer group on a stream.

        XGROUP CREATE fails with BUSYGROUP if the group exists — caught
        and treated as success. MKSTREAM creates the stream if missing.
        """
        stream = SignalStream(topic=topic, brand=brand)
        redis = get_redis_client()
        try:
            await redis.xgroup_create(stream.key, group, id="$", mkstream=True)
            logger.info(f"created consumer group {group} on {stream.key}")
        except Exception as exc:
            # redis.exceptions.ResponseError on BUSYGROUP — group already exists.
            if "BUSYGROUP" in str(exc):
                return
            raise

    async def consume(
        self,
        topic: str,
        brand: str,
        group: str,
        consumer: str,
        block_ms: int = 5_000,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Read up to ``count`` messages for one consumer, blocking up to
        ``block_ms`` ms if none are available.

        Returns a list of {entry_id, brand, topic, payload}.
        Caller must call ``ack()`` for each consumed entry.

        A single ``consume`` call MAY use ``block_ms=0`` (a one-shot
        non-blocking read); the busy-spin guard lives in ``iter_messages``,
        whose unbounded loop is the path that would actually spin.
        """
        stream = SignalStream(topic=topic, brand=brand)
        redis = get_redis_client()
        # ">" = only new (undelivered) messages for this group/consumer.
        result = await redis.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream.key: ">"},
            count=count,
            block=block_ms,
        )
        if not result:
            return []
        messages = []
        for _stream_key, entries in result:
            for entry_id, fields in entries:
                try:
                    payload = json.loads(fields.get("payload", "{}"))
                except (TypeError, json.JSONDecodeError):
                    payload = {}
                messages.append(
                    {
                        "entry_id": entry_id,
                        "brand": fields.get("brand", brand),
                        "topic": fields.get("topic", topic),
                        "payload": payload,
                        "_stream_key": stream.key,
                    }
                )
        return messages

    async def ack(self, message: Dict[str, Any], group: str) -> int:
        """Acknowledge a consumed message so it isn't redelivered."""
        redis = get_redis_client()
        result = await redis.xack(message["_stream_key"], group, message["entry_id"])
        return int(result)

    # ---------------------------------------------------------------- iterate

    async def iter_messages(
        self,
        topic: str,
        brand: str,
        group: str,
        consumer: str,
        block_ms: int = 5_000,
        count: int = 10,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Async iterator over messages — convenient for background loops.

        Caller is responsible for ack()'ing each message after processing.
        """
        # L3 (#694): the unbounded loop below busy-spins if block_ms<=0 (a
        # non-blocking xreadgroup returns immediately on an empty stream).
        # Require a positive block timeout so each empty poll actually blocks.
        if block_ms <= 0:
            raise ValueError(f"block_ms must be > 0 for iter_messages, got {block_ms}")
        await self.ensure_group(topic, brand, group)
        while True:
            batch = await self.consume(topic, brand, group, consumer, block_ms, count)
            for msg in batch:
                yield msg


_bus_singleton: Optional[InsightSignalBus] = None


def get_insight_signal_bus() -> InsightSignalBus:
    """Process-wide singleton for the bus."""
    global _bus_singleton
    if _bus_singleton is None:
        _bus_singleton = InsightSignalBus()
    return _bus_singleton
