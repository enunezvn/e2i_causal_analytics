"""Unit tests for InsightSignalBus (subsystem 5)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from src.memory.coordination.signals import InsightSignalBus


class FakeStreamsRedis:
    """Minimal fake for the subset of Redis Streams the bus uses."""

    def __init__(self) -> None:
        # stream_key -> [(entry_id, fields)]
        self.streams: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
        # (stream_key, group) -> last_delivered_idx
        self.groups: Dict[Tuple[str, str], int] = {}
        self._next_id = 0

    async def xadd(
        self,
        stream_key: str,
        fields: Dict[str, str],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        self._next_id += 1
        entry_id = f"{self._next_id}-0"
        self.streams.setdefault(stream_key, []).append((entry_id, fields))
        if maxlen and len(self.streams[stream_key]) > maxlen:
            self.streams[stream_key] = self.streams[stream_key][-maxlen:]
        return entry_id

    async def xgroup_create(
        self, stream_key: str, group: str, id: str = "$", mkstream: bool = False
    ) -> bool:
        if mkstream:
            self.streams.setdefault(stream_key, [])
        if (stream_key, group) in self.groups:
            # Mirror redis "BUSYGROUP Consumer Group name already exists".
            raise RuntimeError(f"BUSYGROUP {group} on {stream_key}")
        # "$" means start at the tail; we model with current length.
        self.groups[(stream_key, group)] = len(self.streams.get(stream_key, []))
        return True

    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: Dict[str, str],
        count: int = 10,
        block: int = 0,
    ) -> List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]:
        result = []
        for stream_key in streams:
            entries = self.streams.get(stream_key, [])
            start = self.groups.get((stream_key, groupname), 0)
            new = entries[start : start + count]
            if new:
                self.groups[(stream_key, groupname)] = start + len(new)
                result.append((stream_key, new))
        return result

    async def xack(self, stream_key: str, group: str, entry_id: str) -> int:
        # ack just signals success in our fake; offsets advance on xreadgroup.
        return 1


@pytest.fixture
def fake_redis() -> FakeStreamsRedis:
    return FakeStreamsRedis()


@pytest.fixture(autouse=True)
def patch_redis(fake_redis):
    with patch("src.memory.coordination.signals.get_redis_client", return_value=fake_redis):
        yield


@pytest.mark.asyncio
async def test_publish_writes_brand_scoped_stream(fake_redis: FakeStreamsRedis):
    bus = InsightSignalBus()
    entry_id = await bus.publish(
        topic="cohort:rebuilt", brand="Kisqali", payload={"cohort_id": "C42"}
    )
    assert entry_id  # non-empty
    assert "insights:cohort:rebuilt:Kisqali" in fake_redis.streams
    # Fabhalta stream is untouched -- brand scoping at the key level.
    assert "insights:cohort:rebuilt:Fabhalta" not in fake_redis.streams


@pytest.mark.asyncio
async def test_publish_rejects_empty_brand():
    bus = InsightSignalBus()
    with pytest.raises(ValueError):
        await bus.publish(topic="cohort:rebuilt", brand="", payload={})


@pytest.mark.asyncio
async def test_consume_returns_payload(fake_redis: FakeStreamsRedis):
    bus = InsightSignalBus()
    await bus.publish(topic="x", brand="Kisqali", payload={"a": 1})
    await bus.ensure_group(topic="x", brand="Kisqali", group="grp")
    # ensure_group put offset at tail; we need to publish AFTER group creation
    # for the message to be visible to the new group.
    await bus.publish(topic="x", brand="Kisqali", payload={"a": 2})

    msgs = await bus.consume(
        topic="x", brand="Kisqali", group="grp", consumer="c1", block_ms=0, count=10
    )
    assert len(msgs) == 1
    assert msgs[0]["brand"] == "Kisqali"
    assert msgs[0]["payload"] == {"a": 2}


@pytest.mark.asyncio
async def test_ensure_group_is_idempotent(fake_redis: FakeStreamsRedis):
    bus = InsightSignalBus()
    await bus.ensure_group(topic="x", brand="Kisqali", group="grp")
    # Calling again must not raise even though the fake reports BUSYGROUP.
    await bus.ensure_group(topic="x", brand="Kisqali", group="grp")


@pytest.mark.asyncio
async def test_brand_isolation_separates_consumer_streams(fake_redis: FakeStreamsRedis):
    bus = InsightSignalBus()
    await bus.ensure_group(topic="t", brand="Kisqali", group="g")
    await bus.ensure_group(topic="t", brand="Fabhalta", group="g")
    await bus.publish(topic="t", brand="Kisqali", payload={"b": "K"})

    k = await bus.consume(topic="t", brand="Kisqali", group="g", consumer="c", block_ms=0)
    f = await bus.consume(topic="t", brand="Fabhalta", group="g", consumer="c", block_ms=0)

    assert len(k) == 1 and k[0]["brand"] == "Kisqali"
    assert f == []  # cross-brand subscriber sees nothing


@pytest.mark.asyncio
async def test_iter_messages_rejects_nonpositive_block_ms(fake_redis: FakeStreamsRedis):
    """L3 (#694): iter_messages' unbounded loop must reject block_ms<=0 to avoid
    a busy-spin. A one-shot consume(block_ms=0) is still allowed (see above)."""
    bus = InsightSignalBus()
    agen = bus.iter_messages(topic="t", brand="Kisqali", group="g", consumer="c", block_ms=0)
    with pytest.raises(ValueError, match="block_ms must be > 0"):
        await agen.__anext__()
    # A single non-blocking consume is unaffected by the iter_messages guard.
    msgs = await bus.consume(topic="t", brand="Kisqali", group="g", consumer="c", block_ms=0)
    assert msgs == []
