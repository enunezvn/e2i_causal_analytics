"""Fail-open, non-blocking writer for llm_usage_events (spec 2026-07-12).

Capture hooks call enqueue() from the request path; a lazily-started daemon
thread batches rows into Supabase. Failure policy is drop-and-warn at every
stage: a DB outage or full queue loses usage telemetry but can never break an
LLM call or grow memory unboundedly. In-flight events may be lost on process
shutdown — accepted (telemetry, not billing records).
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_QUEUE_MAX = 1000
_BATCH_MAX = 50
_POLL_SECONDS = 2.0


@dataclass
class LLMUsageEvent:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    surface: str = "other"
    component: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "surface": self.surface,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
        }


_queue: "queue.Queue[LLMUsageEvent]" = queue.Queue(maxsize=_QUEUE_MAX)
_flusher_started = False
_flusher_lock = threading.Lock()


def enqueue(event: LLMUsageEvent) -> bool:
    """Never blocks, never raises. False = queue full, event dropped."""
    try:
        _queue.put_nowait(event)
    except queue.Full:
        logger.warning("llm_usage_recorder: queue full, dropping usage event")
        return False
    _ensure_flusher()
    return True


def _ensure_flusher() -> None:
    global _flusher_started
    if _flusher_started:
        return
    with _flusher_lock:
        if _flusher_started:
            return
        thread = threading.Thread(target=_flush_loop, name="llm-usage-flusher", daemon=True)
        thread.start()
        _flusher_started = True


def _drain_batch() -> List[LLMUsageEvent]:
    events: List[LLMUsageEvent] = []
    try:
        events.append(_queue.get(timeout=_POLL_SECONDS))
    except queue.Empty:
        return events
    while len(events) < _BATCH_MAX:
        try:
            events.append(_queue.get_nowait())
        except queue.Empty:
            break
    return events


def _insert_batch(events: List[LLMUsageEvent], client: Any) -> bool:
    """Separated from the loop so unit tests exercise it with a fake client."""
    if not events:
        return True
    try:
        client.table("llm_usage_events").insert([e.to_row() for e in events]).execute()
        return True
    except Exception as e:
        logger.warning(
            "llm_usage_recorder: batch insert failed, dropping %d event(s): %s",
            len(events),
            e,
        )
        return False


def _flush_loop() -> None:
    from src.api.dependencies.supabase_client import get_supabase

    while True:
        events = _drain_batch()
        if not events:
            continue
        try:
            client = get_supabase()
        except Exception as e:
            logger.warning("llm_usage_recorder: no client, dropping %d: %s", len(events), e)
            continue
        if client is None:
            logger.warning("llm_usage_recorder: Supabase unavailable, dropping %d", len(events))
            continue
        _insert_batch(events, client)
