"""Process-local DataFrame handle registry (issue #1734).

Large data frames must NEVER ride LangGraph state. On the chat path the
heterogeneous_optimizer graph runs NESTED under the streamed chatbot graph
(checkpoint_ns ``tools:...|dispatch:...``): langchain's ``astream_events``
emits ``on_chain_start``/``on_chain_end`` for every node with the node's full
input/output state in ``data``, and the AG-UI/CoAgent bridge forwards those to
the browser. A patient-level cohort frame in state therefore re-streams to the
chat client once per node event — measured on the 2026-08-19 post1730 eval,
turn 4.4: one 377.6 MB SSE chat turn (112 RAW events totaling 294.6 MB plus
23 STATE_SNAPSHOT events totaling 82.9 MB; ~11.6 MB of serialized
``tier0_data`` in each event). It also violates the aggregates-only frontend
contract (``patient_journey_id`` rows reaching the browser) and makes state
unserializable for any checkpointer (ormsgpack cannot encode DataFrames — the
#1351 failure class).

Instead: the caller STASHES the frame here and puts only the returned string
handle into graph state; nodes RESOLVE the handle; the caller RELEASES it when
the run completes (``try/finally`` or the ``stashed_frame`` context manager).
The registry is process-local (a module-level dict) — valid because LangGraph
executes graph nodes in the same process as the caller.

Note (verified against the installed langgraph, 2026-08-19): removing the
frame key from the state schema alone is NOT enough — the top-level
``on_chain_start`` event carries the caller's RAW ``ainvoke`` input dict
*before* schema filtering, so every graph invoker must pass the handle, never
the frame.
"""

import logging
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

logger = logging.getLogger(__name__)

_REF_PREFIX = "frame-ref:"
_frames: Dict[str, Any] = {}
_lock = threading.Lock()

# Soft ceiling: a leak-detection aid, not an eviction policy. Evicting would
# yank a frame out from under a concurrent run — worse than the memory it
# saves. Callers release in try/finally; this warning fires when one doesn't.
_LIVE_WARN_THRESHOLD = 16


def stash_frame(frame: Any, *, label: str = "") -> str:
    """Store ``frame`` and return an opaque string handle to put in graph state."""
    ref = f"{_REF_PREFIX}{label + ':' if label else ''}{uuid.uuid4()}"
    with _lock:
        _frames[ref] = frame
        live = len(_frames)
    if live > _LIVE_WARN_THRESHOLD:
        logger.warning(
            "frame_registry holds %d live frames — a caller is likely missing "
            "its release_frame() (refs must be released in try/finally around "
            "the graph run).",
            live,
        )
    return ref


def resolve_frame(ref: Optional[str]) -> Optional[Any]:
    """Return the stashed frame for ``ref``, or None (unknown/released/None ref)."""
    if not ref:
        return None
    with _lock:
        return _frames.get(ref)


def release_frame(ref: Optional[str]) -> None:
    """Drop a stashed frame. Idempotent; releasing an unknown/None ref is a no-op."""
    if not ref:
        return
    with _lock:
        _frames.pop(ref, None)


@contextmanager
def stashed_frame(frame: Any, *, label: str = "") -> Iterator[Optional[str]]:
    """Stash ``frame`` for the duration of a ``with`` block (None-safe).

    Yields the handle (or None when ``frame`` is None) and always releases on
    exit, so a raising graph run cannot leak the frame.
    """
    if frame is None:
        yield None
        return
    ref = stash_frame(frame, label=label)
    try:
        yield ref
    finally:
        release_frame(ref)


def resolve_state_frame(
    state: Mapping[str, Any],
    *,
    ref_key: str = "tier0_frame_ref",
    legacy_key: str = "tier0_data",
) -> Optional[Any]:
    """Resolve the tier0 passthrough frame for a graph node.

    Priority: the ``ref_key`` handle (the only channel that exists in compiled
    graph state), then a ``legacy_key`` frame sitting directly in the mapping.
    The legacy read serves DIRECT node invocations only (unit tests / library
    callers handing ``node.execute`` a plain dict): the state schema no longer
    declares that key, so LangGraph drops it at every compiled-graph boundary
    (verified against the installed langgraph) and an in-dict frame can never
    reach the streamed/checkpointed path.
    """
    frame = resolve_frame(state.get(ref_key))
    if frame is not None:
        return frame
    return state.get(legacy_key)


def live_frame_count() -> int:
    """Number of currently stashed frames (leak diagnostics / tests)."""
    with _lock:
        return len(_frames)


def _clear_all_for_tests() -> None:
    """Test hygiene only — drop every stashed frame."""
    with _lock:
        _frames.clear()
