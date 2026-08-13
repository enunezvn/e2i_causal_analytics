"""llm_usage_recorder: never blocks, never raises; bounded queue drops on
overflow; batch insert failures log-and-drop (fail-open)."""

import queue
from types import SimpleNamespace

import src.services.llm_usage_recorder as recorder
from src.services.llm_usage_recorder import LLMUsageEvent


def _event(**over):
    base = {"provider": "openai", "model": "gpt-4o", "input_tokens": 10, "output_tokens": 5}
    base.update(over)
    return LLMUsageEvent(**base)


def test_to_row_shape():
    row = _event(surface="chat", user_id="u1", session_id="u1~s", request_id="r1").to_row()
    assert row == {
        "provider": "openai",
        "model": "gpt-4o",
        "input_tokens": 10,
        "output_tokens": 5,
        "surface": "chat",
        "component": None,
        "user_id": "u1",
        "session_id": "u1~s",
        "request_id": "r1",
    }


def test_enqueue_drops_when_full(monkeypatch):
    monkeypatch.setattr(recorder, "_ensure_flusher", lambda: None)
    monkeypatch.setattr(recorder, "_queue", queue.Queue(maxsize=2))
    assert recorder.enqueue(_event()) is True
    assert recorder.enqueue(_event()) is True
    assert recorder.enqueue(_event()) is False  # dropped, no exception


def _reset_drop_state(monkeypatch, now=1000.0):
    """Isolate the module-level rate-limit state and freeze the clock."""
    clock = {"now": now}
    monkeypatch.setattr(recorder, "_dropped_since_warn", 0)
    monkeypatch.setattr(recorder, "_last_warn_at", None)
    monkeypatch.setattr(recorder, "_monotonic", lambda: clock["now"])
    return clock


def test_queue_full_warning_is_rate_limited(monkeypatch, caplog):
    monkeypatch.setattr(recorder, "_ensure_flusher", lambda: None)
    monkeypatch.setattr(recorder, "_queue", queue.Queue(maxsize=1))
    _reset_drop_state(monkeypatch)
    assert recorder.enqueue(_event()) is True

    with caplog.at_level("WARNING", logger="src.services.llm_usage_recorder"):
        for _ in range(5):
            assert recorder.enqueue(_event()) is False  # still drops, still False
    warnings = [r for r in caplog.records if "queue full" in r.getMessage()]
    assert len(warnings) == 1  # first drop warns; the next 4 are suppressed


def test_queue_full_warning_resumes_with_cumulative_count(monkeypatch, caplog):
    monkeypatch.setattr(recorder, "_ensure_flusher", lambda: None)
    monkeypatch.setattr(recorder, "_queue", queue.Queue(maxsize=1))
    clock = _reset_drop_state(monkeypatch)
    recorder.enqueue(_event())

    with caplog.at_level("WARNING", logger="src.services.llm_usage_recorder"):
        for _ in range(3):
            recorder.enqueue(_event())  # 1 warned ("1 event"), 2 suppressed
        clock["now"] += recorder._WARN_EVERY_SECONDS + 1
        recorder.enqueue(_event())  # warns again with the 2 suppressed + this one

    warnings = [r.getMessage() for r in caplog.records if "queue full" in r.getMessage()]
    assert len(warnings) == 2
    assert "dropped 1 " in warnings[0]
    assert "dropped 3 " in warnings[1]


def test_insert_batch_success():
    inserted = []

    class _Client:
        def table(self, name):
            assert name == "llm_usage_events"
            return SimpleNamespace(
                insert=lambda rows: SimpleNamespace(
                    execute=lambda: inserted.append(rows) or SimpleNamespace(data=rows)
                )
            )

    assert recorder._insert_batch([_event(), _event()], _Client()) is True
    assert len(inserted[0]) == 2


def test_insert_batch_failure_is_swallowed():
    class _Boom:
        def table(self, name):
            raise RuntimeError("db down")

    assert recorder._insert_batch([_event()], _Boom()) is False  # no raise


def test_drain_batch_respects_max(monkeypatch):
    q = queue.Queue()
    for _ in range(60):
        q.put(_event())
    monkeypatch.setattr(recorder, "_queue", q)
    batch = recorder._drain_batch()
    assert len(batch) == recorder._BATCH_MAX


# =============================================================================
# #1560: the flusher guard must be pid-keyed (fork-safe under gunicorn preload)
# =============================================================================


class _FakeThread:
    """Records construction/start without spawning a real thread."""

    instances: list["_FakeThread"] = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.started = False
        _FakeThread.instances.append(self)

    def start(self):
        self.started = True


def test_flusher_guard_is_pid_keyed_not_boolean():
    """A boolean guard forked True from a preloaded master would silently
    suppress the child's flusher (dead inherited thread, events never flush)."""
    assert not hasattr(recorder, "_flusher_started"), (
        "boolean _flusher_started is fork-unsafe under gunicorn --preload; "
        "use the pid-keyed _flusher_pid guard"
    )
    assert hasattr(recorder, "_flusher_pid")


def test_ensure_flusher_restarts_in_forked_child(monkeypatch):
    import os

    _FakeThread.instances = []
    monkeypatch.setattr(recorder.threading, "Thread", _FakeThread)
    # Simulate state inherited from a master process with a different pid.
    monkeypatch.setattr(recorder, "_flusher_pid", os.getpid() + 1)

    recorder._ensure_flusher()

    assert len(_FakeThread.instances) == 1, "forked child must start its OWN flusher"
    assert _FakeThread.instances[0].started is True
    assert recorder._flusher_pid == os.getpid()


def test_ensure_flusher_is_idempotent_within_a_process(monkeypatch):
    import os

    _FakeThread.instances = []
    monkeypatch.setattr(recorder.threading, "Thread", _FakeThread)
    monkeypatch.setattr(recorder, "_flusher_pid", None)

    recorder._ensure_flusher()
    recorder._ensure_flusher()

    assert len(_FakeThread.instances) == 1, "same-pid re-entry must not spawn again"
    assert recorder._flusher_pid == os.getpid()
