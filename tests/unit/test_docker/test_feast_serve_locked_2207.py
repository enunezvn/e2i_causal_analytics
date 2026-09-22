"""#2207 follow-up (codex r1 HIGH-3): the feast sidecar's materialize endpoints take the
shared registry lock.

Feast's file registry is written in place and every materialize call writes it; the
materializer loop and the serve container's ``apply`` already serialize on
``/feast/data/.registry.lock`` (#556 H2). Once the worker's beats materialize over HTTP
the feature server became a second, unlocked writer. ``docker/feast/serve_locked.py``
wraps the two store methods the handlers call with an exclusive flock on the SAME file.

The image cannot be built on this box (docker build is denied here); these tests pin
the wrapper's semantics with a fake store and lock the entrypoint / Dockerfile wiring.
The live proof is the sidecar log line after the deploy's image rebuild.
"""

from __future__ import annotations

import importlib.util
import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE = REPO / "docker" / "feast" / "serve_locked.py"
ENTRYPOINT = REPO / "docker" / "feast" / "entrypoint.sh"
MATERIALIZER = REPO / "docker" / "feast" / "materializer-entrypoint.sh"
DOCKERFILE = REPO / "docker" / "Dockerfile.feast"


def _load():
    spec = importlib.util.spec_from_file_location("serve_locked", MODULE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
def test_locked_serializes_concurrent_callers(tmp_path):
    mod = _load()
    lock = str(tmp_path / ".registry.lock")
    active = {"n": 0, "max": 0}
    guard = threading.Lock()

    def work(_view):
        with guard:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.05)
        with guard:
            active["n"] -= 1
        return "ok"

    wrapped = mod.locked(work, lock)
    threads = [threading.Thread(target=wrapped, args=(f"v{i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["max"] == 1, "two materialize calls overlapped under the lock"
    assert Path(lock).exists()


@pytest.mark.unit
def test_lock_store_materialization_wraps_both_methods_and_forwards_arguments(tmp_path):
    mod = _load()
    lock = str(tmp_path / ".registry.lock")
    store = MagicMock()
    store.materialize.return_value = "full"
    store.materialize_incremental.return_value = "inc"
    original_full, original_inc = store.materialize, store.materialize_incremental

    mod.lock_store_materialization(store, lock)

    assert store.materialize("s", "e", ["v"]) == "full"
    assert store.materialize_incremental("e", None) == "inc"
    original_full.assert_called_once_with("s", "e", ["v"])
    original_inc.assert_called_once_with("e", None)
    assert store.materialize is not original_full  # the handlers look the attribute up at call time


@pytest.mark.unit
def test_lock_is_released_when_the_call_raises(tmp_path):
    mod = _load()
    lock = str(tmp_path / ".registry.lock")

    def boom():
        raise RuntimeError("materialize failed")

    wrapped = mod.locked(boom, lock)
    with pytest.raises(RuntimeError):
        wrapped()
    # a second call must not block forever: the lock was released in `finally`
    done = threading.Event()

    def second():
        try:
            wrapped()
        except RuntimeError:
            done.set()

    t = threading.Thread(target=second)
    t.start()
    t.join(timeout=2)
    assert done.is_set()


@pytest.mark.unit
def test_lock_path_matches_the_materializer_loop_and_the_apply_lock():
    mod = _load()
    assert mod.LOCK_PATH == "/feast/data/.registry.lock"
    assert "LOCK=/feast/data/.registry.lock" in MATERIALIZER.read_text()
    assert "flock /feast/data/.registry.lock feast --chdir /feast apply" in ENTRYPOINT.read_text()


@pytest.mark.unit
def test_entrypoint_execs_the_locked_server_and_never_falls_back_to_an_unlocked_one():
    """codex r2 HIGH-5: a fallback to plain `feast serve` would bring an UNLOCKED writer
    up; the container must fail instead (the deploy rolls the feast recreate back)."""
    text = ENTRYPOINT.read_text()
    assert re.search(r"^exec python3 /serve_locked\.py\s*$", text, re.M)
    assert "feast --chdir /feast serve" not in text.replace(
        "flock /feast/data/.registry.lock feast --chdir /feast apply", ""
    )


@pytest.mark.unit
def test_lock_wait_is_bounded(tmp_path):
    mod = _load()
    lock = str(tmp_path / ".registry.lock")
    holder_ready = threading.Event()
    release = threading.Event()

    def hold():
        with open(lock, "a") as f:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            holder_ready.set()
            release.wait(5)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    t = threading.Thread(target=hold)
    t.start()
    holder_ready.wait(2)
    wrapped = mod.locked(lambda: "ran", lock, wait_seconds=0.6)
    started = time.monotonic()
    with pytest.raises(mod.RegistryLockTimeout):
        wrapped()
    assert time.monotonic() - started < 3
    release.set()
    t.join()
    assert wrapped() == "ran"  # once released, the call goes through


@pytest.mark.unit
def test_server_uses_the_cli_defaults():
    src = MODULE.read_text()
    assert "keep_alive_timeout=5" in src and "registry_ttl_sec=5" in src
    assert "no_access_log=False" in src


@pytest.mark.unit
def test_dockerfile_ships_the_wrapper():
    assert "COPY --chmod=755 feast/serve_locked.py /serve_locked.py" in DOCKERFILE.read_text()
