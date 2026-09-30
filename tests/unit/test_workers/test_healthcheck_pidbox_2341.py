"""Worker healthchecks must not leak Celery reply mailboxes into Redis (#2341).

The defect
----------
On 2026-09-30 prod Redis db1 held 18,454 ``<oid>.reply.celery.pidbox`` lists
with no TTL. Every leaked key was measured with the same shape: one ``{"ok":
"pong"}`` reply, from 1,589 distinct worker container ids, and 18,433 of them
with no binding in ``_kombu.binding.reply.celery.pidbox``. The currently running
containers added 4 of these keys in about 50 minutes.

That is the ``celery inspect ping`` container healthcheck. Each run is a new
process, so it gets a new mailbox. It collects until 1 s passes with no reply and
then deletes the mailbox and its binding. On the Redis transport a worker's reply
is ``SMEMBERS`` (route it) followed by ``LPUSH`` (deliver it). A worker stalled
between the two, for example under swap, pushes into a mailbox the client
already deleted, which recreates the list with no binding, no reader and no TTL.

Reproduced against a throwaway Redis, with a worker whose ``_put`` sleeps 2 s:
the bare broadcast leaked 5 keys in 5 runs and ``inspect ping -d`` leaked 5 in 5.
The destination + ``limit=1`` probe with a 5 s budget leaked 0 in 5.
``control_queue_expires`` cannot fix it because kombu 5.6.1's Redis
``_new_queue`` ignores ``expires``.

Guards
------
* the probe asks ONE node, stops at its first reply, and only that node's pong
  counts, which also fixes "exits 0 if ANY worker answers";
* every worker tier's compose healthcheck uses the probe for its own node, with
  a Docker timeout that leaves room for the import plus the probe's budget.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.workers import healthcheck

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILES = (
    REPO_ROOT / "docker" / "docker-compose.yml",
    REPO_ROOT / "docker" / "docker-compose.secure.yml",
)
WORKER_TIERS = ("worker_light", "worker_medium", "worker_heavy", "worker_forecast")

# Measured 2026-09-30 in prod: an `inspect ping` healthcheck, which imports the same
# app the probe imports, took 3.6-3.9 s end to end. Leave margin on top of that.
IMPORT_BUDGET_SECONDS = 8.0


class _Control:
    def __init__(self, replies: Any = None, exc: Exception | None = None) -> None:
        self.replies = replies
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def ping(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.replies


class _App:
    def __init__(self, control: _Control) -> None:
        self.control = control


NODE = "worker_light@abc123"


def test_probe_addresses_one_node_and_stops_at_its_first_reply() -> None:
    control = _Control([{NODE: {"ok": "pong"}}])
    assert healthcheck.node_answers(_App(control), NODE, timeout=5.0) is True
    assert control.calls == [{"destination": [NODE], "timeout": 5.0, "limit": 1}]


def test_a_pong_from_another_node_does_not_make_this_one_healthy() -> None:
    control = _Control([{"worker_medium@zzz": {"ok": "pong"}}])
    assert healthcheck.node_answers(_App(control), NODE) is False


@pytest.mark.parametrize("replies", [None, [], [{NODE: {}}], [{NODE: {"error": "x"}}]])
def test_no_pong_is_unhealthy(replies: Any) -> None:
    assert healthcheck.node_answers(_App(_Control(replies)), NODE) is False


def test_main_exit_codes() -> None:
    assert healthcheck.main([NODE], app=_App(_Control([{NODE: {"ok": "pong"}}]))) == 0
    assert healthcheck.main([NODE], app=_App(_Control([]))) == 1
    broken = _Control(exc=ConnectionError("broker down"))
    assert healthcheck.main([NODE, "--timeout", "2"], app=_App(broken)) == 1
    assert broken.calls[0]["timeout"] == 2.0


def _seconds(value: object) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(value).strip())
    assert m, f"unparseable duration {value!r}"
    return float(m.group(1))


def _worker_services(path: Path) -> dict[str, dict[str, Any]]:
    # The secure stack spells its services worker-light etc.; the node prefix is
    # still worker_light (its --hostname), so key by the underscored form.
    services = yaml.safe_load(path.read_text())["services"]
    by_tier = {name.replace("-", "_"): svc for name, svc in services.items()}
    return {name: svc for name, svc in by_tier.items() if name in WORKER_TIERS}


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: p.name)
def test_every_worker_healthcheck_is_the_one_node_probe(path: Path) -> None:
    workers = _worker_services(path)
    assert {"worker_light", "worker_medium", "worker_heavy"} <= set(workers), path
    for name, svc in workers.items():
        test = svc["healthcheck"]["test"]
        assert test[0] == "CMD-SHELL", f"{name}: $HOSTNAME must be shell-expanded"
        command = " ".join(str(x) for x in test[1:])
        assert "inspect ping" not in command, f"{name}: bare/unscoped pidbox ping leaks (#2341)"
        assert f"python -m src.workers.healthcheck {name}@$$HOSTNAME" in command, name
        # The probe must name the node the worker registers as.
        assert f"--hostname={name}@%h" in " ".join(str(svc["command"]).split()), name


@pytest.mark.parametrize("path", COMPOSE_FILES, ids=lambda p: p.name)
def test_docker_timeout_leaves_room_for_import_plus_probe_budget(path: Path) -> None:
    """A probe killed mid-wait leaves its mailbox and its binding behind."""
    floor = IMPORT_BUDGET_SECONDS + healthcheck.DEFAULT_TIMEOUT_SECONDS
    for name, svc in _worker_services(path).items():
        timeout = _seconds(svc["healthcheck"]["timeout"])
        assert timeout >= floor, f"{name}: healthcheck.timeout {timeout}s < {floor}s"
