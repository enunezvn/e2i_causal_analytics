"""#1999: docker must give the api container gunicorn's own graceful window.

The deploy recreates the api with ``docker compose up -d --force-recreate`` and
no ``-t`` (.github/workflows/deploy.yml), so the old container gets its
``stop_grace_period`` (docker's default: 10 s) before SIGKILL. gunicorn runs
with ``--graceful-timeout 30``: on SIGTERM its workers finish in-flight requests
(uvicorn waits for every request task, and a shielded expert-review build with
it) and the master SIGKILLs any leftovers at 30 s. With the 10 s default docker
killed the whole container first, so a build in flight for more than ~10 s was
lost on every deploy (live ``docker inspect e2i_api``: StopTimeout=<nil>).

Measured on this box (docker 29.1.3, compose v5.0.1), scratch project with a
15 s SIGTERM trap: ``stop_grace_period: 25s`` let the old container exit 0 after
15 s on ``up -d --force-recreate``; without it the container died 137 at 10 s.
Measured gunicorn stop bound: with ``--graceful-timeout 3`` and two 60 s requests
in flight, SIGTERM -> master exit took 3.13 s (graceful timeout + 0.13 s). So
the period must exceed the graceful timeout plus that exit overhead; the pinned
headroom below is several times the measured overhead.

Both compose files that run this gunicorn command are checked: the deploy's base
file and the hardened ``docker-compose.secure.yml`` (kept in sync by hand,
docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE_FILES = {
    "base": _REPO_ROOT / "docker" / "docker-compose.yml",
    "secure": _REPO_ROOT / "docker" / "docker-compose.secure.yml",
}
# Beyond gunicorn's graceful timeout: master exit measured at +0.13 s.
_MIN_HEADROOM_SECONDS = 3


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates compose's local ``!override`` / ``!reset`` tags."""


def _passthrough(loader: Any, tag_suffix: str, node: Any) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _passthrough)


def _api(path: Path) -> dict[str, Any]:
    doc = yaml.load(path.read_text(), Loader=_ComposeLoader) or {}
    return doc["services"]["api"]


def _seconds(duration: str) -> float:
    """Compose duration (``35s``, ``1m30s``, ``1m``) in seconds."""
    parts = re.fullmatch(r"(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", str(duration).strip())
    assert parts and any(parts.groups()), f"unparseable compose duration: {duration!r}"
    minutes, seconds = parts.groups()
    return int(minutes or 0) * 60 + float(seconds or 0)


def _graceful_timeout(command: Any) -> float:
    text = command if isinstance(command, str) else " ".join(command)
    match = re.search(r"--graceful-timeout[ =](\d+)", text)
    assert match, "the api gunicorn command must set --graceful-timeout explicitly"
    return float(match.group(1))


@pytest.mark.parametrize("label", sorted(_COMPOSE_FILES))
def test_api_stop_grace_period_covers_gunicorn_graceful_timeout(label: str) -> None:
    api = _api(_COMPOSE_FILES[label])
    assert "stop_grace_period" in api, (
        f"{label}: api has no stop_grace_period, so docker SIGKILLs it at 10 s, before "
        "gunicorn's graceful timeout (#1999)"
    )
    graceful = _graceful_timeout(api["command"])
    grace = _seconds(api["stop_grace_period"])
    assert grace >= graceful + _MIN_HEADROOM_SECONDS, (
        f"{label}: stop_grace_period {grace:g}s must cover --graceful-timeout {graceful:g}s "
        f"plus {_MIN_HEADROOM_SECONDS}s headroom"
    )


def test_duration_parser_reads_compose_forms() -> None:
    assert _seconds("35s") == 35
    assert _seconds("1m30s") == 90
    assert _seconds("1m") == 60
