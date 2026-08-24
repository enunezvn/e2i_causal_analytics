"""supabase-meta must run under an init shim, and the override must not drift (#1801).

All 40 zombies on the prod box are children of ``supabase-meta``'s PID 1 -- a
containerized ``node dist/server/server.js`` running with ``HostConfig.Init`` unset.
PID 1 in a namespace ignores signals it has no handler for and this one installs no
``SIGCHLD`` handler, so its exited children accumulate forever. A host-side reap
cannot fix that (see #1803); the fix belongs where the container is defined.

The second test exists because the tracked override and the deployed copy were found
to have already drifted, and **nothing syncs them**: the only writer of
``/opt/supabase/docker/`` is ``scripts/supabase/setup_self_hosted.sh``, a one-time
provisioner. An override edited in git but never copied to the box is inert -- the
same shape as the FalkorDB persistence no-op, where a compose fix for an untouched
service did nothing until the container was force-recreated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACKED_OVERRIDE = REPO_ROOT / "docker" / "supabase" / "docker-compose.override.yml"
DEPLOYED_OVERRIDE = Path("/opt/supabase/docker/docker-compose.override.yml")


def _services(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text())
    return (doc or {}).get("services") or {}


def test_the_tracked_override_defines_meta() -> None:
    """Positive control: if `meta` ever disappears, the init assertion below would
    pass vacuously over a missing key."""
    assert "meta" in _services(TRACKED_OVERRIDE), (
        "the tracked supabase override no longer defines a `meta` service"
    )


def test_supabase_meta_runs_under_an_init_shim() -> None:
    """`init: true` gives the container tini as PID 1, which reaps.

    Red before the fix: the meta service had networks and resource limits but no
    init, so its node server was PID 1 with no reaper (#1801).
    """
    meta = _services(TRACKED_OVERRIDE)["meta"]
    assert meta.get("init") is True, (
        "supabase-meta must set `init: true` -- without an init shim its node server "
        "is PID 1 with no SIGCHLD handler, and its exited children become permanent "
        "zombies that nothing on the host can reap (#1801). Got init="
        f"{meta.get('init')!r}"
    )


@pytest.mark.skipif(
    not DEPLOYED_OVERRIDE.exists(),
    reason="no deployed supabase override here (CI, or a box without the stack)",
)
def test_the_deployed_override_has_not_drifted_from_the_tracked_one() -> None:
    """Nothing syncs docker/supabase/ to /opt/supabase/docker/.

    Comments are ignored: only the parsed service definitions matter, because a
    comment-only difference cannot change behaviour and would make this guard
    noisy enough to be disabled.
    """
    tracked = _services(TRACKED_OVERRIDE)
    deployed = _services(DEPLOYED_OVERRIDE)

    differing = sorted(
        name for name in set(tracked) | set(deployed) if tracked.get(name) != deployed.get(name)
    )
    assert not differing, (
        "the deployed supabase override has drifted from the tracked one for: "
        f"{differing}. Nothing copies docker/supabase/docker-compose.override.yml to "
        "/opt/supabase/docker/, so an edit in git is INERT until someone copies it and "
        "recreates the affected container. Sync it, then `docker compose up -d "
        "--force-recreate <service>` for anything whose definition changed."
    )
