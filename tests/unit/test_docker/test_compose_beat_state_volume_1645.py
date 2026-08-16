"""CI guard: celery beat's PersistentScheduler state must live on a named volume (#1645).

``PersistentScheduler`` keeps ``last_run_at`` for every ``beat_schedule`` entry in
a shelve file. The scheduler service pointed ``--schedule`` at
``/tmp/celerybeat-schedule`` — and ``/tmp`` in that container is a **tmpfs**, so
every deploy destroyed the state and reset ``last_run_at`` to boot time. An
interval schedule only becomes due once *uptime > interval*, so on a box that
deploys several times a day no 24-hour entry could ever fire (measured in the
issue: in a 5h container life only the <=4h entries fired). ``sync_operational_corpus``
was the observable casualty (#1649) — scheduled, implemented, queue consumed, never run.

The compose half of the fix is not unit-testable end to end (it needs a deploy),
so this module pins the declared wiring it depends on, with no Docker daemon
required:

* the ``--schedule`` path lives under a **named volume** mount, not a tmpfs;
* that volume is declared at the top level, so it is a real persistent volume
  rather than an anonymous one;
* the mount directory exists in **both** Dockerfile stages — a named volume whose
  mountpoint is absent from the image is created *root-owned*, and beat runs as
  the non-root ``e2i`` user under a ``read_only`` rootfs, so it could not write it
  (the same trap the ``optimized_modules`` comment records);
* the #528-A property survives: the scheduler still does NOT merge
  ``<<: *common-env``.

The schedule-side half is guarded by
``tests/unit/test_workers/test_beat_daily_wallclock_1645.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# tests/unit/test_docker/<this file>  ->  parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"

SCHEDULER = "scheduler"


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates compose's local ``!override`` / ``!reset`` tags."""


def _passthrough(loader: yaml.Loader, tag_suffix: str, node: yaml.Node):  # noqa: ANN401
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _passthrough)


def _load(path: Path) -> dict:
    with open(path) as fh:
        return yaml.load(fh, Loader=_ComposeLoader) or {}


def _scheduler(compose: dict) -> dict:
    svc = (compose.get("services") or {}).get(SCHEDULER)
    assert svc, "docker-compose.yml has no `scheduler` service"
    return svc


def _command_text(svc: dict) -> str:
    cmd = svc.get("command")
    if isinstance(cmd, list):
        return " ".join(str(part) for part in cmd)
    return str(cmd or "")


def _schedule_path(svc: dict) -> str:
    text = _command_text(svc)
    match = re.search(r"--schedule[=\s]+(\S+)", text)
    assert match, (
        "the scheduler command must pass an explicit --schedule=<path> (#1645). "
        "Without it celery falls back to a CWD-relative 'celerybeat-schedule', and "
        f"/app is read_only. Command was: {text!r}"
    )
    return match.group(1)


def _named_volume_targets(svc: dict) -> dict[str, str]:
    """``{container_path: volume_name}`` for every named-volume mount (not binds)."""
    targets: dict[str, str] = {}
    for vol in svc.get("volumes") or []:
        if isinstance(vol, str):
            parts = vol.split(":")
            if len(parts) < 2:
                continue
            source, target = parts[0], parts[1]
            # A bind mount's source is a path; a named volume's is a bare name.
            if source.startswith((".", "/", "~", "$")):
                continue
            targets[target] = source
        elif isinstance(vol, dict) and vol.get("type") == "volume" and vol.get("source"):
            targets[str(vol["target"])] = str(vol["source"])
    return targets


def _tmpfs_paths(svc: dict) -> list[str]:
    raw = svc.get("tmpfs")
    if raw is None:
        return []
    entries = [raw] if isinstance(raw, str) else list(raw)
    return [entry.partition(":")[0] for entry in entries]


def _is_under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def test_beat_state_file_lives_on_a_named_volume() -> None:
    compose = _load(BASE_COMPOSE)
    svc = _scheduler(compose)
    state_path = _schedule_path(svc)
    mounts = _named_volume_targets(svc)

    holder = next(
        (target for target in mounts if _is_under(state_path, target)),
        None,
    )
    assert holder is not None, (
        f"the beat state file ({state_path}) is not under any named-volume mount on "
        f"the scheduler. Named-volume mounts present: {sorted(mounts)}. Without a "
        "volume the file dies with the container on every deploy, last_run_at resets, "
        "and no >24h interval entry can ever become due (#1645)."
    )

    volume_name = mounts[holder]
    declared = compose.get("volumes") or {}
    assert volume_name in declared, (
        f"the scheduler mounts '{volume_name}' but it is not declared in the top-level "
        "`volumes:` block, so compose would treat it as undeclared/anonymous."
    )


def test_beat_state_file_is_not_on_a_tmpfs() -> None:
    """The original defect verbatim: /tmp in this container is a tmpfs."""
    svc = _scheduler(_load(BASE_COMPOSE))
    state_path = _schedule_path(svc)

    on_tmpfs = [tmpfs for tmpfs in _tmpfs_paths(svc) if _is_under(state_path, tmpfs)]
    assert not on_tmpfs, (
        f"the beat state file ({state_path}) sits on tmpfs {on_tmpfs} — that is the "
        "#1645 defect: the tmpfs is recreated with the container on every deploy, so "
        "last_run_at resets and 24-hour entries never fire."
    )


def test_beat_state_directory_exists_in_both_dockerfile_stages() -> None:
    """A named volume inherits ownership from the image dir; if absent it is root-owned.

    beat runs as the non-root ``e2i`` user (USER e2i, uid 1000) under
    ``read_only: true``, so a root-owned mountpoint means it cannot write its state
    at all — the same failure the optimized_modules comment in the Dockerfile records.
    """
    svc = _scheduler(_load(BASE_COMPOSE))
    state_dir = str(Path(_schedule_path(svc)).parent)

    dockerfile = DOCKERFILE.read_text()
    mkdir_blocks = re.findall(r"RUN mkdir -p((?:.*?)(?:\\\n.*?)*)\n", dockerfile)
    # Token match, not substring: "/app/tmp" must not satisfy a "/tmp" lookup.
    creating = [block for block in mkdir_blocks if state_dir in block.replace("\\", " ").split()]

    assert len(creating) >= 2, (
        f"{state_dir} must be created in BOTH Dockerfile stages (development and "
        f"production) so the celerybeat_state volume is initialized e2i-owned; found "
        f"{len(creating)} of {len(mkdir_blocks)} `RUN mkdir -p` blocks mentioning it."
    )


def test_scheduler_still_does_not_inherit_common_env() -> None:
    """#528-A: the scheduler sets its env explicitly and must NOT merge *common-env.

    PyYAML resolves the ``<<:`` merge key at parse time, so a service that merged the
    anchor would come back with all ~49 common-env keys in its environment dict. The
    scheduler must keep only its own handful — #1645 touches volumes and the beat
    schedule, never env, and this pins that.
    """
    compose = _load(BASE_COMPOSE)
    common_env = set((compose.get("x-common-env") or {}).keys())
    assert len(common_env) >= 20, (
        f"x-common-env unexpectedly small ({len(common_env)} keys) — this guard would "
        "be vacuous; did the anchor move or get renamed?"
    )

    scheduler_env = set((_scheduler(compose).get("environment") or {}).keys())
    leaked = common_env - {
        "ENVIRONMENT",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "SUPABASE_URL",
        "SUPABASE_KEY",
    }
    assert not (scheduler_env & leaked), (
        "the scheduler picked up x-common-env keys it does not set explicitly, which "
        f"means it now merges <<: *common-env — reverting #528-A. Leaked: "
        f"{sorted(scheduler_env & leaked)}"
    )
    assert scheduler_env, "scheduler lost its explicit environment block"
