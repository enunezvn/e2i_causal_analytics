"""CI guard: every writable named volume under ``/app`` has an e2i-owned mountpoint (#2273).

The api and worker containers run as the non-root ``e2i`` user (uid 1000). Docker
initializes an empty named volume from the image directory it is mounted over,
ownership included. When that directory is ABSENT from the image, the engine
creates the mountpoint as ``root:root 755`` and the volume inherits it, so uid 1000
cannot write a single file into it.

That happened to ``audit_artifacts`` (#2273): the adaptive-validity sidecar, which
#238 names the canonical audit record, was never written on prod. Measured
2026-09-23: ``/app/data/audit_artifacts`` was ``root:root 755`` in ``e2i_api`` and
``worker_medium``, held 0 files since 2026-05-15, and the writer's ``except``
turned every PermissionError into a WARN. Both ``RUN mkdir -p`` lists in
``docker/Dockerfile`` created every other ``/app/data`` volume directory but not
this one. ``celerybeat`` (#1645) and ``optimized_modules`` hit the same trap one at
a time; this guard covers the whole class instead of one path per incident.

Hermetic: the compose files and the Dockerfile are parsed as text, no daemon and
no build. The named-volume semantics this relies on were measured on the prod
image (1849f66d2) with throwaway volumes: an empty volume mounted over an
e2i-owned image directory comes up ``e2i:e2i``; mounted over an absent path it
comes up ``root:root`` and ``touch`` fails with Permission denied.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# tests/unit/test_docker/<this file>  ->  parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"
# The dev overlay retargets the same services at the ``development`` stage and
# merges its own mounts on top of the base ones, so both files feed the check.
DEV_COMPOSE = REPO_ROOT / "docker" / "docker-compose.dev.yml"

APP_DOCKERFILE = "docker/Dockerfile"
RUNTIME_STAGES = ("development", "production")


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


def _app_image_services(*composes: dict) -> set[str]:
    """Services that build the app image from ``docker/Dockerfile`` in any of the files.

    The dev overlay's retargeted services inherit ``dockerfile`` from the base file;
    its dev-only ones (``falkordb-seeder``, ``test``) declare it themselves.
    """
    out: set[str] = set()
    for compose in composes:
        for name, body in (compose.get("services") or {}).items():
            build = (body or {}).get("build")
            if isinstance(build, dict) and build.get("dockerfile") == APP_DOCKERFILE:
                out.add(name)
    return out


def _writable_named_volume_targets(svc: dict) -> dict[str, str]:
    """``{container_path: volume_name}`` for writable named-volume mounts (not binds).

    ``:ro`` mounts are skipped: ownership only decides whether uid 1000 can WRITE.
    """
    targets: dict[str, str] = {}
    for vol in (svc or {}).get("volumes") or []:
        if isinstance(vol, str):
            parts = vol.split(":")
            if len(parts) < 2:
                continue
            source, target = parts[0], parts[1]
            if source.startswith((".", "/", "~", "$")):
                continue
            if len(parts) > 2 and "ro" in parts[2].split(","):
                continue
            targets[target] = source
        elif isinstance(vol, dict) and vol.get("type") == "volume" and vol.get("source"):
            if vol.get("read_only"):
                continue
            targets[str(vol["target"])] = str(vol["source"])
    return targets


def _required_mountpoints() -> dict[str, set[str]]:
    """``{container_path: {"service:volume", ...}}`` for app-image writable volumes under /app."""
    base = _load(BASE_COMPOSE)
    dev = _load(DEV_COMPOSE)
    app_services = _app_image_services(base, dev)
    required: dict[str, set[str]] = {}
    for compose in (base, dev):
        for name, body in (compose.get("services") or {}).items():
            if name not in app_services:
                continue
            for target, volume in _writable_named_volume_targets(body).items():
                if target == "/app" or target.startswith("/app/"):
                    required.setdefault(target, set()).add(f"{name}:{volume}")
    return required


def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations, drop blank + comment lines."""
    out: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1].strip() + " "
            continue
        buf += stripped
        out.append(buf.strip())
        buf = ""
    if buf:
        out.append(buf.strip())
    return out


def _stage_lines() -> dict[str, list[str]]:
    """``{stage_name: [logical instruction lines]}`` for every ``FROM … AS <name>``."""
    stages: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in _logical_lines(DOCKERFILE.read_text()):
        tokens = line.split()
        if tokens[0].upper() == "FROM":
            name = tokens[3] if len(tokens) >= 4 and tokens[2].upper() == "AS" else tokens[1]
            current = stages.setdefault(name, [])
            continue
        if current is not None:
            current.append(line)
    return stages


def _mkdir_paths(lines: list[str]) -> set[str]:
    """Every path a ``RUN mkdir -p`` in the stage creates."""
    paths: set[str] = set()
    for line in lines:
        tokens = line.split()
        if tokens[:3] == ["RUN", "mkdir", "-p"]:
            for token in tokens[3:]:
                if token in {"&&", ";"}:
                    break
                paths.add(token.rstrip("/"))
    return paths


def _chowns_app_to_e2i(lines: list[str]) -> bool:
    return any("chown -R e2i:e2i /app" in line for line in lines)


def test_guard_sees_the_known_volumes() -> None:
    """Anti-vacuity: the compose parse must find the app volumes this guard exists for."""
    required = _required_mountpoints()
    for path in ("/app/data/ml_artifacts", "/app/data/celerybeat", "/app/data/audit_artifacts"):
        assert path in required, (
            f"expected {path} among the app-image writable named-volume mountpoints; "
            f"found {sorted(required)}. Did the compose layout change?"
        )


def test_every_writable_app_volume_mountpoint_is_created_in_each_runtime_stage() -> None:
    stages = _stage_lines()
    required = _required_mountpoints()
    missing: list[str] = []
    for stage in RUNTIME_STAGES:
        lines = stages.get(stage)
        assert lines is not None, f"docker/Dockerfile has no `{stage}` stage"
        assert _chowns_app_to_e2i(lines), (
            f"the `{stage}` stage no longer runs `chown -R e2i:e2i /app`, so creating the "
            "mountpoints does not make them e2i-owned."
        )
        created = _mkdir_paths(lines)
        for path, users in sorted(required.items()):
            if path not in created:
                missing.append(f"{stage}: {path} (mounted by {', '.join(sorted(users))})")
    assert not missing, (
        "named-volume mountpoints absent from a Dockerfile runtime stage. Docker creates "
        "an absent mountpoint root:root 755, the volume inherits it, and the non-root e2i "
        "user (uid 1000) cannot write to it (#2273: audit_artifacts, never written on "
        "prod). Add each path to that stage's `RUN mkdir -p` list:\n  " + "\n  ".join(missing)
    )


def test_dev_only_app_image_services_are_in_scope() -> None:
    """codex r1 LOW: services that exist only in the dev overlay but build the app image
    (``falkordb-seeder``, ``test``) must feed the check too, not just the base file's."""
    base, dev = _load(BASE_COMPOSE), _load(DEV_COMPOSE)
    dev_only = {
        name
        for name, body in (dev.get("services") or {}).items()
        if name not in (base.get("services") or {})
        and isinstance((body or {}).get("build"), dict)
        and body["build"].get("dockerfile") == APP_DOCKERFILE
    }
    assert dev_only, "expected dev-only app-image services; did the overlay change?"
    assert dev_only <= _app_image_services(base, dev)
