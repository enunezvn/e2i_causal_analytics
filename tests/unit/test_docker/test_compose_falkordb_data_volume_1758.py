"""CI guard: FalkorDB's data dir must live on a named volume (#1758).

The ``falkordb/falkordb`` image declares ``VOLUME /data`` but its entrypoint
(``run.sh``) starts redis with ``--dir "${FALKORDB_DATA_PATH}"``, and the image
default is ``FALKORDB_DATA_PATH=/var/lib/falkordb/data`` — a container-local
path. The compose file mounted the named volume at ``/data``, so ``dump.rdb``
was written to the container's ephemeral filesystem and RDB persistence was a
no-op: every container *recreation* started an empty graph. The 2026-08-16
deploy (a compose change, PR #1653) recreated ``e2i_falkordb`` and silently
wiped the knowledge graph — /knowledge-graph rendered with zero nodes while
``/api/graph/health`` stayed green (it checks connectivity, not content).

The fix pins ``FALKORDB_DATA_PATH`` on the service so the entrypoint's
``--dir`` lands on the named volume. Like the #1645 beat-state guard this is
not unit-testable end to end (it needs a deploy), so this module pins the
declared wiring, no Docker daemon required:

* the service sets ``FALKORDB_DATA_PATH`` explicitly (the image default is the
  ephemeral trap, so *absence* of the override IS the defect);
* that path sits under a named-volume mount on the service;
* the volume is declared at the top level, so it is a real persistent volume
  rather than an anonymous one.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# tests/unit/test_docker/<this file>  ->  parents[3] == repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_COMPOSE = REPO_ROOT / "docker" / "docker-compose.yml"

FALKORDB = "falkordb"


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


def _falkordb(compose: dict) -> dict:
    svc = (compose.get("services") or {}).get(FALKORDB)
    assert svc, "docker-compose.yml has no `falkordb` service"
    return svc


def _environment(svc: dict) -> dict[str, str]:
    """Service environment as a dict, accepting both list and mapping forms."""
    raw = svc.get("environment")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    env: dict[str, str] = {}
    for entry in raw:
        key, _, value = str(entry).partition("=")
        env[key] = value
    return env


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


def _is_under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def test_falkordb_data_path_is_overridden() -> None:
    """The image default (/var/lib/falkordb/data) is the ephemeral trap (#1758)."""
    env = _environment(_falkordb(_load(BASE_COMPOSE)))
    assert "FALKORDB_DATA_PATH" in env, (
        "the falkordb service must set FALKORDB_DATA_PATH explicitly. Without it the "
        "image's run.sh starts redis with --dir /var/lib/falkordb/data (the image "
        "default), dump.rdb lands on the container filesystem instead of the named "
        "volume, and every container recreation wipes the knowledge graph (#1758 — "
        "the 2026-08-16 deploy did exactly that)."
    )


def test_falkordb_data_path_sits_on_a_named_volume() -> None:
    compose = _load(BASE_COMPOSE)
    svc = _falkordb(compose)
    env = _environment(svc)
    data_path = env.get("FALKORDB_DATA_PATH", "/var/lib/falkordb/data")
    mounts = _named_volume_targets(svc)

    holder = next((target for target in mounts if _is_under(data_path, target)), None)
    assert holder is not None, (
        f"FalkorDB's data dir ({data_path}) is not under any named-volume mount on the "
        f"falkordb service. Named-volume mounts present: {sorted(mounts)}. Without a "
        "volume, dump.rdb dies with the container on every deploy-driven recreation "
        "and the graph starts empty (#1758)."
    )

    volume_name = mounts[holder]
    declared = compose.get("volumes") or {}
    assert volume_name in declared, (
        f"the falkordb service mounts '{volume_name}' but it is not declared in the "
        "top-level `volumes:` block, so compose would treat it as undeclared/anonymous."
    )
