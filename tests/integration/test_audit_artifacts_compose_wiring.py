"""Codify the docker-compose wiring for the Layer-4 audit_artifacts volume.

Plan: .claude/plans/layer4_evaluator_audit_consumer.md Task 11 Step 3
(Codex Gate-2 MED-4 forcing function). Ralph-loop CORRECTION-1
(2026-05-15): the original Task 11 Step 3 was run interactively at the
shell once but never codified — any future PR that renames the volume
or removes a mount would silently break acceptance criterion #1 with
no test to catch it. This test is that test.

Reads the raw docker-compose.yml as YAML (no docker daemon needed, so
it runs cleanly in CI) and asserts the structure directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker" / "docker-compose.yml"


def _load_compose() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_COMPOSE_PATH.read_text()))


def test_audit_artifacts_volume_is_declared():
    """The audit_artifacts named volume must be declared at the top
    level so docker compose can create it."""
    compose = _load_compose()
    volumes = compose.get("volumes", {}) or {}
    assert "audit_artifacts" in volumes, (
        f"audit_artifacts volume missing from compose volumes block. Found: {sorted(volumes)}"
    )
    # Pin the docker-level name so cross-stack references (backup script,
    # observability dashboards) stay stable.
    assert volumes["audit_artifacts"].get("name") == "e2i_audit_artifacts"


def test_audit_artifacts_env_var_in_common_env_anchor():
    """The env var must live on the x-common-env anchor so every service
    that merges *common-env inherits it. Plan ralph-loop refinement: the
    var must NOT live on x-common-worker (no environment block there)."""
    compose = _load_compose()
    common_env = compose.get("x-common-env") or {}
    assert common_env.get("ADAPTIVE_VALIDITY_ARTIFACTS_DIR") == "/app/data/audit_artifacts", (
        "ADAPTIVE_VALIDITY_ARTIFACTS_DIR is not set to "
        "/app/data/audit_artifacts on the x-common-env anchor; "
        f"got: {common_env.get('ADAPTIVE_VALIDITY_ARTIFACTS_DIR')!r}"
    )


def _services_with_common_env_merge(compose: dict) -> list[str]:
    """Return service names whose `environment:` block merges
    `<<: *common-env`. PyYAML resolves the merge key, so after parsing
    every such service's environment dict already CONTAINS the common-env
    keys — we use presence of ADAPTIVE_VALIDITY_ARTIFACTS_DIR as the
    proxy for membership."""
    services = compose.get("services", {}) or {}
    out: list[str] = []
    for name, body in services.items():
        env = (body or {}).get("environment") or {}
        if isinstance(env, dict) and (
            env.get("ADAPTIVE_VALIDITY_ARTIFACTS_DIR") == "/app/data/audit_artifacts"
        ):
            out.append(name)
    return out


def _services_with_audit_mount(compose: dict) -> list[str]:
    """Return service names that mount audit_artifacts at
    /app/data/audit_artifacts."""
    services = compose.get("services", {}) or {}
    out: list[str] = []
    for name, body in services.items():
        for vol in (body or {}).get("volumes") or []:
            if isinstance(vol, str) and (
                vol == "audit_artifacts:/app/data/audit_artifacts"
                or vol.startswith("audit_artifacts:/app/data/audit_artifacts:")
            ):
                out.append(name)
                break
    return out


def test_audit_artifacts_mounted_on_every_common_env_service():
    """Every service that gets the env var via *common-env MUST also mount
    the volume — otherwise the producer would try to mkdir under a path
    that doesn't exist in the container filesystem.

    Plan acceptance criterion #1; Risk R4 in the plan.
    """
    compose = _load_compose()
    services_with_env = set(_services_with_common_env_merge(compose))
    services_with_mount = set(_services_with_audit_mount(compose))
    missing_mount = services_with_env - services_with_mount
    assert not missing_mount, (
        f"Services with ADAPTIVE_VALIDITY_ARTIFACTS_DIR set via *common-env "
        f"but no audit_artifacts mount: {sorted(missing_mount)}. "
        f"This would cause the producer to fail mkdir'ing the sidecar dir."
    )


def test_audit_artifacts_wiring_covers_expected_service_set():
    """Pin the expected service set as of 2026-05-15: api + 3 worker
    tiers. If a new service is added or one is removed, this test
    fails so a human reviews whether the audit trail should follow.
    """
    compose = _load_compose()
    services_with_mount = sorted(_services_with_audit_mount(compose))
    expected = ["api", "worker_heavy", "worker_light", "worker_medium"]
    assert services_with_mount == expected, (
        f"Expected audit_artifacts mounted on exactly {expected}, "
        f"got {services_with_mount}. If a service was added or removed, "
        f"update this assertion and confirm the audit trail follows."
    )
