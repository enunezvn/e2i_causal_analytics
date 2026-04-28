"""Block 6B-infra-4: Compose worker / scheduler ↔ supabase-network attachment.

The Feast offline store and the canonical Tier-0 ETL tables both live on
``supabase-db``, which sits on the external ``supabase-network``. For Celery
workers and the beat scheduler to run materialize jobs and ETL drift checks,
each of those containers must be a member of ``supabase-network``.

This test inspects each running worker / scheduler container with
``docker inspect`` and asserts ``supabase-network`` appears in its
``NetworkSettings.Networks`` mapping. It is deliberately *live-only* — we
skip cleanly when:

    1. The ``docker`` CLI is not on PATH (slim CI runners).
    2. The Docker daemon is unreachable (no socket, no permissions).
    3. The target service has no running container under the active compose
       project (developer hasn't run ``up`` for it, or replicas: 0).

Resolution strategy: use ``docker compose -f <files> ps -q <service>`` to map
service name → container ID. This handles both:

    * services with explicit ``container_name:`` (e.g. ``worker_medium``,
      ``scheduler``)
    * services without a ``container_name`` (e.g. ``worker_light``, which
      has ``replicas: 2`` and so cannot use ``container_name``)

We do NOT inspect ``worker_heavy`` because base compose pins it to
``replicas: 0`` (on-demand), so its container is normally absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Compose stack used in dev (matches CLAUDE.md production architecture note —
# dev and prod are the same machine, same compose files). The Opik overlay is
# included to keep the active compose project consistent with how the stack
# is actually run.
COMPOSE_FILES = [
    "-f",
    str(PROJECT_ROOT / "docker" / "docker-compose.yml"),
    "-f",
    str(PROJECT_ROOT / "docker" / "docker-compose.dev.yml"),
    "-f",
    str(PROJECT_ROOT / "docker" / "docker-compose.opik.yml"),
]

REQUIRED_NETWORK = "supabase-network"

# Service names (NOT container_names — some have replicas and so no fixed
# container_name). worker_heavy is intentionally excluded: replicas: 0.
WORKER_SERVICES = ["worker_light", "worker_medium", "scheduler"]


@pytest.fixture(scope="module")
def docker_available() -> None:
    """Skip the module if the docker CLI / daemon isn't usable."""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not on PATH")
    probe = subprocess.run(
        ["docker", "ps"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(f"docker daemon unreachable: {probe.stderr.strip()[:200]}")


def _resolve_container_id(service: str) -> str | None:
    """Return one running container ID for the compose service, or None.

    ``docker compose ps -q <service>`` prints zero or more IDs (one per
    replica). We take the first one — for the network-attachment check we
    only need to verify any single replica.
    """
    cmd = ["docker", "compose"] + COMPOSE_FILES + ["ps", "-q", service]
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ids[0] if ids else None


@pytest.mark.parametrize("service", WORKER_SERVICES)
def test_worker_attached_to_supabase_network(
    docker_available: Any,  # noqa: ARG001 — fixture is the gate
    service: str,
) -> None:
    """Each worker / scheduler must join supabase-network for materialize + ETL access.

    Block 6B-infra-4: workers run Feast materialize and ETL tasks that hit
    the canonical Feast tables on supabase-db. Without supabase-network
    attachment those tasks fail with DNS-resolution errors.
    """
    container_id = _resolve_container_id(service)
    if container_id is None:
        pytest.skip(
            f"compose service {service!r} has no running container (stack not up, or replicas: 0)"
        )

    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            container_id,
            "--format",
            "{{json .NetworkSettings.Networks}}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if inspect.returncode != 0:
        pytest.skip(
            f"docker inspect failed for {service} (container_id={container_id}): "
            f"stderr={inspect.stderr.strip()[:200]!r}"
        )

    networks = json.loads(inspect.stdout)
    assert REQUIRED_NETWORK in networks, (
        f"service {service!r} (container {container_id}) expected to be attached "
        f"to {REQUIRED_NETWORK!r}; actual networks: {sorted(networks.keys())}"
    )
