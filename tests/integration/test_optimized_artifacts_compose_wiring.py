"""Codify the docker-compose wiring for the DSPy optimization artifact roots.

Discovered during the DSPy-lane A/B (docs/reports/dspy_lane_ab_20260718.md
section 7): ``optimized_modules/`` (GEPA module saves via
src/optimization/gepa/versioning.py + the daily trigger's state file) and
``optimized_prompts/`` (recipient prompt bundles,
src/agents/feedback_learner/prompt_bundles.py BUNDLE_ROOT) are CWD-relative
paths under /app, gitignored, absent from every Dockerfile COPY and compose
mount. The daily ``dspy-prompt-optimization-daily`` beat task (analytics
queue -> worker_medium) would write its outputs into the worker's ephemeral
container filesystem, invisible to the read-only api container whose startup
``install_all_prompt_bundles()`` (src/api/main.py) and
``_load_optimized_pattern_module`` (pattern_analyzer) are the intended
consumers. These named volumes are the persistence handshake between them.

Mirrors tests/integration/test_audit_artifacts_compose_wiring.py: reads the
raw docker-compose.yml as YAML (no docker daemon needed) and asserts the
structure directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker" / "docker-compose.yml"
_DOCKERFILE_PATH = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"

# Volume name -> container mount target. Targets are /app-relative because the
# code addresses both roots with CWD-relative defaults (WORKDIR /app):
# "./optimized_modules" (versioning.py) and "optimized_prompts" (BUNDLE_ROOT).
_ROOTS = {
    "optimized_modules": "/app/optimized_modules",
    "optimized_prompts": "/app/optimized_prompts",
}


def _load_compose() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_COMPOSE_PATH.read_text()))


def _service_mounts(compose: dict[str, Any], service: str) -> list[str]:
    body = (compose.get("services", {}) or {}).get(service) or {}
    return [v for v in body.get("volumes") or [] if isinstance(v, str)]


def test_optimized_artifact_volumes_are_declared():
    """Both named volumes must be declared top-level with pinned docker-level
    names so cross-stack references stay stable (audit_artifacts idiom)."""
    compose = _load_compose()
    volumes = compose.get("volumes", {}) or {}
    for vol in _ROOTS:
        assert vol in volumes, (
            f"{vol} volume missing from compose volumes block. Found: {sorted(volumes)}"
        )
        assert (volumes[vol] or {}).get("name") == f"e2i_{vol}", (
            f"{vol} must pin docker-level name e2i_{vol}; got {(volumes[vol] or {}).get('name')!r}"
        )


def test_worker_medium_mounts_both_roots_writable():
    """worker_medium is the producer: the daily beat task (analytics queue)
    saves GEPA modules + trigger state and prompt bundles. A ``:ro`` suffix
    would silently break every save (all writers catch-and-warn)."""
    compose = _load_compose()
    mounts = _service_mounts(compose, "worker_medium")
    for vol, target in _ROOTS.items():
        accepted = {f"{vol}:{target}", f"{vol}:{target}:rw"}
        assert any(m in accepted for m in mounts), (
            f"worker_medium must mount {vol} read-write at {target}; mounts: {mounts}"
        )


def test_api_mounts_both_roots_read_only():
    """api is a pure consumer today (pattern_analyzer load +
    install_all_prompt_bundles at startup); its only potential writer is the
    dormant ChatbotOptimizer GEPA path (chatbot_dspy.py, no router — issue
    #1282). ``:ro`` codifies least privilege; wiring an api-side writer later
    must consciously flip this test."""
    compose = _load_compose()
    mounts = _service_mounts(compose, "api")
    for vol, target in _ROOTS.items():
        assert f"{vol}:{target}:ro" in mounts, (
            f"api must mount {vol} read-only at {target}; mounts: {mounts}"
        )


def test_dockerfile_creates_both_roots():
    """The image must mkdir both roots BEFORE the chown -R e2i:e2i /app so a
    fresh named volume is initialized with e2i ownership — without this the
    engine creates the mountpoint root-owned and every non-root worker write
    fails (caught-and-warned, i.e. silently)."""
    dockerfile = _DOCKERFILE_PATH.read_text()
    for target in _ROOTS.values():
        assert target in dockerfile, (
            f"docker/Dockerfile must mkdir {target} (in BOTH stages) so named-volume "
            f"initialization copies e2i ownership"
        )
