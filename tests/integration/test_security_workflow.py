"""Codify the .github/workflows/security.yml invariants for Issue #233.

Issue #233 (Container Scanning Trivy failing on main + docs-only PRs):
Root cause confirmed from run 25931172542 logs — the heavy ML image
(torch + cuda + transformers + scikit-survival) plus Trivy's docker-
daemon image export blow past the runner's free disk space mid-scan:

    Free space left: 10 MB
    FATAL ... no space left on device

`jlumbroso/free-disk-space@main` reclaims ~30 GB before docker build,
but Trivy then issues `docker save` to materialize an uncompressed
tar of every layer, and the export overflows /var/lib/docker/tmp.

The fix has four parts, each codified below as a forcing function:

1. Bump the scan `timeout` from 15m → 25m so the heavy image has
   headroom even on slow runners.
2. Expand `skip-dirs` past `/app/.venv` to include cache/build/test
   directories that wouldn't contribute new OS- or library-level
   findings but add I/O cost (`.cache`, `.mypy_cache`, `.pytest_cache`,
   `frontend/node_modules`, `tests`, `docs`).
3. Add a post-build cleanup step that prunes intermediate buildx
   state + dangling images right before Trivy runs (frees the
   docker-export staging area).
4. Pin `TRIVY_DB_REPOSITORY` to a mirror to side-step transient
   ghcr.io rate-limiting (defence-in-depth for category D).

Reads the raw workflow YAML (no GitHub Actions runner needed) and
asserts the post-fix state directly. Companion test pattern to
tests/integration/test_audit_artifacts_compose_wiring.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "security.yml"


def _load_workflow() -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(_WORKFLOW_PATH.read_text()))


def _container_scan_steps() -> list[dict[str, Any]]:
    workflow = _load_workflow()
    jobs = workflow["jobs"]
    assert "container-scan" in jobs, f"container-scan job missing. Found jobs: {sorted(jobs)}"
    steps = jobs["container-scan"]["steps"]
    assert isinstance(steps, list) and steps, "container-scan has no steps"
    return cast(list[dict[str, Any]], steps)


def _find_step(steps: list[dict[str, Any]], name_fragment: str) -> dict[str, Any]:
    for step in steps:
        name = step.get("name", "")
        if name_fragment in name:
            return step
    available = [s.get("name", "<unnamed>") for s in steps]
    raise AssertionError(f"No step matching {name_fragment!r}; have: {available}")


def test_workflow_yaml_parses_cleanly() -> None:
    """Sanity gate: the workflow must be valid YAML."""
    workflow = _load_workflow()
    assert isinstance(workflow, dict)
    assert "jobs" in workflow


def test_trivy_timeout_raised_to_25m() -> None:
    """Issue #233 fix part 1: 15m wasn't enough for the heavy ML image
    plus DB pull plus layer export; the failing run took ~13m of scan
    time before hitting the disk-space wall. Give 25m of headroom."""
    steps = _container_scan_steps()
    trivy = _find_step(steps, "Run Trivy")
    timeout = trivy["with"]["timeout"]
    assert timeout == "25m", (
        f"Trivy timeout must be 25m for Issue #233 (was {timeout!r}). "
        "15m proved insufficient on slow runners."
    )


def test_trivy_skip_dirs_expanded_for_disk_pressure() -> None:
    """Issue #233 fix part 2: expand skip-dirs beyond /app/.venv to
    keep Trivy from walking build/test/cache trees that won't surface
    new OS- or library-CVEs but add I/O + disk pressure on the runner.

    skip-dirs is a comma-separated string per the trivy-action contract.
    Each required entry must be present; ordering is irrelevant."""
    steps = _container_scan_steps()
    trivy = _find_step(steps, "Run Trivy")
    skip_dirs_raw = trivy["with"]["skip-dirs"]
    assert isinstance(skip_dirs_raw, str), (
        f"skip-dirs must be a string, got {type(skip_dirs_raw).__name__}"
    )
    skip_dirs = {p.strip() for p in skip_dirs_raw.split(",") if p.strip()}
    required = {
        "/app/.venv",
        "/app/.cache",
        "/app/.mypy_cache",
        "/app/.pytest_cache",
        "/app/frontend/node_modules",
        "/app/tests",
        "/app/docs",
    }
    missing = required - skip_dirs
    assert not missing, (
        f"Issue #233 fix requires these skip-dirs entries: {sorted(missing)}. "
        f"Currently configured: {sorted(skip_dirs)}"
    )


def test_post_build_disk_cleanup_step_present() -> None:
    """Issue #233 fix part 3: between the docker build and Trivy run,
    purge intermediate buildx state + dangling images so Trivy's
    `docker save` staging area (/var/lib/docker/tmp/docker-export-*)
    has room. The failing run hit 10 MB free mid-scan.

    We pin two invariants:
      - the step exists with a recognizable name,
      - it runs after `docker-build` and before `Run Trivy`,
      - it invokes `docker system prune` or `docker builder prune`."""
    steps = _container_scan_steps()
    build_idx: int | None = None
    cleanup_idx: int | None = None
    trivy_idx: int | None = None
    for i, step in enumerate(steps):
        name = step.get("name", "")
        if "Build Docker image" in name:
            build_idx = i
        elif "Reclaim disk after build" in name or "Prune docker" in name:
            cleanup_idx = i
        elif "Run Trivy" in name:
            trivy_idx = i

    assert build_idx is not None, "Expected a 'Build Docker image' step"
    assert trivy_idx is not None, "Expected a 'Run Trivy' step"
    assert cleanup_idx is not None, (
        "Issue #233 fix requires a post-build / pre-Trivy disk-cleanup "
        "step (name must contain 'Reclaim disk after build' or "
        "'Prune docker'). Trivy's docker-daemon image export needs "
        "/var/lib/docker/tmp headroom."
    )
    assert build_idx < cleanup_idx < trivy_idx, (
        f"Cleanup step out of order: build={build_idx}, cleanup={cleanup_idx}, trivy={trivy_idx}"
    )
    cleanup_run = steps[cleanup_idx].get("run", "")
    assert "docker" in cleanup_run and "prune" in cleanup_run, (
        f"Cleanup step must run docker prune; got: {cleanup_run!r}"
    )
    # Codex MED follow-up (#233 review): prune is best-effort cleanup —
    # if `docker builder prune` exits non-zero (e.g. nothing to prune
    # on a fresh runner) we MUST NOT fail the whole scan. Every prune
    # command in the cleanup body must therefore be guarded with
    # `|| true`. Counting `|| true` occurrences is a cheap forcing
    # function: each prune line should have its own guard.
    prune_lines = [line.strip() for line in cleanup_run.splitlines() if "prune" in line]
    assert prune_lines, "Expected at least one prune line in cleanup step"
    for line in prune_lines:
        assert "|| true" in line, (
            f"Cleanup prune line missing `|| true` guard: {line!r}. "
            "A prune exit-code must not take down the Trivy scan."
        )


def test_trivy_db_repository_pinned_for_rate_limit_resilience() -> None:
    """Issue #233 fix part 4 (defence-in-depth for category D — DB pull
    failures from ghcr.io rate-limit). Pin TRIVY_DB_REPOSITORY at the
    job env level so retries hit the GCR mirror Trivy uses by default
    in newer versions (mirror.gcr.io/aquasec/trivy-db).

    Codex MED follow-up (#233 review): pinning a single repo would
    REMOVE the action's default mirror+ghcr fallback. Trivy honors a
    comma-separated list and tries each in order — we keep ghcr.io as
    fallback so a mirror outage doesn't reintroduce category D."""
    workflow = _load_workflow()
    job = workflow["jobs"]["container-scan"]
    env = job.get("env", {}) or {}
    db_repo = env.get("TRIVY_DB_REPOSITORY", "")
    assert "trivy-db" in db_repo, (
        f"container-scan job env must pin TRIVY_DB_REPOSITORY to a "
        f"trivy-db mirror (got {db_repo!r}). Defends against transient "
        "ghcr.io rate-limit (Issue #233 category D)."
    )
    # Fallback invariant: must have BOTH mirror.gcr.io and ghcr.io
    # entries (comma-separated) so we're not single-point-of-failure
    # on either registry.
    repos = {r.strip() for r in db_repo.split(",") if r.strip()}
    has_mirror = any("mirror.gcr.io" in r for r in repos)
    has_ghcr = any("ghcr.io" in r for r in repos)
    assert has_mirror and has_ghcr, (
        f"TRIVY_DB_REPOSITORY must list both mirror.gcr.io and ghcr.io "
        f"as comma-separated fallback entries (got {sorted(repos)}). "
        "Pinning a single repo removes Trivy's default fallback."
    )


def test_cleanup_step_name_is_quoted_so_it_parses_intact() -> None:
    """Codex LOW follow-up (#233 review): an unquoted YAML scalar
    containing `(Issue #233)` gets parsed as `Reclaim disk after build
    (Issue` — the `#` after the `(` starts a YAML comment. Quote the
    name so the Actions UI shows the issue reference intact."""
    steps = _container_scan_steps()
    cleanup = _find_step(steps, "Reclaim disk after build")
    name = cleanup.get("name", "")
    assert "#233" in name, (
        f"Cleanup step name lost its `#233` reference (got {name!r}). "
        "Quote the name in YAML so the `#` isn't treated as a comment."
    )


def test_trivy_action_pin_unchanged() -> None:
    """Anti-regression: PR #22 pinned trivy-action to v0.36.0 because
    @master was silently timing out. Issue #233 fix MUST NOT silently
    bump that pin — if a future change does, it should be a deliberate
    decision with its own evidence."""
    steps = _container_scan_steps()
    trivy = _find_step(steps, "Run Trivy")
    uses = trivy.get("uses", "")
    assert uses == "aquasecurity/trivy-action@v0.36.0", (
        f"trivy-action pin changed unexpectedly: {uses!r}. "
        "PR #22 chose v0.36.0 deliberately; bump in a separate PR."
    )


# PyYAML can't always round-trip workflow YAML when a `run: |` block
# contains a multi-line Python heredoc inside a `python -c "..."` shell
# call — the embedded source ends up at column 0 inside what PyYAML
# (rightly) thinks is still part of the run-block string, and the
# tokenizer fails. GitHub Actions' own parser handles it because it
# does shell-aware block-scalar scoping. These files are well-formed
# on the runner; they just aren't `yaml.safe_load`-parseable.
#
# This set is the load-bearing escape hatch — adding a workflow here
# should be a deliberate choice, not a silent regression. New entries
# must reference the offending heredoc line.
_KNOWN_PYYAML_UNPARSEABLE: frozenset[str] = frozenset(
    {
        # python -c heredoc at lines 282-289 (governance_verifier_sha
        # manifest injection) — runs fine on GitHub Actions, fails
        # `yaml.safe_load`. Out of scope for Issue #233.
        "tier1b_b2_experiment.yml",
    }
)


def test_other_workflows_still_parse() -> None:
    """Belt-and-braces: editing security.yml shouldn't have collateral
    damage on sibling workflows. This sweeps the whole workflows dir.

    Known PyYAML-unparseable files (legitimate run-block heredocs) are
    listed in `_KNOWN_PYYAML_UNPARSEABLE` and skipped — but the test
    asserts every entry in that allow-list still exists, so we notice
    when a known-broken file is finally fixed and can drop the skip."""
    workflows_dir = _WORKFLOW_PATH.parent
    yml_files = sorted(workflows_dir.glob("*.yml"))
    assert yml_files, f"No workflow YAML found under {workflows_dir}"
    actual_names = {p.name for p in yml_files}
    stale_skips = _KNOWN_PYYAML_UNPARSEABLE - actual_names
    assert not stale_skips, (
        f"Stale entries in _KNOWN_PYYAML_UNPARSEABLE (files no longer "
        f"present): {sorted(stale_skips)}. Drop them from the allow-list."
    )
    for path in yml_files:
        if path.name in _KNOWN_PYYAML_UNPARSEABLE:
            continue
        try:
            doc = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:  # pragma: no cover - explicit failure
            raise AssertionError(f"{path.name} failed to parse: {exc}") from exc
        assert isinstance(doc, dict), f"{path.name} is not a mapping"
        assert "jobs" in doc, f"{path.name} has no 'jobs' block"
