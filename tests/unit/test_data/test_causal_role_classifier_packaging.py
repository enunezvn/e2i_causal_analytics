"""The Layer-4 classifier artifact must actually reach the Docker image.

Same shape as #1607 / #600: the artifact is committed and the loader resolves a
path under PROJECT_ROOT, yet the deployed container had no /app/artifacts at
all. Measured 2026-09-22 on the production image: the cause was ONLY a missing
COPY — `.dockerignore` never excluded it, because a slash-less pattern such as
`*.json` matches at the build-context root only (nested
scripts/benchmarks/routing/data/agent_contracts.json is present in the same
image under the same rule; Docker docs: "markdown files under subdirectories
are still included"). The dockerignore check below is therefore a guard against
a FUTURE exclusion (`artifacts/`, `**/*.json`), and `_matches` deliberately
models Docker's root-only semantics for slash-less patterns — do not "fix" it
to gitignore semantics.

Static checks only: they read `.dockerignore` and the Dockerfile, so they run in
the normal unit lane. Spec: docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.data.causal_role_classifier_loader import DEFAULT_ARTIFACT_PATH, PROJECT_ROOT

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile"
_ARTIFACT_REL = DEFAULT_ARTIFACT_PATH.relative_to(PROJECT_ROOT).as_posix()


def _matches(pattern: str, rel_path: str) -> bool:
    pattern = pattern.rstrip("/")
    if not pattern:
        return False
    regex = re.escape(pattern).replace(r"\*\*", "\x00").replace(r"\*", "[^/]*")
    regex = regex.replace("\x00", ".*").replace(r"\?", "[^/]")
    if re.fullmatch(regex, rel_path):
        return True
    return bool(re.fullmatch(regex + "(/.*)?", rel_path))


def _dockerignore_excludes(rel_path: str) -> bool:
    """Last-match-wins, exactly as Docker evaluates it."""
    excluded = False
    for raw in _DOCKERIGNORE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        pattern = line[1:] if negate else line
        if _matches(pattern, rel_path):
            excluded = not negate
    return excluded


def test_the_loader_path_is_relative_to_the_repo_root() -> None:
    """PROJECT_ROOT must be the repo root here and /app in the image; if the loader
    ever moves, the path this guard checks moves with it."""
    assert PROJECT_ROOT == _REPO_ROOT
    assert _ARTIFACT_REL == "artifacts/dspy/causal_role_classifier.json"


def test_the_classifier_artifact_is_committed() -> None:
    assert (_REPO_ROOT / _ARTIFACT_REL).is_file(), f"{_ARTIFACT_REL} is not committed"


def test_the_classifier_artifact_survives_dockerignore() -> None:
    assert not _dockerignore_excludes(_ARTIFACT_REL), (
        f"{_ARTIFACT_REL} is excluded from the Docker build context by .dockerignore; "
        "remove or narrow the rule that excludes it (a slash-less pattern only matches "
        "the root, so look for a directory or `**/` rule), otherwise Layer 4 silently "
        "skips in production."
    )


def test_matcher_models_docker_root_only_semantics_for_slashless_patterns() -> None:
    """Measured on the production image 2026-09-22: `*.json` did not drop nested
    scripts/benchmarks/routing/data/agent_contracts.json. `**/*.json` would."""
    assert _matches("*.json", "root.json")
    assert not _matches("*.json", "artifacts/dspy/causal_role_classifier.json")
    assert _matches("**/*.json", "artifacts/dspy/causal_role_classifier.json")
    assert _matches("artifacts", "artifacts/dspy/causal_role_classifier.json")


def test_dockerfile_copies_the_artifact_into_every_app_stage() -> None:
    """Both `development` and `production` are separate FROMs; each needs its own COPY."""
    text = _DOCKERFILE.read_text()
    copies = [
        ln
        for ln in text.splitlines()
        if ln.startswith("COPY") and "artifacts/dspy/causal_role_classifier.json" in ln
    ]
    assert len(copies) >= 2, (
        "expected a causal_role_classifier.json COPY in BOTH the development and "
        f"production stages of docker/Dockerfile; found {len(copies)}: {copies}"
    )


def test_the_copy_lands_where_the_loader_looks() -> None:
    """`COPY <src> ./artifacts/dspy/` must place the file at /app/artifacts/dspy/…"""
    text = _DOCKERFILE.read_text()
    for ln in text.splitlines():
        if ln.startswith("COPY") and "causal_role_classifier.json" in ln:
            dest = ln.split()[-1]
            assert dest in ("./artifacts/dspy/", "./artifacts/dspy/causal_role_classifier.json"), (
                f"COPY destination {dest!r} does not match the loader's "
                f"PROJECT_ROOT/artifacts/dspy/ layout"
            )
