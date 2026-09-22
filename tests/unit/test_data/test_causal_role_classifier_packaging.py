"""The Layer-4 classifier artifact must actually reach the Docker image.

Same shape as #1607 / #600: the artifact is committed and the loader resolves a
path under PROJECT_ROOT, yet the deployed container had no /app/artifacts at
all. Measured 2026-09-22 on the production image: the cause was ONLY a missing
COPY — `.dockerignore` never excluded it, because a slash-less pattern such as
`*.json` matches at the build-context root only (nested
scripts/benchmarks/routing/data/agent_contracts.json is present in the same
image under the same rule; Docker docs: "markdown files under subdirectories
are still included"). The dockerignore check below is therefore a guard against
a FUTURE exclusion (`artifacts/`, `**/*.json`), and the shared
`tests.unit.test_docker.dockerignore_semantics.matches` helper deliberately
models Docker's root-only semantics for slash-less patterns — do not "fix" it
to gitignore semantics.

Static checks only: they read `.dockerignore` and the Dockerfile, so they run in
the normal unit lane. Spec: docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.data.causal_role_classifier_loader import DEFAULT_ARTIFACT_PATH, PROJECT_ROOT
from tests.unit.test_docker.dockerignore_semantics import dockerignore_excludes, matches
from tests.unit.test_docker.test_deploy_trigger_covers_image_inputs_1783 import (
    _image_input_paths,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile"
_ARTIFACT_REL = DEFAULT_ARTIFACT_PATH.relative_to(PROJECT_ROOT).as_posix()


def test_the_loader_path_is_relative_to_the_repo_root() -> None:
    """PROJECT_ROOT must be the repo root here and /app in the image; if the loader
    ever moves, the path this guard checks moves with it."""
    assert PROJECT_ROOT == _REPO_ROOT
    assert _ARTIFACT_REL == "artifacts/dspy/causal_role_classifier.json"


def test_the_classifier_artifact_is_committed() -> None:
    assert (_REPO_ROOT / _ARTIFACT_REL).is_file(), f"{_ARTIFACT_REL} is not committed"
    # `is_file()` alone is also satisfied by an untracked file sitting on disk (e.g. a
    # local rebuild artifact); assert it is actually tracked by git, not merely present.
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", _ARTIFACT_REL],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0, (
        f"{_ARTIFACT_REL} exists on disk but is not git-tracked: {tracked.stderr.strip()}"
    )


def test_the_classifier_artifact_survives_dockerignore() -> None:
    assert not dockerignore_excludes(_DOCKERIGNORE, _ARTIFACT_REL), (
        f"{_ARTIFACT_REL} is excluded from the Docker build context by .dockerignore; "
        "remove or narrow the rule that excludes it (a slash-less pattern only matches "
        "the root, so look for a directory or `**/` rule), otherwise Layer 4 silently "
        "skips in production."
    )


def test_matcher_models_docker_root_only_semantics_for_slashless_patterns() -> None:
    """Measured on the production image 2026-09-22: `*.json` did not drop nested
    scripts/benchmarks/routing/data/agent_contracts.json. `**/*.json` would."""
    assert matches("*.json", "root.json")
    assert not matches("*.json", "artifacts/dspy/causal_role_classifier.json")
    assert matches("**/*.json", "artifacts/dspy/causal_role_classifier.json")
    assert matches("artifacts", "artifacts/dspy/causal_role_classifier.json")
    # `**/` (moby's compiler) matches ZERO OR MORE path segments, so it must also
    # match a root-level file with no directory to consume — a leading `**/*.json`
    # covers `root.json` exactly like a bare `*.json` would.
    assert matches("**/*.json", "root.json")


def test_dockerfile_copies_the_artifact_into_every_app_stage() -> None:
    """Both `development` and `production` are separate FROMs; each needs a COPY that is
    an image input of ITS OWN reachable stage closure — not merely >=2 COPY lines
    anywhere in the file, which a planted-failure review proved passes even when both
    lines sit in the same stage (see the PR description for the planted-failure output).
    Reuses the #1783 guard's stage-closure parser rather than a second, weaker one."""
    text = _DOCKERFILE.read_text()
    for stage in ("production", "development"):
        inputs = _image_input_paths(text, root=stage)
        assert _ARTIFACT_REL in inputs, (
            f"{_ARTIFACT_REL} is not an image input reachable from the {stage!r} stage "
            f"(found: {sorted(inputs)}); it must be COPYed directly in that stage (or a "
            "stage its closure pulls the file from), not merely somewhere else in the "
            "Dockerfile."
        )


def test_the_copy_lands_where_the_loader_looks() -> None:
    """`COPY <src> ./artifacts/dspy/` must place the file at /app/artifacts/dspy/…

    Collects the matching lines FIRST and asserts the collection is non-empty before
    checking destinations: a bare `for ln in ...: assert ...` loop passes vacuously
    when there are zero matching lines (proven — deleting both COPY lines left only
    the count-based test failing; see the PR description for that output)."""
    text = _DOCKERFILE.read_text()
    copies = [
        ln
        for ln in text.splitlines()
        if ln.startswith("COPY") and "causal_role_classifier.json" in ln
    ]
    assert copies, "no COPY of the artifact found in docker/Dockerfile"
    for ln in copies:
        dest = ln.split()[-1]
        assert dest in ("./artifacts/dspy/", "./artifacts/dspy/causal_role_classifier.json"), (
            f"COPY destination {dest!r} does not match the loader's "
            f"PROJECT_ROOT/artifacts/dspy/ layout"
        )
