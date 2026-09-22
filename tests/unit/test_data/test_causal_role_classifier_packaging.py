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
    _parse_stages,
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
    # Go's `filepath.Match` (what Docker's pattern matcher is built on) supports
    # bracket character classes; the real .dockerignore has `*.py[cod]`.
    assert matches("*.py[cod]", "root.pyc")
    assert not matches("*.py[cod]", "root.py")
    assert not matches("*.py[cod]", "sub/x.pyc")


def test_every_real_dockerignore_pattern_compiles_through_the_matcher() -> None:
    """Conformance: no pattern currently in .dockerignore raises inside `matches`.

    Guards the tokenizer itself (character classes, `**` forms, escaping) against a
    FUTURE .dockerignore edit that adds a glob shape the translator cannot handle —
    such a pattern should fail a match, never raise, in the guards above."""
    for raw in _DOCKERIGNORE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern = line[1:] if line.startswith("!") else line
        matches(pattern, "x")  # must not raise


def _own_copy_sources(dockerfile_text: str, stage_name: str) -> set[str]:
    """Repo paths COPYed DIRECTLY in `stage_name` — its own COPY lines only, never its
    reachable-stage CLOSURE.

    The closure (the #1783 guard's `_image_input_paths`) is the right question for "does
    a change here need a deploy trigger", because `production`'s `COPY --from=dependencies
    /app/.venv /app/.venv` means dependencies' inputs (requirements.lock, patches/) do
    feed what ends up in the image, indirectly, via the venv build. It is the WRONG
    question for "does THIS repo path land in THIS stage's filesystem": that same
    `COPY --from=dependencies` only pulls the one container path `/app/.venv`, not
    dependencies' whole tree — so a file COPYed only into `dependencies` never reaches
    `/app/artifacts/...` in production, even though the closure check would score it as
    covered. Mirrors `_image_input_paths`'s own normalisation (a `./` prefix stripped,
    `COPY --from=` sources skipped) but scoped to one stage's own `_Stage.copies`.
    """
    stages = _parse_stages(dockerfile_text)
    assert stage_name in stages, f"no `AS {stage_name}` stage found; parsed={sorted(stages)}"
    sources: set[str] = set()
    for cp in stages[stage_name].copies:
        if cp.from_stage is not None:
            continue
        for src in cp.sources:
            sources.add(src[2:] if src.startswith("./") else src)
    return sources


def test_dockerfile_copies_the_artifact_into_every_app_stage() -> None:
    """Both `development` and `production` are separate FROMs; each needs the artifact in
    its OWN COPY sources — not merely >=2 COPY lines anywhere in the file (a
    planted-failure review proved that proxy passes even when both lines sit in the same
    stage), and not the reachable-stage CLOSURE either (a second planted-failure review
    proved THAT proxy passes when both lines sit only in the upstream `dependencies`
    stage, which production's `COPY --from=dependencies /app/.venv` would never actually
    pull in). See the PR description for both planted-failure outputs."""
    text = _DOCKERFILE.read_text()
    for stage in ("production", "development"):
        own = _own_copy_sources(text, stage)
        assert _ARTIFACT_REL in own, (
            f"{_ARTIFACT_REL} is not COPYed directly in the {stage!r} stage itself "
            f"(found: {sorted(own)}); a COPY in an upstream stage such as `dependencies` "
            "does not count — production only pulls the specific container path "
            "/app/.venv from dependencies via `COPY --from=dependencies`, not its whole "
            "filesystem, so the artifact would never reach /app/artifacts/… that way."
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
