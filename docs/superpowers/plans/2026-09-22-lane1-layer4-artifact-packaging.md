# Lane 1: Ship the Layer‑4 classifier artifact — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `artifacts/dspy/causal_role_classifier.json` reaches the Docker image so Layer 4 (DSPy causal-role classifier) fires in production retrains instead of silently skipping.

**Architecture:** No runtime code changes. Two build files (`docker/Dockerfile`, `.dockerignore`) plus one static guard test that mirrors `tests/unit/test_data/test_kg/test_kg_cache_packaging.py`. The loader (`src/data/causal_role_classifier_loader.py`) already resolves `PROJECT_ROOT / "artifacts" / "dspy" / "causal_role_classifier.json"`, which is `/app/artifacts/dspy/...` in the image.

**Tech Stack:** Docker multi-stage build, pytest, the repo's static `.dockerignore` matcher.

**Spec:** `docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md` (Lane 1).

**Before control (recorded 2026-09-22, keep for the cert):** inside `e2i_api`, `ls /app/artifacts` → "No such file or directory"; `_try_load_layer_4_classifier()` → `None`.

---

## Worktree

```bash
git -C /home/enunez/Projects/e2i_causal_analytics worktree add \
  /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane1-layer4-artifact \
  -b claude/lane1-layer4-artifact claude/public-apis-live-path-spec
cd /home/enunez/Projects/e2i_causal_analytics/.worktrees/lane1-layer4-artifact
git branch --show-current   # claude/lane1-layer4-artifact
```

Every `pytest` below runs with `-n 0` from this directory, and every python invocation asserts `src.__file__` starts with this worktree path (the editable `.pth` otherwise imports `src` from the main checkout).

---

### Task 1: Guard test (red first)

**Files:**
- Create: `tests/unit/test_data/test_causal_role_classifier_packaging.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -c "import src; assert src.__file__.startswith('$PWD')" && pytest -n 0 tests/unit/test_data/test_causal_role_classifier_packaging.py -v`
Expected: only `test_dockerfile_copies_the_artifact_into_every_app_stage` FAILS (found 0 COPY lines). The other five PASS — including `test_the_classifier_artifact_survives_dockerignore`, which passes immediately: `.dockerignore` was never the cause (measured 2026-09-22).

- [ ] **Step 3: Commit the red test**

```bash
git add tests/unit/test_data/test_causal_role_classifier_packaging.py
git commit -m "test(layer4): guard that the classifier artifact reaches the image (red)

Measured on the production image 2026-09-22: .dockerignore never excluded it
(slash-less `*.json` matches the root only); the only defect is the missing
COPY. The matcher deliberately models Docker's root-only semantics.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Re-include the artifact in the build context

NOT NEEDED — premise disproved 2026-09-22. Measured on the production image: a
slash-less pattern like `*.json` matches the build-context root only (confirmed
via `scripts/benchmarks/routing/data/agent_contracts.json`, present in that
same image under that same rule), so `.dockerignore` never excluded the
artifact. `.dockerignore` is left untouched; the only defect was the missing
`COPY`, fixed in Task 3.

---

### Task 3: COPY the artifact in both Dockerfile stages

**Files:**
- Modify: `docker/Dockerfile` — after `COPY data/kg_cache/ ./data/kg_cache/` in the `development` stage (line ~109) and after the same line in the `production` stage (line ~203)

- [ ] **Step 1: Add the COPY in the development stage**

Directly after the development-stage `COPY data/kg_cache/ ./data/kg_cache/`:

```dockerfile
# Compiled Layer-4 causal-role classifier (DSPy). The loader resolves
# PROJECT_ROOT/artifacts/dspy/causal_role_classifier.json, i.e. /app/artifacts/…
# here. Absent, Layer 4 skips silently (#1607 shape). Guarded by
# tests/unit/test_data/test_causal_role_classifier_packaging.py.
COPY artifacts/dspy/causal_role_classifier.json ./artifacts/dspy/
```

- [ ] **Step 2: Add the identical COPY in the production stage**

Directly after the production-stage `COPY data/kg_cache/ ./data/kg_cache/`:

```dockerfile
# Compiled Layer-4 causal-role classifier (DSPy) — see the development stage.
COPY artifacts/dspy/causal_role_classifier.json ./artifacts/dspy/
```

- [ ] **Step 3: Run the guard and the sibling guard**

Run: `pytest -n 0 tests/unit/test_data/test_causal_role_classifier_packaging.py tests/unit/test_data/test_kg/test_kg_cache_packaging.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add docker/Dockerfile
git commit -m "build(layer4): COPY the classifier artifact into both app stages

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Gates, PR, review, merge

- [ ] **Step 1: CI's lint commands, whole tree, no cache**

```bash
ruff check --no-cache src/ tests/ && ruff format --check --no-cache src/ tests/
```
Expected: `All checks passed!` twice. Do NOT run mypy on the droplet (CI is the arbiter).

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin claude/lane1-layer4-artifact
gh pr create --base main --title "build(layer4): ship the causal-role classifier artifact in the image" --body-file - <<'EOF'
Part of the 2026-09-22 public-APIs-live-path design (docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md), lane 1 of 3.

**Problem.** `artifacts/dspy/causal_role_classifier.json` is committed and the loader resolves it, but the Dockerfile never COPYed it into either app stage. Measured 2026-09-22 inside `e2i_api`: `/app/artifacts` does not exist; `_try_load_layer_4_classifier()` returns `None`; Layer 4 silently skips in every production retrain. The artifact was absent only because no COPY existed — `.dockerignore` never excluded it (a slash-less pattern like `*.json` matches the build-context root only; confirmed via `scripts/benchmarks/routing/data/agent_contracts.json`, present in the same image under the same rule). Same #1607/#600 *symptom* as the KG cache, different cause.

**Change.** COPY the file in both app stages; a static guard test (red on main) mirrors the KG-cache packaging guard and pins the matcher's Docker root-only semantics for slash-less patterns so it isn't mistaken for a gitignore-style any-depth match. `.dockerignore` is untouched.

**Behaviour once deployed.** Layer 4 fires for `ambiguous` (3σ<z≤5σ) features during `execute_model_retraining`, using the loader's default `anthropic/claude-sonnet-4-6` (key present in the container). Citation resolution stays bounded by `ADAPTIVE_CITATION_RESOLUTION_BUDGET` (default 25). Model-provider choice is out of scope and flagged in the spec.

**Live cert plan.** After deploy, inside `e2i_api`: `load_compiled_classifier(strict=True)` returns a classifier; `_try_load_layer_4_classifier()` is not None. Before control recorded 2026-09-22.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

- [ ] **Step 3: Watch CI via the actions API with the full SHA** (the PAT 403s on `gh pr checks`)

```bash
SHA=$(git rev-parse HEAD)
gh api "repos/enunezvn/e2i_causal_analytics/actions/runs?head_sha=$SHA" --jq '.workflow_runs[] | [.name,.status,.conclusion] | @tsv'
```
Expected: every run `completed success`. The mypy gate is a ceiling: download the `mypy-report` artifact and confirm the count is unchanged from main (this lane touches no Python under `src/`).

- [ ] **Step 4: Codex review brief** — must include verbatim:

> If a recommendation solves a labeling problem instead of a functional problem, flag it as HIGH finding. If a recommendation preserves code without investigating intent (PR history, linked issues, user-requested functionality), flag it as HIGH finding. If a recommendation deletes code without verifying intent, flag it as HIGH finding. Audit the question being asked, not just the answer given.

Point the reviewer at a frozen ref (the pushed SHA), not the worktree. Address findings; re-run the gates.

- [ ] **Step 5: Merge preserving history** — `gh pr merge --merge` (never squash). First confirm `closingIssuesReferences` is empty via graphql (no issue should auto-close).

---

### Task 5: Deploy and live cert

- [ ] **Step 1: Wait for the deploy** (merge to main triggers it; images are built in CI and pulled by the rollout). Confirm with the actions API that the deploy run completed, then:

```bash
docker inspect -f '{{.State.StartedAt}} {{.State.Health.Status}}' e2i_api
```
Expected: `StartedAt` later than the merge time, `healthy`.

- [ ] **Step 2: Cert inside the container**

```bash
mkdir -p docs/demos/results/2026-09-22_public_apis_live_path
docker exec -i -e PYTHONPATH=/app e2i_api python - <<'EOF' | tee docs/demos/results/2026-09-22_public_apis_live_path/lane1_layer4_artifact_cert.txt
import os
print("artifact exists:", os.path.exists("/app/artifacts/dspy/causal_role_classifier.json"))
from src.data.causal_role_classifier_loader import load_compiled_classifier
clf = load_compiled_classifier(strict=True)
print("strict load:", type(clf).__name__)
from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import _try_load_layer_4_classifier
print("layer4 loader:", type(_try_load_layer_4_classifier()).__name__)
EOF
docker inspect -f 'container StartedAt {{.State.StartedAt}}' e2i_api | tee -a docs/demos/results/2026-09-22_public_apis_live_path/lane1_layer4_artifact_cert.txt
```
Expected: `artifact exists: True`, `strict load: <a classifier class name>`, `layer4 loader: <same class>` (not `NoneType`). Before control: `False` / `FileNotFoundError` / `NoneType` (2026-09-22).

- [ ] **Step 3: Record the cert in-repo** — commit the cert file on a follow-up docs branch (a docs-only merge fires no deploy). Note in the cert that the cited before-control was measured on container started `2026-09-22T01:09:44Z`.
