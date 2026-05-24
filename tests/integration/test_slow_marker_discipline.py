"""Discipline tests guarding the slow-marker contract between pytest
configuration and the CI workflows.

Issue #481 routes heavy integration tests off the PR-blocking
``integration-tests`` lane by:

1. Registering the ``slow`` pytest marker in ``pyproject.toml`` so
   ``@pytest.mark.slow`` doesn't trip ``--strict-markers``.
2. Deselecting slow-marked tests on the PR-required lane via
   ``-m "not slow"`` in ``.github/workflows/backend-tests.yml``.
3. Running the same set off-PR via ``-m slow`` in
   ``.github/workflows/slow-tests.yml`` (schedule + workflow_dispatch).

If any of those three contract points drift, the suite either
silently regresses wall-clock (slow tests creep back onto PRs) or
silently loses coverage (slow tests run on neither lane). These
runtime-assertion tests trip CI before either happens.

Note: these tests parse YAML strings rather than running the
workflow, because the workflow runs only on GitHub Actions and we
need local + CI parity. They use ``yaml.safe_load`` to validate
structure, then grep the relevant ``run:`` blocks to verify the
pytest invocation. This is the same shape as the existing
``tests/integration/test_validation_package.py`` drift-guard
pattern (see #464).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_BACKEND_WF = _REPO_ROOT / ".github" / "workflows" / "backend-tests.yml"
_SLOW_WF = _REPO_ROOT / ".github" / "workflows" / "slow-tests.yml"


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _find_step_run(jobs: dict, job_key: str, step_name_substr: str) -> str:
    """Return the ``run:`` block of the first step in ``jobs[job_key].steps``
    whose ``name`` contains ``step_name_substr``. Raise AssertionError
    (not lookup error) so failures show up as test failures."""
    assert job_key in jobs, f"workflow has no job named {job_key!r}; have: {list(jobs)}"
    steps = jobs[job_key].get("steps", [])
    for step in steps:
        if step_name_substr in step.get("name", ""):
            run = step.get("run", "")
            assert run, f"step {step_name_substr!r} in job {job_key!r} has empty 'run'"
            return run
    raise AssertionError(
        f"no step matching {step_name_substr!r} in job {job_key!r}; "
        f"steps were: {[s.get('name') for s in steps]}"
    )


# ──────────────────────────────────────────────────────────────────
# Marker registration
# ──────────────────────────────────────────────────────────────────


def test_slow_marker_registered_in_pyproject() -> None:
    """``slow`` marker must be registered under
    ``[tool.pytest.ini_options].markers`` so ``@pytest.mark.slow``
    doesn't trip ``--strict-markers`` or emit an unregistered-marker
    warning that becomes an error under
    ``filterwarnings = error``."""
    with _PYPROJECT.open("rb") as fh:
        cfg = tomllib.load(fh)
    markers = cfg.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
    slow_decls = [m for m in markers if m.startswith("slow:") or m == "slow"]
    assert slow_decls, (
        "Expected the 'slow' marker to be registered in "
        "[tool.pytest.ini_options].markers. Without registration, "
        "@pytest.mark.slow trips --strict-markers."
    )


# ──────────────────────────────────────────────────────────────────
# Backend-tests integration lane: deselects slow
# ──────────────────────────────────────────────────────────────────


def test_backend_tests_integration_lane_deselects_slow() -> None:
    """Issue #481: the PR-required ``integration-tests`` lane MUST
    pass ``-m "not slow"`` to pytest so slow-marked tests stay off
    the PR-blocking path."""
    jobs = _load_yaml(_BACKEND_WF)["jobs"]
    run = _find_step_run(jobs, "integration-tests", "Run integration tests")
    # Accept either single-quoted or double-quoted form; the YAML
    # multiline string may also contain backslash continuations.
    assert re.search(r"-m\s+['\"]not slow['\"]", run), (
        "backend-tests.yml integration-tests step must pass "
        '-m "not slow" to deselect @pytest.mark.slow tests. '
        "Without this, slow tests run on the PR-blocking lane and "
        "the wall-clock fix from #481 silently regresses."
    )


# ──────────────────────────────────────────────────────────────────
# Slow-tests workflow: selects slow
# ──────────────────────────────────────────────────────────────────


def test_slow_tests_workflow_runs_slow_marker() -> None:
    """``slow-tests.yml`` Job A MUST pass ``-m slow`` so the tests
    deselected from the PR lane still execute off-PR. Without this
    half of the contract, the #481 routing silently drops
    coverage."""
    jobs = _load_yaml(_SLOW_WF)["jobs"]
    run = _find_step_run(jobs, "slow-tests", "Run slow-marked tests")
    # ``-m slow`` (no negation, with optional surrounding spaces); reject
    # a bare ``-m "not slow"`` form that would leave coverage empty.
    assert re.search(r"-m\s+slow(\s|$|\\)", run), (
        "slow-tests.yml slow-tests job MUST pass -m slow to actually "
        "run the slow-marked tests. Found run block:\n" + run[:500]
    )
    assert not re.search(r"-m\s+['\"]not slow['\"]", run), (
        'slow-tests.yml slow-tests job must NOT pass -m "not slow" — '
        "that would deselect every test it's supposed to run."
    )
