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


def _extract_pytest_invocation(run_block: str) -> str:
    """Reconstruct the ``pytest ...`` command line(s) from a multi-line
    shell ``run:`` block, stripping shell comments (``# ...``) and joining
    backslash-continuations into a single logical line. This is more
    discerning than a raw substring match — codex-review feedback: a
    comment like ``# keep -m "not slow"`` would otherwise spoof the
    earlier ``re.search(run_block)`` form.
    """
    # Drop shell comments (anything from a `#` not inside quotes — we use
    # the simpler heuristic of stripping comments at start-of-stripped-line
    # only, which is the only form pytest invocations use in our workflows).
    cleaned_lines = []
    for raw in run_block.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        # Inline trailing comment after the command: pytest never uses #
        # mid-line in our workflows, but be defensive.
        if " #" in stripped and "pytest" not in stripped.split(" #", 1)[1]:
            stripped = stripped.split(" #", 1)[0].rstrip()
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines)
    # Join backslash-continuations.
    cleaned = re.sub(r"\\\s*\n\s*", " ", cleaned)
    # Find the first `pytest ` invocation and return everything up to the
    # next standalone newline (i.e. the full continued command).
    m = re.search(r"(^|\n)\s*(pytest\s+.*?)(\n\S|\n\s*$|\Z)", cleaned, re.DOTALL)
    assert m, f"could not locate a pytest invocation in run block:\n{run_block[:300]}"
    return m.group(2).strip()


_MARK_RE_NOT_SLOW = re.compile(r"-m\s+(?:\"not slow\"|'not slow')")
# Match -m followed by `slow`, `"slow"`, or `'slow'` (only — not `not slow`).
_MARK_RE_SLOW = re.compile(r"-m\s+(?:slow|\"slow\"|'slow')(?:\s|$)")


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
    the PR-blocking path. Asserted against the parsed pytest invocation
    (comments stripped, line continuations collapsed) — a shell comment
    that happens to contain the magic string can't spoof this guard."""
    jobs = _load_yaml(_BACKEND_WF)["jobs"]
    run = _find_step_run(jobs, "integration-tests", "Run integration tests")
    cmd = _extract_pytest_invocation(run)
    assert _MARK_RE_NOT_SLOW.search(cmd), (
        "backend-tests.yml integration-tests step must pass "
        '-m "not slow" to deselect @pytest.mark.slow tests. '
        "Without this, slow tests run on the PR-blocking lane and "
        "the wall-clock fix from #481 silently regresses.\n"
        f"Parsed pytest cmd:\n  {cmd[:400]}"
    )


# ──────────────────────────────────────────────────────────────────
# Slow-tests workflow: selects slow
# ──────────────────────────────────────────────────────────────────


def test_slow_tests_workflow_runs_slow_marker() -> None:
    """``slow-tests.yml`` Job A MUST pass ``-m slow`` so the tests
    deselected from the PR lane still execute off-PR. Without this
    half of the contract, the #481 routing silently drops coverage.
    Accept ``-m slow``, ``-m "slow"``, or ``-m 'slow'``; reject
    ``-m "not slow"``."""
    jobs = _load_yaml(_SLOW_WF)["jobs"]
    run = _find_step_run(jobs, "slow-tests", "Run slow-marked tests")
    cmd = _extract_pytest_invocation(run)
    assert _MARK_RE_SLOW.search(cmd), (
        "slow-tests.yml slow-tests job MUST pass -m slow to actually "
        "run the slow-marked tests.\n"
        f"Parsed pytest cmd:\n  {cmd[:500]}"
    )
    assert not _MARK_RE_NOT_SLOW.search(cmd), (
        'slow-tests.yml slow-tests job must NOT pass -m "not slow" — '
        "that would deselect every test it's supposed to run."
    )


# ──────────────────────────────────────────────────────────────────
# Env parity: slow-tests.yml Job A must inherit the integration-tests
# real-data + asyncio-pollution opt-ins, or every slow real_data test
# (test_csu_negative_control_*, test_optum_held_out_*,
# test_g1_lineage_audit_sweep) hard-fails on the GH Actions runner
# (codex finding HIGH-1 in the #481 audit; observed in scheduled run
# 26355735293 and 4 preceding nights).
# ──────────────────────────────────────────────────────────────────


def test_slow_tests_workflow_has_real_data_optin() -> None:
    """``slow-tests.yml`` Job A MUST set ``ALLOW_MISSING_REAL_DATA=1``
    so real_data + slow tests skip cleanly when CSU/Optum cohort
    files are absent on the runner (instead of hard-failing the
    fixture per the codex pass-1 HIGH-1 default-hard-fail policy in
    tests/integration/test_csu_negative_control_20260510.py)."""
    jobs = _load_yaml(_SLOW_WF)["jobs"]
    env = jobs["slow-tests"].get("env", {})
    val = env.get("ALLOW_MISSING_REAL_DATA")
    assert val in {"1", 1, "true", "True"}, (
        "slow-tests.yml slow-tests job must set ALLOW_MISSING_REAL_DATA: '1' "
        "to match backend-tests.yml integration-tests env parity. "
        f"Got: {val!r}. Without this, every scheduled slow-tests run "
        "errors on the real_data + slow tests."
    )
