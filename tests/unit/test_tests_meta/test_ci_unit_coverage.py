"""CI-coverage guard for the unit-test suite (issue #555).

Every ``tests/unit/<dir>/`` directory that contains test files must be either:

* **RUN** by the ``backend-tests.yml`` unit job — i.e. it appears as a
  whole-directory positional argument (``tests/unit/<dir>/``) in the
  ``Run unit tests with coverage`` step; **or**
* **DOCUMENTED** in :data:`INTENTIONALLY_EXCLUDED` with a non-empty reason.

Background
----------
The unit-test CI job runs an *explicit allowlist* of ``tests/unit/*`` dirs
(plus a few ``--ignore`` carve-outs). For a long time ~23 directories — ~450
test files, including the entire ``tests/unit/test_agents/`` tree — were in the
allowlist for **no** job and therefore never executed in CI. The ``Unit Tests``
check went green while a large fraction of the unit suite never ran: a silent
false-green (issue #555).

This guard makes that class of regression impossible to reintroduce silently.
Create a new ``tests/unit`` directory? You must *either* wire it into the CI
allowlist *or* record — right here, with a reason — why it is deliberately not
run. A directory that is neither fails this test.

The guard parses the workflow YAML rather than hard-coding the allowlist, so it
stays correct as the allowlist evolves.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

# tests/unit/test_tests_meta/test_ci_unit_coverage.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "backend-tests.yml"
UNIT_DIR = REPO_ROOT / "tests" / "unit"

# ---------------------------------------------------------------------------
# Registry of directories intentionally NOT run by the unit CI job.
# dir name -> reason (must be non-empty). Keep reasons specific so a future
# reader can tell whether the exclusion is still warranted.
# ---------------------------------------------------------------------------
INTENTIONALLY_EXCLUDED: dict[str, str] = {
    # Populated in the green step from a CI-faithful (feast-blocked,
    # serviceless) audit of each excluded directory.
}


def _unit_job_run_block() -> str:
    """Return the shell body of the unit-test job's pytest step."""
    data = yaml.safe_load(WORKFLOW.read_text())
    for job in data.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run", "") or ""
            if "tests/unit/" in run and "--cov" in run:
                return run
    raise AssertionError(
        "Could not locate the unit-test pytest step (tests/unit/ + --cov) "
        "in backend-tests.yml — the CI-coverage guard cannot parse the allowlist."
    )


_WHOLE_DIR_RE = re.compile(r"^tests/unit/([A-Za-z0-9_]+)/\s*\\?$")


def _run_whole_dirs(run_block: str) -> set[str]:
    """Directories run as whole-dir positional args (not --ignore, not a file)."""
    whole: set[str] = set()
    for line in run_block.splitlines():
        s = line.strip()
        if s.startswith("--ignore"):
            continue
        m = _WHOLE_DIR_RE.match(s)
        if m:
            whole.add(m.group(1))
    return whole


def _existing_test_dirs() -> set[str]:
    """Immediate ``tests/unit`` subdirectories that contain at least one test file."""
    dirs: set[str] = set()
    for p in UNIT_DIR.iterdir():
        if not p.is_dir() or p.name == "__pycache__":
            continue
        if any(p.rglob("test_*.py")):
            dirs.add(p.name)
    return dirs


def test_every_unit_dir_is_run_or_documented() -> None:
    existing = _existing_test_dirs()
    run_whole = _run_whole_dirs(_unit_job_run_block())
    documented = set(INTENTIONALLY_EXCLUDED)
    uncovered = sorted(existing - run_whole - documented)
    assert not uncovered, (
        "tests/unit dirs neither run by the CI unit job nor documented as "
        f"intentionally excluded: {uncovered}. Add each whole dir to the "
        "backend-tests.yml unit allowlist, or add it to INTENTIONALLY_EXCLUDED "
        "in this file with a reason."
    )


def test_no_dir_is_both_run_and_excluded() -> None:
    run_whole = _run_whole_dirs(_unit_job_run_block())
    overlap = sorted(run_whole & set(INTENTIONALLY_EXCLUDED))
    assert not overlap, (
        f"dirs are both run by CI and listed as excluded: {overlap}. "
        "Remove them from INTENTIONALLY_EXCLUDED."
    )


def test_no_stale_exclusions() -> None:
    existing = _existing_test_dirs()
    stale = sorted(set(INTENTIONALLY_EXCLUDED) - existing)
    assert not stale, (
        f"INTENTIONALLY_EXCLUDED names dirs that no longer exist: {stale}. "
        "Remove the stale entries."
    )


def test_all_exclusions_have_reasons() -> None:
    blank = sorted(d for d, reason in INTENTIONALLY_EXCLUDED.items() if not reason.strip())
    assert not blank, f"INTENTIONALLY_EXCLUDED entries missing a reason: {blank}"
