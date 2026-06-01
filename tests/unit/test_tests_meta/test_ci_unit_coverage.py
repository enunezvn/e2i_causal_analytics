"""CI-coverage guard for the unit-test suite (issue #555).

Every ``tests/unit/<dir>/`` directory that contains test files must be either:

* **RUN** by *any* ``backend-tests.yml`` job — i.e. it appears as a whole-
  directory positional argument (``tests/unit/<dir>/``) in the unit-tests step
  OR in the serviceless ``heavy-unit-tests`` job (or any future test job); **or**
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
    # Empty as of #583: every tests/unit/* directory now runs in a CI job.
    # History: #555 wired ~24 dirs into the unit + heavy-unit-tests lanes and
    # deferred the last two here. #583 PR1 moved test_data_preparer into the
    # serviceless heavy-unit-tests lane (its only blocker was slow
    # sklearn-permutation tests busting the 30s thread timeout — fixed with
    # per-test @pytest.mark.timeout overrides; no service deps). #583 PR2 moved
    # test_agents into a new service-provisioned `agents-tests` lane (it makes
    # un-mocked mlflow calls that retry-backoff-hang against a dead tracking URI
    # and aborts the session under timeout_method=thread; a real MLflow server +
    # Redis in that lane resolves it).
    # To exclude a NEW dir, add it here with a non-empty reason — or wire it
    # into a ci-success.needs job's whole-dir pytest args (the guard enforces
    # one or the other).
}

# ---------------------------------------------------------------------------
# Registry of ROOT-LEVEL ``tests/unit/test_*.py`` files intentionally NOT run.
# file name -> reason (must be non-empty).
#
# Root-level files are invisible to the whole-dir allowlist matching (the unit
# job lists ``tests/unit/<dir>/`` positionals, never bare ``tests/unit/``), so
# they get their own run-or-documented guard. A root file that is neither run as
# a single-file positional in a required job nor listed here fails the guard —
# the same #555 silent-rot class, one directory level up.
# ---------------------------------------------------------------------------
INTENTIONALLY_EXCLUDED_FILES: dict[str, str] = {
    "test_cognitive_simple.py": (
        "Manual cognitive-cycle smoke SCRIPT, not a pytest module: it defines "
        "zero test functions and is run via `python tests/unit/test_cognitive_"
        "simple.py` (its `__main__` calls asyncio.run(main())). It needs a live "
        "Anthropic API key + Supabase + FalkorDB. pytest collects 0 items from "
        "it (EXIT=5), so wiring it into a lane would run nothing. Kept as a "
        "developer smoke; wire it only if it gains real serviceless test fns."
    ),
}


def _load_workflow() -> dict:
    """Parse the backend-tests workflow YAML."""
    return yaml.safe_load(WORKFLOW.read_text())


# The aggregate gate every PR-blocking job feeds into. A test dir only counts as
# "covered" if it runs in a job REQUIRED by this gate — a dir run by an unwired or
# skipped job would not block a merge, so counting it would reopen the #555
# false-green (codex review).
CI_SUCCESS_JOB = "ci-success"


def _required_jobs(workflow: dict) -> set[str]:
    """Job names required by the ``ci-success`` aggregate gate."""
    needs = workflow.get("jobs", {}).get(CI_SUCCESS_JOB, {}).get("needs", [])
    if isinstance(needs, str):
        needs = [needs]
    return set(needs)


def _test_run_blocks(workflow: dict) -> list[str]:
    """Shell bodies of test steps that run ``tests/unit/`` dirs, restricted to
    jobs REQUIRED by ``ci-success``.

    A dir is "covered" if it runs in any required job — the coverage unit job,
    the serviceless ``heavy-unit-tests`` job (test_causal_engine /
    test_digital_twin), or any future test job wired into ``ci-success.needs``.
    The guard unions the whole-dir args across those jobs; a job NOT in the
    required set is ignored, because its failures cannot block a merge.
    """
    required = _required_jobs(workflow)
    blocks: list[str] = []
    for job_name, job in workflow.get("jobs", {}).items():
        if job_name not in required:
            continue
        for step in job.get("steps", []):
            run = step.get("run", "") or ""
            if "tests/unit/" in run:
                blocks.append(run)
    if not blocks:
        raise AssertionError(
            "Could not locate any pytest step running tests/unit/ in a job "
            f"required by '{CI_SUCCESS_JOB}' in backend-tests.yml — the "
            "CI-coverage guard cannot parse the allowlist."
        )
    return blocks


_WHOLE_DIR_RE = re.compile(r"^tests/unit/([A-Za-z0-9_]+)/\s*\\?$")


def _run_whole_dirs(run_blocks: list[str]) -> set[str]:
    """Dirs run as whole-dir positional args across all given run blocks
    (skipping ``--ignore`` carve-outs and single-file args)."""
    whole: set[str] = set()
    for run_block in run_blocks:
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
    run_whole = _run_whole_dirs(_test_run_blocks(_load_workflow()))
    documented = set(INTENTIONALLY_EXCLUDED)
    uncovered = sorted(existing - run_whole - documented)
    assert not uncovered, (
        "tests/unit dirs neither run by the CI unit job nor documented as "
        f"intentionally excluded: {uncovered}. Add each whole dir to the "
        "backend-tests.yml unit allowlist, or add it to INTENTIONALLY_EXCLUDED "
        "in this file with a reason."
    )


def test_no_dir_is_both_run_and_excluded() -> None:
    run_whole = _run_whole_dirs(_test_run_blocks(_load_workflow()))
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


def test_dir_run_in_any_job_counts_as_covered() -> None:
    """A dir run by ANY backend-tests job counts as covered — not only the
    coverage unit job. Heavy dirs (test_causal_engine, test_digital_twin) run in
    a separate serviceless ``heavy-unit-tests`` job, so the guard must union the
    whole-dir args across every job that runs ``tests/unit/`` (issue #555)."""
    workflow = {
        "jobs": {
            "unit-tests": {"steps": [{"run": "pytest \\\n  tests/unit/foo/ \\\n  --cov=src"}]},
            "heavy-unit-tests": {"steps": [{"run": "pytest \\\n  tests/unit/bar/ \\\n  -n 2"}]},
            # a job that touches no tests/unit dir must contribute nothing
            "lint": {"steps": [{"run": "ruff check src/ tests/"}]},
            "ci-success": {"needs": ["unit-tests", "heavy-unit-tests", "lint"]},
        }
    }
    covered = _run_whole_dirs(_test_run_blocks(workflow))
    assert covered == {"foo", "bar"}


def test_dir_run_only_in_non_required_job_is_not_covered() -> None:
    """A dir run only by a job NOT in ``ci-success.needs`` does not count.

    Such a job does not gate a merge, so a failing test there is invisible to the
    required check — counting it would reopen the exact #555 false-green this
    guard prevents (codex review)."""
    workflow = {
        "jobs": {
            "unit-tests": {"steps": [{"run": "pytest \\\n  tests/unit/foo/ \\\n  --cov=src"}]},
            "stray-tests": {"steps": [{"run": "pytest \\\n  tests/unit/baz/ \\\n  -n 2"}]},
            "ci-success": {"needs": ["unit-tests"]},
        }
    }
    covered = _run_whole_dirs(_test_run_blocks(workflow))
    assert covered == {"foo"}
    assert "baz" not in covered


# ---------------------------------------------------------------------------
# Root-level tests/unit/*.py coverage guard (issue #583 follow-up).
#
# The dir guard above only sees subdirectories. Ten root-level
# ``tests/unit/test_*.py`` files (incl. ``test_ml_foundation_schemas.py``, the
# 95-test DataPreparerState contract that backend-tests.yml names only in a
# COMMENT) were collected by no lane. This guard closes that one-level-up hole.
# ---------------------------------------------------------------------------

# Single-file positional like ``tests/unit/test_ml_foundation_schemas.py`` or
# ``tests/unit/test_ml_foundation_schemas.py \`` (trailing line continuation).
_SINGLE_FILE_RE = re.compile(r"^tests/unit/(test_[A-Za-z0-9_]+\.py)\s*\\?$")

# A bare ``tests/unit/`` whole-tree positional would also cover every root file.
_WHOLE_UNIT_TREE_RE = re.compile(r"^tests/unit/\s*\\?$")


def _run_single_files(run_blocks: list[str]) -> set[str]:
    """Root-level ``tests/unit/*.py`` files run as single-file positional args
    across the given run blocks (skipping ``--ignore`` carve-outs).

    If a block runs the whole ``tests/unit/`` tree as a bare positional, every
    existing root file counts as covered.
    """
    files: set[str] = set()
    for run_block in run_blocks:
        for line in run_block.splitlines():
            s = line.strip()
            if s.startswith("--ignore"):
                continue
            if _WHOLE_UNIT_TREE_RE.match(s):
                return _existing_root_test_files()
            m = _SINGLE_FILE_RE.match(s)
            if m:
                files.add(m.group(1))
    return files


def _existing_root_test_files() -> set[str]:
    """Immediate ``tests/unit/test_*.py`` files (depth 1). These are invisible
    to the whole-dir allowlist matching and so need their own guard."""
    return {p.name for p in UNIT_DIR.glob("test_*.py") if p.is_file()}


def test_every_root_unit_file_is_run_or_documented() -> None:
    existing = _existing_root_test_files()
    run_files = _run_single_files(_test_run_blocks(_load_workflow()))
    documented = set(INTENTIONALLY_EXCLUDED_FILES)
    uncovered = sorted(existing - run_files - documented)
    assert not uncovered, (
        "root-level tests/unit/*.py files neither run by a required CI job nor "
        f"documented as intentionally excluded: {uncovered}. Add each file as a "
        "single-file positional to the backend-tests.yml unit allowlist, or add "
        "it to INTENTIONALLY_EXCLUDED_FILES in this file with a reason."
    )


def test_no_root_file_is_both_run_and_excluded() -> None:
    run_files = _run_single_files(_test_run_blocks(_load_workflow()))
    overlap = sorted(run_files & set(INTENTIONALLY_EXCLUDED_FILES))
    assert not overlap, (
        f"root files are both run by CI and listed as excluded: {overlap}. "
        "Remove them from INTENTIONALLY_EXCLUDED_FILES."
    )


def test_no_stale_file_exclusions() -> None:
    existing = _existing_root_test_files()
    stale = sorted(set(INTENTIONALLY_EXCLUDED_FILES) - existing)
    assert not stale, (
        f"INTENTIONALLY_EXCLUDED_FILES names files that no longer exist: {stale}. "
        "Remove the stale entries."
    )


def test_all_file_exclusions_have_reasons() -> None:
    blank = sorted(
        f for f, reason in INTENTIONALLY_EXCLUDED_FILES.items() if not reason.strip()
    )
    assert not blank, f"INTENTIONALLY_EXCLUDED_FILES entries missing a reason: {blank}"


def test_single_file_positional_counts_as_covered() -> None:
    """A root file run as a single-file positional in a required job counts."""
    workflow = {
        "jobs": {
            "unit-tests": {
                "steps": [{"run": "pytest \\\n  tests/unit/test_foo.py \\\n  --cov=src"}]
            },
            "ci-success": {"needs": ["unit-tests"]},
        }
    }
    covered = _run_single_files(_test_run_blocks(workflow))
    assert covered == {"test_foo.py"}


def test_bare_unit_tree_positional_covers_all_root_files() -> None:
    """A bare ``tests/unit/`` positional covers every existing root file."""
    workflow = {
        "jobs": {
            "unit-tests": {"steps": [{"run": "pytest \\\n  tests/unit/ \\\n  --cov=src"}]},
            "ci-success": {"needs": ["unit-tests"]},
        }
    }
    covered = _run_single_files(_test_run_blocks(workflow))
    assert covered == _existing_root_test_files()


def test_root_file_only_in_non_required_job_is_not_covered() -> None:
    """A root file run only by a job NOT in ``ci-success.needs`` does not count
    — same merge-invisibility reasoning as the dir guard."""
    workflow = {
        "jobs": {
            "unit-tests": {
                "steps": [{"run": "pytest \\\n  tests/unit/test_foo.py \\\n  --cov=src"}]
            },
            "stray": {
                "steps": [{"run": "pytest \\\n  tests/unit/test_bar.py \\\n  -n 2"}]
            },
            "ci-success": {"needs": ["unit-tests"]},
        }
    }
    covered = _run_single_files(_test_run_blocks(workflow))
    assert covered == {"test_foo.py"}
    assert "test_bar.py" not in covered
