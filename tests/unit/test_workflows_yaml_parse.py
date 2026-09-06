"""
#499 regression guard: every workflow YAML in .github/workflows/ must parse
without error. The tier1b_b2_experiment.yml had a column-0 python -c block
that YAML mistook for mapping keys, causing startup_failure on every push.

#1544 regression guard: every job that runs jlumbroso/free-disk-space must
budget at least FREE_DISK_SPACE_MIN_TIMEOUT minutes. The step's runtime is
runner-load-dependent (measured 2m01s on a quiet runner vs 9m42s under load);
a 10-minute job budget left the MyPy job's mypy step cancelled on two
consecutive deploy attempts while every sibling free-disk-space job at 20-25
minutes absorbed the same tail. Jobs with no explicit timeout-minutes inherit
GitHub's 360-minute default and are exempt — headroom is not their problem.

#1901 item 4m sync guard: backend-tests.yml keeps its ``push.paths`` filter and
the ``changes`` job's ``PATTERN='^(...)'`` regex by hand (the file comment says
to keep them in sync; #1903 item 2 edited both). The two must name the same
set of paths - a glob missing from PATTERN lets a PR skip backend CI for a
file whose push would have run it, and vice versa.
"""

import pathlib
import re

import pytest
import yaml

WORKFLOWS_DIR = pathlib.Path(__file__).parent.parent.parent / ".github" / "workflows"

FREE_DISK_SPACE_MIN_TIMEOUT = 20


def collect_workflow_files():
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


@pytest.mark.parametrize("workflow_path", collect_workflow_files(), ids=lambda p: p.name)
def test_workflow_yaml_parses(workflow_path: pathlib.Path) -> None:
    """Each workflow file must be valid YAML (no ScannerError / parse failure)."""
    content = workflow_path.read_text()
    # If yaml.safe_load raises, the test fails with the YAML error as the message.
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as exc:
        pytest.fail(f"{workflow_path.name} failed YAML parse:\n{exc}")


@pytest.mark.parametrize("workflow_path", collect_workflow_files(), ids=lambda p: p.name)
def test_free_disk_space_jobs_have_timeout_headroom(workflow_path: pathlib.Path) -> None:
    """Jobs running free-disk-space need an explicit timeout of >= 20 minutes (#1544)."""
    try:
        workflow = yaml.safe_load(workflow_path.read_text())
    except yaml.YAMLError:
        return  # Already caught by test_workflow_yaml_parses; avoid duplicate failures.
    if not isinstance(workflow, dict):
        return
    underprovisioned = []
    for job_name, job in (workflow.get("jobs") or {}).items():
        steps = job.get("steps") or []
        uses_free_disk_space = any("free-disk-space" in (step.get("uses") or "") for step in steps)
        timeout = job.get("timeout-minutes")
        if uses_free_disk_space and timeout is not None and timeout < FREE_DISK_SPACE_MIN_TIMEOUT:
            underprovisioned.append(f"{job_name} (timeout-minutes={timeout})")
    assert not underprovisioned, (
        f"{workflow_path.name}: jobs run jlumbroso/free-disk-space with "
        f"timeout-minutes < {FREE_DISK_SPACE_MIN_TIMEOUT}: {', '.join(underprovisioned)}. "
        "The step's runtime is runner-load-dependent (observed up to ~10 minutes), "
        "so a tight job budget cancels the real work behind it (#1544). "
        f"Raise the job timeout to >= {FREE_DISK_SPACE_MIN_TIMEOUT}."
    )


# =============================================================================
# #1901 item 4m: backend-tests.yml push.paths <-> changes-job PATTERN sync guard
# =============================================================================

BACKEND_TESTS_WORKFLOW = WORKFLOWS_DIR / "backend-tests.yml"

# The literal as the changes job writes it: PATTERN='^(src/|tests/|...)'
_PATTERN_LITERAL_RE = re.compile(r"^\s*PATTERN='(?P<pattern>[^']+)'\s*$", re.MULTILINE)


def _backend_test_filters(workflow_path: pathlib.Path) -> tuple[list[str], str]:
    """(``push.paths`` globs, changes-job ``PATTERN`` literal) of a workflow file."""
    workflow = yaml.safe_load(workflow_path.read_text())
    # PyYAML (YAML 1.1) reads the bare ``on:`` key as boolean True.
    triggers = workflow.get("on", workflow.get(True))
    push_paths = list(triggers["push"]["paths"])
    steps = workflow["jobs"]["changes"]["steps"]
    literals = [
        m.group("pattern")
        for step in steps
        if step.get("run")
        for m in _PATTERN_LITERAL_RE.finditer(step["run"])
    ]
    assert len(literals) == 1, (
        f"{workflow_path.name}: expected exactly one PATTERN='...' literal in the "
        f"changes job, found {literals}"
    )
    return push_paths, literals[0]


def _representative_file(glob: str) -> str:
    """A concrete path the push glob matches (``src/**`` -> ``src/x.py``)."""
    return glob[: -len("/**")] + "/x.py" if glob.endswith("/**") else glob


def _pattern_alternatives_as_globs(pattern: str) -> list[str]:
    """Read ``^(a/|b\\.py$|...)`` back into the ``push.paths`` glob vocabulary."""
    assert pattern.startswith("^(") and pattern.endswith(")"), pattern
    globs = []
    for alternative in pattern[len("^(") : -len(")")].split("|"):
        literal = alternative.replace("\\.", ".")
        if literal.endswith("/"):
            globs.append(literal + "**")
        else:
            assert literal.endswith("$"), (
                f"exact-file alternative {alternative!r} must be $-anchored, or "
                f"'{literal}.bak' would also turn backend CI on"
            )
            globs.append(literal[:-1])
    return globs


def _assert_backend_filters_agree(push_paths: list[str], pattern: str) -> None:
    regex = re.compile(pattern)
    unmatched = [glob for glob in push_paths if not regex.search(_representative_file(glob))]
    assert not unmatched, f"push.paths globs the changes-job PATTERN does not match: {unmatched}"
    # Anchoring: a sibling directory or a suffixed file must not match.
    near_misses = [
        glob
        for glob in push_paths
        if regex.search(glob[: -len("/**")] + "x/x.py" if glob.endswith("/**") else glob + ".bak")
    ]
    assert not near_misses, f"PATTERN is not anchored for: {near_misses}"
    assert sorted(_pattern_alternatives_as_globs(pattern)) == sorted(push_paths), (
        "changes-job PATTERN alternatives and push.paths must name the same set"
    )


def test_backend_tests_push_paths_and_changes_pattern_agree() -> None:
    """The two hand-maintained backend-CI path filters name the same set (#1901 4m)."""
    push_paths, pattern = _backend_test_filters(BACKEND_TESTS_WORKFLOW)
    assert push_paths, "backend-tests.yml push.paths is empty"
    _assert_backend_filters_agree(push_paths, pattern)
