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
"""

import pathlib

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
