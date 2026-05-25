"""
#499 regression guard: every workflow YAML in .github/workflows/ must parse
without error. The tier1b_b2_experiment.yml had a column-0 python -c block
that YAML mistook for mapping keys, causing startup_failure on every push.
"""

import pathlib

import pytest
import yaml

WORKFLOWS_DIR = (
    pathlib.Path(__file__).parent.parent.parent / ".github" / "workflows"
)


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
        pytest.fail(
            f"{workflow_path.name} failed YAML parse:\n{exc}"
        )
