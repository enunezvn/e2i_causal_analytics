"""Forcing-function test: docker/bentoml/requirements-bentoml.txt must pin
bentoml exactly to match the root requirements.txt.

Issue #321 MED bullet — surfaced by C2 / PR #315: the Docker requirements
file used a range pin ``bentoml>=1.4.0,<2.0.0`` while the root requirements
pinned ``bentoml==1.4.39``. Range pins on Docker rebuilds risk deploying a
silently-different bentoml version than what production was tested against.

This test reads both files and asserts they agree on the exact pin.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQS = REPO_ROOT / "requirements.txt"
DOCKER_BENTOML_REQS = REPO_ROOT / "docker" / "bentoml" / "requirements-bentoml.txt"

_BENTOML_LINE = re.compile(r"^\s*bentoml\s*([<>=!~]+.*)\s*$", re.MULTILINE)


def _read_bentoml_constraint(path: Path) -> str:
    text = path.read_text()
    match = _BENTOML_LINE.search(text)
    assert match, f"no bentoml requirement line found in {path}"
    return match.group(1).strip()


def test_docker_bentoml_requirements_use_exact_pin():
    """The Docker bentoml requirements file must use an exact ``==`` pin."""
    constraint = _read_bentoml_constraint(DOCKER_BENTOML_REQS)
    assert constraint.startswith("=="), (
        f"docker/bentoml/requirements-bentoml.txt must pin bentoml with == "
        f"(got {constraint!r}); range pins create silent version skew on rebuild."
    )


def test_docker_bentoml_pin_matches_root_requirements():
    """The Docker bentoml pin must equal the root requirements.txt pin."""
    docker_constraint = _read_bentoml_constraint(DOCKER_BENTOML_REQS)
    root_constraint = _read_bentoml_constraint(ROOT_REQS)
    assert docker_constraint == root_constraint, (
        f"bentoml pin drift: docker={docker_constraint!r} vs root={root_constraint!r}"
    )
