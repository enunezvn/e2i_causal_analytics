"""Conftest for skill unit tests.

Provides a fixture skills directory so tests don't depend on .claude/skills/
(which is gitignored).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

FIXTURES_SKILLS_DIR = Path(__file__).parent / "fixtures" / "skills"


@pytest.fixture(autouse=True)
def _use_fixture_skills():
    """Redirect SkillLoader default base path to the test fixtures directory."""
    with patch(
        "src.skills.loader.SkillLoader.DEFAULT_BASE_PATH",
        str(FIXTURES_SKILLS_DIR),
    ):
        # Also reset the module-level cached instances so they pick up
        # the patched path on next access.
        with patch("src.skills.loader._default_loader", None):
            with patch("src.skills.matcher._default_matcher", None):
                yield
