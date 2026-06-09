"""Guardrail: no src/ module may import the relocated golden-seed test fixture.

The test fixture ``tests/.../_recipient_seed_fixtures.py`` was relocated out of
``src/`` so that production never optimizes on synthetic hand-authored seeds.
This test walks the ``src/`` tree and asserts that no production module imports
``recipient_seeds`` (the old path) or ``_recipient_seed_fixtures`` (the new
test-only path), keeping the no-synthetic-in-prod invariant locked in.

The B0 scaffold tests already cover:
  - ``src.agents.feedback_learner.recipient_seeds`` is not importable (deleted).
  - ``recipient_optimizer`` source contains no ``recipient_seeds`` import.

This test adds the complementary check:
  - no ``src/`` file imports the relocated test-only fixture module by its new name.
"""

from __future__ import annotations

import os

_SRC_ROOT = os.path.join(
    os.path.dirname(__file__),  # tests/unit/test_agents/test_feedback_learner/
    "..",
    "..",
    "..",
    "..",
    "src",
)


def _collect_src_py_files() -> list[str]:
    paths: list[str] = []
    for dirpath, _, filenames in os.walk(os.path.abspath(_SRC_ROOT)):
        for fname in filenames:
            if fname.endswith(".py"):
                paths.append(os.path.join(dirpath, fname))
    return paths


def test_no_src_module_imports_recipient_seeds_or_test_fixture() -> None:
    """Walk src/ and assert no file IMPORTS the golden-seed modules.

    Forbidden import patterns (both the deleted module name and the relocated fixture):
      - ``import recipient_seeds`` / ``from ... import recipient_seeds``
      - ``import _recipient_seed_fixtures`` / ``from ... import _recipient_seed_fixtures``

    Doc-string references (e.g. "seeds live only as a test fixture (``tests/.../_recipient_seed_fixtures.py``).")
    in comments are NOT violations — those are honesty notes, not imports.
    """
    import re

    # Match actual import statements, not docstring / comment references.
    _IMPORT_RE = re.compile(
        r"^\s*(?:import|from)\s+[^\n#]*\b(recipient_seeds|_recipient_seed_fixtures)\b",
        re.MULTILINE,
    )

    violations: list[str] = []

    for fpath in _collect_src_py_files():
        try:
            with open(fpath, encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue

        if _IMPORT_RE.search(content):
            violations.append(fpath)

    assert violations == [], (
        "The following src/ files import the golden-seed modules "
        "(recipient_seeds / _recipient_seed_fixtures). "
        "Production must run on real emitted signals or skip — never on synthetic seeds.\n"
        + "\n".join(f"  {v}" for v in violations)
    )
