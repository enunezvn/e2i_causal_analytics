"""Guard the health_score fast-path design contract: NO eager DSPy import.

``src/agents/health_score/__init__.py`` documents "Zero LLM usage in critical
path" and a "<1s quick check"; the route describes a "Fast path design - no LLM
usage". Importing ``src.agents.health_score`` must therefore NOT pull in DSPy
(~714 MB), which it historically did via this eager-import chain::

    score_composer -> feedback_learner.recipient_emit
        -> feedback_learner.__init__ -> feedback_learner.agent
        -> feedback_learner.dspy_integration  (top-level ``import dspy``)

and, independently, ``health_score.dspy_integration`` had the same top-level
``import dspy``.

Under 4-way pytest-xdist on the 7 GB CI runner this 714 MB import OOM-killed the
worker, hanging the job until timeout. These tests assert the contract directly.

Each assertion runs in a FRESH subprocess interpreter so it is unaffected by any
other test in this process that may already have imported dspy.
"""

from __future__ import annotations

import subprocess
import sys


def _import_loads_dspy(module: str) -> bool:
    """Return whether importing ``module`` in a clean interpreter loads dspy.

    Runs in a subprocess so the result reflects a cold import, isolated from the
    parent process's already-loaded modules.
    """
    code = f"import sys\nimport {module}\nimport json\nprint(json.dumps('dspy' in sys.modules))\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"importing {module} failed in subprocess:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return proc.stdout.strip().endswith("true")


def test_importing_health_score_does_not_load_dspy() -> None:
    """The fast-path health_score package must not eagerly import dspy."""
    assert not _import_loads_dspy("src.agents.health_score"), (
        "import src.agents.health_score loaded dspy (~714 MB) into sys.modules; "
        "the fast-path design forbids any LLM/DSPy import on the critical path."
    )


def test_importing_health_score_dspy_integration_does_not_load_dspy() -> None:
    """Even the health_score dspy_integration module must defer the dspy import."""
    assert not _import_loads_dspy("src.agents.health_score.dspy_integration"), (
        "import src.agents.health_score.dspy_integration loaded dspy eagerly; "
        "dspy must be imported lazily, only when an optimizer path runs."
    )


def test_importing_feedback_learner_does_not_load_dspy() -> None:
    """feedback_learner is on the health_score import chain; it too must defer dspy."""
    assert not _import_loads_dspy("src.agents.feedback_learner"), (
        "import src.agents.feedback_learner loaded dspy eagerly; this is the chain "
        "that drags 714 MB onto the health_score fast path."
    )


def test_importing_feedback_learner_dspy_integration_does_not_load_dspy() -> None:
    """The feedback_learner dspy_integration module must defer the dspy import."""
    assert not _import_loads_dspy("src.agents.feedback_learner.dspy_integration"), (
        "import src.agents.feedback_learner.dspy_integration loaded dspy eagerly; "
        "dspy must be imported lazily, only when an optimizer path runs."
    )
