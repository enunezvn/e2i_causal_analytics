"""Per-agent audit_workflow_id threading tests (sub-shard D1.2).

Sub-shard D1.2 threads ``audit_workflow_id`` from caller-supplied
``input_data`` into each agent's ``initial_state`` dict literal.
With ``default_factory=uuid4`` still present on each State (D1.4 removes it),
``None`` from ``.get()`` still triggers the default — so this PR is a
backward-compat-preserving precursor to D1.4.

These tests:
    1. STATIC — every agent.py contains ``"audit_workflow_id":
       input_data.get("audit_workflow_id"),`` in its initial_state dict
       literal (regression guard).
    2. RUNTIME — sample two agents whose ``run`` is cheap to mock
       (observability_connector and scope_definer) and verify the
       caller-supplied UUID flows into the State.
    3. RUNTIME — model_trainer's ``_build_fold_input`` preserves
       audit_workflow_id across fold recursion (the W3-lite repeated-fold
       site flagged in the D1 investigation).
"""

from __future__ import annotations

import re
from pathlib import Path

AGENT_FILES = [
    "src/agents/ml_foundation/scope_definer/agent.py",
    "src/agents/ml_foundation/data_preparer/agent.py",
    "src/agents/ml_foundation/model_trainer/agent.py",
    "src/agents/ml_foundation/model_selector/agent.py",
    "src/agents/ml_foundation/feature_analyzer/agent.py",
    "src/agents/ml_foundation/model_deployer/agent.py",
    "src/agents/ml_foundation/observability_connector/agent.py",
]

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent


def test_every_agent_initial_state_threads_audit_workflow_id() -> None:
    """D1.2 static check: every ml_foundation agent.py contains
    ``"audit_workflow_id": input_data.get("audit_workflow_id"),`` inside
    its ``initial_state: <Type> = {...}`` dict literal.

    Regression guard — a refactor that adds a new agent or rewrites the
    initial_state dict construction without threading audit_workflow_id
    will fail this test loudly.
    """
    needle = '"audit_workflow_id": input_data.get("audit_workflow_id")'

    missing: list[str] = []
    for rel in AGENT_FILES:
        path = REPO_ROOT / rel
        assert path.is_file(), f"missing agent file: {path}"
        src = path.read_text()
        if needle not in src:
            missing.append(rel)

    assert not missing, (
        "D1.2 regression — these agents do NOT thread audit_workflow_id "
        f"into initial_state: {missing}. Pattern expected: {needle!r}"
    )


def test_observability_connector_threads_audit_workflow_id_in_both_run_paths() -> None:
    """D1.2: observability_connector has TWO ``initial_state`` dict literals
    (one in ``get_quality_metrics`` at ~line 317, one in ``run`` at ~line
    368). Both must thread audit_workflow_id.
    """
    path = REPO_ROOT / "src/agents/ml_foundation/observability_connector/agent.py"
    src = path.read_text()
    occurrences = src.count('"audit_workflow_id": input_data.get("audit_workflow_id")')
    assert occurrences >= 2, (
        "observability_connector has 2 initial_state dict literals; both "
        f"must thread audit_workflow_id (D1.2 regression). Found: {occurrences}"
    )


def test_model_trainer_build_fold_input_preserves_audit_workflow_id() -> None:
    """D1.2: ``_build_fold_input`` at model_trainer/agent.py:1045 builds
    a per-fold input via ``per_fold = dict(input_data)``. Since input_data
    carries audit_workflow_id (post-D1.2), the dict copy preserves it.
    Pin this so a future refactor that filters keys does not silently
    drop the workflow UUID across fold recursion.
    """
    path = REPO_ROOT / "src/agents/ml_foundation/model_trainer/agent.py"
    src = path.read_text()
    # The fold builder uses ``dict(input_data)`` which preserves all keys.
    # If a future refactor switches to a key-filtered copy, this regex catches it.
    fold_builder_match = re.search(
        r"def _build_fold_input\(.*?\)\s*->\s*Dict\[str,\s*Any\]:",
        src,
        re.DOTALL,
    )
    assert fold_builder_match, (
        "_build_fold_input signature not found — refactor moved or renamed it"
    )
    # Locate the per_fold = dict(input_data) line.
    per_fold_match = re.search(r"per_fold\s*=\s*dict\(input_data\)", src)
    assert per_fold_match, (
        "_build_fold_input no longer uses ``per_fold = dict(input_data)`` — "
        "refactor may have lost audit_workflow_id across fold recursion. "
        "Verify the new code path still copies audit_workflow_id."
    )


def test_data_preparer_does_not_pop_audit_workflow_id() -> None:
    """D1.2 sanity: data_preparer's initial_state pulls audit_workflow_id
    via ``input_data.get(...)`` and the rest of run() does NOT pop it
    from input_data before downstream code reads it (defensive check
    against regressions where someone adds a ``input_data.pop`` for
    cleanup that accidentally drops the workflow UUID).
    """
    path = REPO_ROOT / "src/agents/ml_foundation/data_preparer/agent.py"
    src = path.read_text()
    # Should NOT have ``input_data.pop("audit_workflow_id"`` anywhere.
    assert 'input_data.pop("audit_workflow_id"' not in src
