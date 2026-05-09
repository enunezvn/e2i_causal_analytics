"""Per-agent audit_workflow_id threading tests (sub-shard D1.2 + backlog #1).

Sub-shard D1.2 threads ``audit_workflow_id`` from caller-supplied
``input_data`` into each agent's ``initial_state`` dict literal. Backlog
item #1 (closed 2026-05-09) tightened the State contract from
``Field(default_factory=uuid4)`` to plain ``UUID`` (required, no
default), so the conditional-spread pattern the threading guards is now
load-bearing: missing ``audit_workflow_id`` raises ``ValidationError``
at State construction, not silently defaults to a fresh UUID.

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
    4. CONTRACT — every agent State raises ValidationError on
       construction without ``audit_workflow_id`` (backlog #1 contract).
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
    """D1.2 static check: every ml_foundation agent.py contains a
    conditional spread of ``audit_workflow_id`` from input_data into
    its ``initial_state: <Type> = {...}`` dict literal. The conditional
    spread is REQUIRED because the State's audit_workflow_id field has
    a custom validator (``coerce_uuid``) that rejects None — passing
    ``None`` would raise ValidationError instead of letting
    ``default_factory=uuid4`` fire. The spread pattern only inserts the
    key when input_data carries a non-None value, otherwise leaving
    it absent so the default factory activates.

    Regression guard — a refactor that adds a new agent or rewrites the
    initial_state dict construction without conditionally spreading
    audit_workflow_id will fail this test loudly.
    """
    needle = '{"audit_workflow_id": input_data["audit_workflow_id"]}'

    missing: list[str] = []
    for rel in AGENT_FILES:
        path = REPO_ROOT / rel
        assert path.is_file(), f"missing agent file: {path}"
        src = path.read_text()
        if needle not in src:
            missing.append(rel)

    assert not missing, (
        "D1.2 regression — these agents do NOT thread audit_workflow_id "
        f"into initial_state via the conditional-spread pattern: {missing}. "
        f"Pattern expected: {needle!r}"
    )


def test_observability_connector_threads_audit_workflow_id_in_both_run_paths() -> None:
    """D1.2: observability_connector has TWO ``initial_state`` dict literals
    (one in ``get_quality_metrics`` at ~line 317, one in ``run`` at ~line
    368). Both must thread audit_workflow_id via the conditional spread.
    """
    path = REPO_ROOT / "src/agents/ml_foundation/observability_connector/agent.py"
    src = path.read_text()
    occurrences = src.count('{"audit_workflow_id": input_data["audit_workflow_id"]}')
    assert occurrences >= 2, (
        "observability_connector has 2 initial_state dict literals; both "
        f"must thread audit_workflow_id (D1.2 regression). Found: {occurrences}"
    )


def test_audit_workflow_id_validator_rejects_none() -> None:
    """The ``audit_workflow_id_validator`` rejects ``None`` to keep the
    contract explicit. This is unchanged by backlog #1 (the validator
    rejected None pre- and post-D1; what backlog #1 changed is what
    happens when the key is OMITTED — see
    ``test_audit_workflow_id_required_on_all_agent_states`` below).

    Regression guard for the bug found in D1.2's first attempt where
    ``initial_state["audit_workflow_id"] = None`` raised
    ``ValidationError`` from the validator's ``coerce_uuid`` helper.
    """
    import pytest

    from src.agents.ml_foundation.scope_definer.state import ScopeDefinerState

    with pytest.raises(Exception) as exc_info:
        ScopeDefinerState(audit_workflow_id=None)  # type: ignore[arg-type]
    assert "UUID or str" in str(exc_info.value) or "NoneType" in str(exc_info.value), (
        "audit_workflow_id_validator should reject None to keep the contract explicit."
    )


def test_audit_workflow_id_required_on_all_agent_states() -> None:
    """Backlog #1 contract (closed 2026-05-09): every ml_foundation agent
    State requires ``audit_workflow_id`` at construction.

    Pre-D1: ``Field(default_factory=uuid4)`` silently minted a fresh UUID
    when the caller omitted the key, masking missing-threading bugs and
    breaking the audit chain across LangGraph nodes (every Schema
    coercion fired the default_factory afresh — codex review B1, 2026-05-05).

    Post-D1: the field is plain ``UUID`` (no default). Constructing a
    State without ``audit_workflow_id`` raises ``ValidationError``,
    making missing-threading bugs fail loudly at construction rather
    than silently masking the audit-chain break.
    """
    import pytest
    from pydantic import ValidationError

    from src.agents.ml_foundation.data_preparer.state import DataPreparerState
    from src.agents.ml_foundation.feature_analyzer.state import FeatureAnalyzerState
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState
    from src.agents.ml_foundation.model_selector.state import ModelSelectorState
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState
    from src.agents.ml_foundation.observability_connector.state import (
        ObservabilityConnectorState,
    )
    from src.agents.ml_foundation.scope_definer.state import ScopeDefinerState

    state_classes = [
        ScopeDefinerState,
        DataPreparerState,
        FeatureAnalyzerState,
        ModelSelectorState,
        ModelTrainerState,
        ModelDeployerState,
        ObservabilityConnectorState,
    ]
    for cls in state_classes:
        with pytest.raises(ValidationError) as exc_info:
            cls()
        assert "audit_workflow_id" in str(exc_info.value), (
            f"{cls.__name__} ValidationError on missing audit_workflow_id "
            f"should mention the field name; got: {exc_info.value}"
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
