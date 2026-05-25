"""Issue #501 — node-level structural-attestation consumer (plan §8.4, §0 row 17).

Tests ``_apply_structural_attestation`` (the in-loop consumer wired ALONGSIDE
#508's ``would_flag_role_leak_disagreement`` line) for:
  * collider narrows remediation to {drop} when the gate is ON;
  * dark-launch: telemetry recorded but NO remediation override when gate OFF;
  * un-attested feature → full no-op;
  * coexistence: #508's key is never clobbered (the dict is mutated in place,
    not reassigned — guards the sibling-agent `return {}` bug).
"""

from __future__ import annotations

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _apply_structural_attestation,
)
from src.data.feature_contract import (
    CausalStructureAttestation,
    FeatureContract,
    KnowableAt,
)


def _m_structure_contract(name: str = "on_treatment_at_12m_flag") -> FeatureContract:
    """A contract attesting the M-structure T→V←U→Y (derived role = collider)."""
    return FeatureContract(
        name=name,
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("treatment_status",),
        causal_role="collider",
        causal_structure=CausalStructureAttestation(
            treatment_node="T",
            outcome_node="Y",
            feature_node="V",
            edges=(("T", "V"), ("U", "V"), ("U", "Y")),
        ),
    )


def _base_verdict(**overrides):
    """A verdict dict shaped like the producer output, with the #501 keys + #508 key."""
    v = {
        "feature": "on_treatment_at_12m_flag",
        "severity": "high",
        "remediation": "window",
        "llm_role": "mediator",
        "llm_remediation": "window",
        # #508 key (precomputed BEFORE the structural consumer runs).
        "would_flag_role_leak_disagreement": None,
        # #501 keys (None defaults from the producer).
        "structural_role": None,
        "structural_llm_disagreement": None,
        "structural_remediation_override": None,
        "structural_gate_fired": None,
    }
    v.update(overrides)
    return v


def test_structural_gate_collider_narrows_remediation_to_drop(monkeypatch) -> None:
    """Gate ON + collider attestation vs llm=mediator → remediation forced drop,
    structural_gate_fired=R-STRUCT, severity UNCHANGED, #508 key UNCHANGED."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    verdict = _base_verdict()
    _apply_structural_attestation(verdict, _m_structure_contract())

    assert verdict["structural_role"] == "collider"
    assert verdict["structural_llm_disagreement"] is True
    assert verdict["remediation"] == "drop"
    assert verdict["structural_remediation_override"] == "drop"
    assert verdict["structural_gate_fired"] == "R-STRUCT"
    # Severity NEVER mutated by the structural gate.
    assert verdict["severity"] == "high"
    # #508 key preserved (not clobbered by a dict reassignment).
    assert verdict["would_flag_role_leak_disagreement"] is None


def test_structural_gate_dark_launch_records_telemetry_only(monkeypatch) -> None:
    """Gate OFF (default): telemetry keys recorded, but NO remediation override
    and structural_gate_fired stays None (dark-launchable)."""
    monkeypatch.delenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", raising=False)
    verdict = _base_verdict()
    _apply_structural_attestation(verdict, _m_structure_contract())

    assert verdict["structural_role"] == "collider"
    assert verdict["structural_llm_disagreement"] is True
    # No override when the gate is off.
    assert verdict["remediation"] == "window"
    assert verdict["structural_remediation_override"] is None
    assert verdict["structural_gate_fired"] is None


def test_structural_gate_unattested_feature_no_op(monkeypatch) -> None:
    """No causal_structure → full no-op; all structural keys stay None and the
    remediation is unchanged, even with the gate ON."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    contract = FeatureContract(
        name="age_at_index",
        knowable_at=KnowableAt(reference="index_date"),
        source="demo",
        derivation_inputs=("birth_date",),
    )
    verdict = _base_verdict()
    _apply_structural_attestation(verdict, contract)

    assert verdict["structural_role"] is None
    assert verdict["structural_llm_disagreement"] is None
    assert verdict["structural_remediation_override"] is None
    assert verdict["structural_gate_fired"] is None
    assert verdict["remediation"] == "window"


def test_structural_gate_none_contract_no_op(monkeypatch) -> None:
    """contract=None (no manifest entry) → full no-op."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    verdict = _base_verdict()
    _apply_structural_attestation(verdict, None)
    assert verdict["structural_role"] is None
    assert verdict["remediation"] == "window"


def test_structural_keys_coexist_with_508_leak_crosscheck(monkeypatch) -> None:
    """A verdict that ALSO carries #508's would_flag_role_leak_disagreement=True
    keeps that value after the structural consumer runs (no clobbering)."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    verdict = _base_verdict(would_flag_role_leak_disagreement=True)
    _apply_structural_attestation(verdict, _m_structure_contract())

    # #508 key untouched by the structural consumer.
    assert verdict["would_flag_role_leak_disagreement"] is True
    # #501 structural keys set independently.
    assert verdict["structural_role"] == "collider"
    assert verdict["structural_gate_fired"] == "R-STRUCT"


def test_structural_gate_descendant_does_not_overrestrict(monkeypatch) -> None:
    """structural=descendant vs llm=collider, LLM proposed transform → the gate
    widens back to the descendant-permitted transform (not over-restricted)."""
    monkeypatch.setenv("ADAPTIVE_VALIDITY_STRUCTURAL_GATE_ENABLED", "1")
    contract = FeatureContract(
        name="hepatotoxicity_grade3_post_index_flag",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("lft",),
        causal_role="descendant",
        causal_structure=CausalStructureAttestation(
            treatment_node="T",
            outcome_node="Y",
            feature_node="V",
            edges=(("T", "V"), ("T", "Y")),  # off-path descendant
        ),
    )
    verdict = _base_verdict(
        feature="hepatotoxicity_grade3_post_index_flag",
        remediation="drop",
        llm_role="collider",
        llm_remediation="transform",
    )
    _apply_structural_attestation(verdict, contract)
    assert verdict["structural_role"] == "descendant"
    assert verdict["structural_llm_disagreement"] is True
    assert verdict["remediation"] == "transform"
    assert verdict["structural_remediation_override"] == "transform"
    assert verdict["structural_gate_fired"] == "R-STRUCT"
