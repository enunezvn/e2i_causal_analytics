"""Issue #240 Stage 3 — population wiring for the soft-gate audit columns.

Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 3
+ §5 R-4 (worker_severity_pre_gate audit-loop-coupling mitigation).

Pins that the two Stage-3 keys are NOT inert (the gap that bit Stage 1):

1. ``_ensemble_to_legacy_dict`` surfaces ``gate_rule_fired`` and
   ``worker_severity_pre_gate`` from the EnsembleVerdict, including the
   R-4 invariant that a gate-flipped verdict records the *un-mutated*
   worker severity ("moderate") rather than the escalated "high".
2. The bypass-path legacy dicts carry both keys as None (schema uniformity).
3. The keys round-trip through the SidecarReader onto VerdictRecord and are
   registered in ``_KNOWN_VERDICT_KEYS`` (no unknown-key WARN).
4. The reader's pinned schema version is bumped to 1.3.
"""

from __future__ import annotations

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _ensemble_to_legacy_dict,
    _legacy_adversarial_alone_verdict,
    _legacy_info_verdict,
    _legacy_short_circuit_verdict,
)
from src.data.audit_sidecar_reader import (
    _KNOWN_VERDICT_KEYS,
    _READER_SCHEMA_VERSION,
)
from src.data.kg.types import EnsembleVerdict, LLMEvaluatorAudit, LLMVerdict

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _llm_verdict() -> LLMVerdict:
    return LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
        evaluator_audit=LLMEvaluatorAudit(
            satisfied=False,
            rationale_complete=True,
            missed_considerations=("temporal_filter",),
            notes="",
            evaluator_model="anthropic/claude-haiku-4-5-20251001",
        ),
    )


def _gate_flipped_verdict() -> EnsembleVerdict:
    """A verdict as the voter's gate would emit after R1 escalates
    info→moderate: severity=moderate, remediation=review,
    decided_by=evaluator_gate, gate_rule_fired=R1 (reframed 2026-05-25)."""
    return EnsembleVerdict(
        feature_name="feat_x",
        severity="moderate",
        remediation="review",
        decided_by="evaluator_gate",
        confidence=0.6,
        final_role="ancestor",
        evidence=("layer-4 llm", "evaluator_gate:R1:info→moderate"),
        llm_input=_llm_verdict(),
        gate_rule_fired="R1",
    )


def _ungated_verdict() -> EnsembleVerdict:
    return EnsembleVerdict(
        feature_name="feat_x",
        severity="moderate",
        remediation="review",
        decided_by="adversarial",
        confidence=0.6,
        final_role=None,
        evidence=("Adversarial probe: severity=moderate",),
        llm_input=_llm_verdict(),
        gate_rule_fired=None,
    )


# ---------------------------------------------------------------------------
# (1) main-path population + R-4 invariant
# ---------------------------------------------------------------------------


def test_gate_flipped_verdict_records_pre_gate_worker_severity():
    payload = _ensemble_to_legacy_dict(_gate_flipped_verdict(), adversarial_input=None)
    assert payload["gate_rule_fired"] == "R1"
    # R-4: the un-mutated worker severity ("info"), NOT the escalated
    # "moderate" — so curation never trains on the gate-escalated label.
    assert payload["worker_severity_pre_gate"] == "info"
    # The mutated value still flows to ``severity`` (consumer compatibility).
    assert payload["severity"] == "moderate"


def test_ungated_verdict_leaves_gate_keys_null():
    payload = _ensemble_to_legacy_dict(_ungated_verdict(), adversarial_input=None)
    assert payload["gate_rule_fired"] is None
    assert payload["worker_severity_pre_gate"] is None
    assert payload["severity"] == "moderate"


# ---------------------------------------------------------------------------
# (2) bypass-path schema uniformity
# ---------------------------------------------------------------------------


def test_bypass_paths_carry_gate_keys_as_none():
    adv_input = {
        "z_score": 3.2,
        "severity": "moderate",
        "remediation": "ambiguous",
        "evidence": "adv-only",
        "severity_pre_joint_check": "moderate",
    }
    payloads = [
        _legacy_adversarial_alone_verdict("feat_x", adv_input),
        _legacy_info_verdict("feat_x", adversarial_input=None, evidence="info"),
        _legacy_short_circuit_verdict("feat_x", evidence="too-few-rows"),
    ]
    for payload in payloads:
        assert "gate_rule_fired" in payload and payload["gate_rule_fired"] is None
        assert "worker_severity_pre_gate" in payload and payload["worker_severity_pre_gate"] is None


# ---------------------------------------------------------------------------
# (3) reader round-trip + (4) schema-version bump
# ---------------------------------------------------------------------------


def test_reader_known_keys_include_gate_keys():
    assert "gate_rule_fired" in _KNOWN_VERDICT_KEYS
    assert "worker_severity_pre_gate" in _KNOWN_VERDICT_KEYS


def test_reader_schema_version_is_1_5():
    # Bumped 1.4 → 1.5 by Issue #501 (additive M-structure structural-remediation
    # gate shadow keys), mirroring the per-additive-keyset minor bump #508 made
    # (1.3 → 1.4 for the leak-crosscheck key) and #240 Stage 3 (→ 1.3). MAJOR
    # stays 1 (additive, nullable, backward-compatible).
    assert _READER_SCHEMA_VERSION == "1.5"


def test_reader_surfaces_gate_keys_onto_verdict_record(tmp_path):
    import json
    from datetime import datetime, timezone

    from src.data.audit_sidecar_reader import SidecarReader

    payload = _ensemble_to_legacy_dict(_gate_flipped_verdict(), adversarial_input=None)
    sidecar = {
        "schema_version": "1.3",
        "experiment_id": "exp-1",
        "written_at": datetime(2026, 5, 24, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
        "adaptive_verdicts": [payload],
        "role_attributions": [],
    }
    p = tmp_path / "adaptive_verdicts_exp-1.json"
    p.write_text(json.dumps(sidecar))

    records = list(SidecarReader(artifacts_dir=tmp_path).iter_verdict_records())
    assert len(records) == 1
    rec = records[0]
    assert rec.gate_rule_fired == "R1"
    assert rec.worker_severity_pre_gate == "info"
