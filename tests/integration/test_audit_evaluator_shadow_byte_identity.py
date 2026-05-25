"""Stage-1 shadow-mode byte-identity invariant for the audit-evaluator gate.

Design reference: ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 1
acceptance criterion AC1.2 — *severity AND remediation byte-identical when any
rule flag fires* (shadow truly shadows; no behaviour change).

What the tests pin
==================

1. **Byte-identity of the existing audit-only payload.** With the same
   :class:`EnsembleVerdict` input, the bytes of the legacy dict (excluding the
   three Stage-1 shadow keys themselves) must equal the bytes of the legacy
   dict produced by a mocked ``PROMOTION_RULES`` registry that returns ``None``
   for every rule. Concretely: adding the shadow path cannot mutate
   ``severity``, ``remediation``, the 5 ``evaluator_audit`` keys, or any other
   pre-existing field in :func:`_ensemble_to_legacy_dict`.

2. **The three shadow columns are populated when their triggers fire and
   ``NULL`` otherwise.** Direct read-after-write check.

3. **The 5 protected ``evaluator_audit`` write paths at the
   ``adaptive_validity_check.py:1272-1276 / 1351-1355 / 1420-1424 / 1484-1488``
   line ranges remain UNTOUCHED in their value semantics.** Pinned by
   constructing the three bypass-path verdicts and asserting both the existing
   five keys and the new three shadow keys are all ``None`` (the existing
   nullability contract from §3 Stage 1 Mechanism).
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _ensemble_to_legacy_dict,
    _legacy_adversarial_alone_verdict,
    _legacy_info_verdict,
    _legacy_short_circuit_verdict,
)
from src.data import evaluator_promotion_rules as rules_mod
from src.data.kg.types import (
    EnsembleVerdict,
    LLMEvaluatorAudit,
    LLMVerdict,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _audit(
    *,
    satisfied: bool,
    rationale_complete: bool = True,
    missed: tuple[str, ...] = (),
) -> LLMEvaluatorAudit:
    return LLMEvaluatorAudit(
        satisfied=satisfied,
        rationale_complete=rationale_complete,
        missed_considerations=missed,
        notes="",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )


def _llm_verdict_with_audit(audit: LLMEvaluatorAudit | None) -> LLMVerdict:
    return LLMVerdict(
        causal_role="confounder",
        mechanism="cites temporal window [anchor-180, anchor] and Pearl arrows",
        recommended_remediation="keep_with_caveat",
        evaluator_audit=audit,
    )


def _make_ensemble_verdict(
    *,
    severity: str,
    audit: LLMEvaluatorAudit | None,
) -> EnsembleVerdict:
    return EnsembleVerdict(
        feature_name="feat_x",
        severity=severity,  # type: ignore[arg-type]
        remediation="keep_with_caveat",
        decided_by="llm",
        confidence=0.7,
        final_role="confounder",
        evidence=("layer-4 llm",),
        llm_input=_llm_verdict_with_audit(audit),
    )


# Stage-1 shadow keys produced by `_ensemble_to_legacy_dict`. Excluded from
# the byte-identity check (their existence is the whole point of Stage 1) so
# the remaining fields can be diffed.
_SHADOW_KEYS = frozenset(
    {
        "would_promote_severity",
        "would_flag_for_review",
        "rationale_incomplete_flag",
    }
)


def _without_shadow_keys(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in _SHADOW_KEYS}


def _canonical_bytes(payload: dict) -> bytes:
    """Stable bytes for the legacy dict.

    ``LLMEvaluatorAudit.missed_considerations`` is a tuple in the producer,
    which ``json.dumps(default=list)`` coerces to a JSON array — making the
    bytes stable across the worker-vs-mocked-rules comparison.
    """
    return json.dumps(payload, sort_keys=True, default=list).encode("utf-8")


# ---------------------------------------------------------------------------
# AC1.2 byte-identity invariant — main path (`_ensemble_to_legacy_dict`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "severity, audit",
    [
        # R1 should fire: info + dissatisfied + ≥1 missed (reframed
        # info→moderate; the audited path is high/info, never moderate).
        ("info", _audit(satisfied=False, missed=("temporal_filter",))),
        # R2 should fire: ≥2 missed (and R1 too, on info).
        (
            "info",
            _audit(satisfied=False, missed=("temporal_filter", "pearl_arrows")),
        ),
        # R3 should fire: rationale_complete=False.
        (
            "info",
            _audit(satisfied=True, rationale_complete=False),
        ),
        # No rule fires: satisfied + complete + no missed.
        ("info", _audit(satisfied=True, rationale_complete=True)),
        # Worker severity high (leak role): R1 cannot fire (precondition is
        # info) even if evaluator dissatisfied; R2 still fires here.
        ("high", _audit(satisfied=False, missed=("temporal_filter",))),
        # Real moderate candidate (adversarial-alone) carries no audit.
        ("moderate", None),
    ],
)
def test_shadow_path_is_byte_identical_to_disabled_rules(monkeypatch, severity, audit):
    """For every (severity, audit) tuple, the legacy dict produced with the
    real promotion rules MUST equal — modulo the three shadow keys — the
    legacy dict produced when every rule is forced to return ``None``."""
    verdict = _make_ensemble_verdict(severity=severity, audit=audit)

    # "shadow on" — real rules.
    payload_on = _ensemble_to_legacy_dict(verdict, adversarial_input=None)

    # "shadow off" — every rule patched to return None. Patch the registry the
    # producer iterates so the shadow keys land as None.
    disabled_registry = tuple((rid, lambda _s, _a: None) for rid, _ in rules_mod.PROMOTION_RULES)
    monkeypatch.setattr(rules_mod, "PROMOTION_RULES", disabled_registry)
    payload_off = _ensemble_to_legacy_dict(verdict, adversarial_input=None)

    # 1) The shadow-stripped bytes are identical.
    assert _canonical_bytes(_without_shadow_keys(payload_on)) == _canonical_bytes(
        _without_shadow_keys(payload_off)
    )

    # 2) severity AND remediation are byte-identical (the explicit AC1.2
    # invariant, restated as a focused assertion the grep-able way).
    assert payload_on["severity"] == payload_off["severity"] == verdict.severity
    assert payload_on["remediation"] == payload_off["remediation"] == verdict.remediation

    # 3) Disabled-rules path leaves all three shadow keys None.
    assert payload_off["would_promote_severity"] is None
    assert payload_off["would_flag_for_review"] is None
    assert payload_off["rationale_incomplete_flag"] is None


# ---------------------------------------------------------------------------
# Shadow-column population (per-rule)
# ---------------------------------------------------------------------------


def test_r1_fires_populates_would_promote_severity_moderate():
    verdict = _make_ensemble_verdict(
        severity="info",
        audit=_audit(satisfied=False, missed=("temporal_filter",)),
    )
    payload = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert payload["would_promote_severity"] == "moderate"


def test_r2_fires_populates_would_flag_for_review_true():
    verdict = _make_ensemble_verdict(
        severity="moderate",
        audit=_audit(satisfied=False, missed=("temporal_filter", "pearl_arrows")),
    )
    payload = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert payload["would_flag_for_review"] is True


def test_r3_fires_populates_rationale_incomplete_flag_true():
    verdict = _make_ensemble_verdict(
        severity="moderate",
        audit=_audit(satisfied=True, rationale_complete=False),
    )
    payload = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert payload["rationale_incomplete_flag"] is True


def test_all_three_shadow_keys_null_when_no_rule_fires():
    verdict = _make_ensemble_verdict(
        severity="moderate",
        audit=_audit(satisfied=True, rationale_complete=True),
    )
    payload = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert payload["would_promote_severity"] is None
    assert payload["would_flag_for_review"] is None
    assert payload["rationale_incomplete_flag"] is None


def test_all_three_shadow_keys_null_when_evaluator_audit_absent():
    """LLM verdict without evaluator_audit (evaluator disabled / failed).

    Fail-open semantics per §3 Stage 3 Rollback (Stages 1-2 preserve this
    via the audit being None → every rule returns None)."""
    verdict = _make_ensemble_verdict(severity="moderate", audit=None)
    payload = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert payload["would_promote_severity"] is None
    assert payload["would_flag_for_review"] is None
    assert payload["rationale_incomplete_flag"] is None


# ---------------------------------------------------------------------------
# 5 protected evaluator_audit fields UNTOUCHED — bypass paths leave shadow
# keys NULL alongside the existing 5 None fields.
# ---------------------------------------------------------------------------

# These bind the on-disk constants the design doc cites at line ranges
# 1272-1276 (main path) / 1351-1355 (adversarial-only) /
# 1420-1424 (info-only) / 1484-1488 (short-circuit). Bypass dicts MUST
# carry the new shadow keys as None for sidecar-schema uniformity.
_PROTECTED_EVALUATOR_KEYS = (
    "evaluator_satisfied",
    "evaluator_rationale_complete",
    "evaluator_missed_considerations",
    "evaluator_notes",
    "evaluator_model",
)


def test_adversarial_only_bypass_keeps_protected_fields_none_and_shadow_keys_none():
    adv_input = {
        "z_score": 3.2,
        "actual_auc": 0.55,
        "null_mean": 0.50,
        "null_std": 0.02,
        "p_value": 0.001,
        "n_permutations": 200,
        "severity": "moderate",
        "remediation": "ambiguous",
        "evidence": "adv-only",
        "severity_pre_joint_check": "moderate",
    }
    payload = _legacy_adversarial_alone_verdict("feat_x", adv_input)
    for key in _PROTECTED_EVALUATOR_KEYS:
        assert payload[key] is None, key
    for key in _SHADOW_KEYS:
        assert key in payload and payload[key] is None, key


def test_info_only_bypass_keeps_protected_fields_none_and_shadow_keys_none():
    payload = _legacy_info_verdict("feat_x", adversarial_input=None, evidence="info")
    for key in _PROTECTED_EVALUATOR_KEYS:
        assert payload[key] is None, key
    for key in _SHADOW_KEYS:
        assert key in payload and payload[key] is None, key


def test_short_circuit_bypass_keeps_protected_fields_none_and_shadow_keys_none():
    payload = _legacy_short_circuit_verdict("feat_x", evidence="too-few-rows")
    for key in _PROTECTED_EVALUATOR_KEYS:
        assert payload[key] is None, key
    for key in _SHADOW_KEYS:
        assert key in payload and payload[key] is None, key


# ---------------------------------------------------------------------------
# Defensive: deep copy of input verdict is not mutated by the shadow path.
# ---------------------------------------------------------------------------


def test_ensemble_verdict_inputs_are_not_mutated_by_shadow_rules():
    audit = _audit(satisfied=False, missed=("temporal_filter",))
    verdict = _make_ensemble_verdict(severity="moderate", audit=audit)
    verdict_snapshot = deepcopy(verdict)
    _ = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    # Severity / remediation on the input verdict unchanged.
    assert verdict.severity == verdict_snapshot.severity
    assert verdict.remediation == verdict_snapshot.remediation
    # Audit field on the input verdict unchanged.
    assert verdict.llm_input is not None
    assert verdict.llm_input.evaluator_audit == verdict_snapshot.llm_input.evaluator_audit  # type: ignore[union-attr]
