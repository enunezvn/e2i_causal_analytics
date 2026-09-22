"""Lane E — the legacy verdict dict carries the ensemble role, confidence and
the LLM mechanism, and Layer 4 sees the causal treatment.

The feature-role panel (``src/causal_engine/feature_role_panel.py``) reuses the
data-preparer node's four layers and reads its verdict dicts. Before this lane
the legacy adapter dropped three things the spec's panel needs
(``EnsembleVerdict.final_role``, ``EnsembleVerdict.confidence`` and
``LLMVerdict.mechanism``), so a panel could only have re-derived them by
re-implementing the voter's precedence — exactly what the spec forbids. They are
additive, nullable, emitted on EVERY producer path (voter-routed and the three
bypasses) and registered with the sidecar reader (schema 1.9).
"""

from __future__ import annotations

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _build_layer_4_inputs,
    _compose_legacy_verdict,
    _ensemble_to_legacy_dict,
    _get_ensemble_voter_class,
    _layer_1_input,
    _legacy_adversarial_alone_verdict,
    _legacy_info_verdict,
    _legacy_short_circuit_verdict,
)
from src.data.feature_contract import FeatureContract, KnowableAt
from src.data.kg.types import EnsembleVerdict, LLMVerdict

_NEW_KEYS = ("final_role", "confidence", "llm_mechanism", "citation_verdicts")


def _post_index_contract() -> FeatureContract:
    return FeatureContract(
        name="treatment_initiated",
        knowable_at=KnowableAt(reference="post_index"),
        source="derived",
    )


def test_voter_routed_verdict_carries_role_confidence_and_mechanism() -> None:
    voter = _get_ensemble_voter_class()()
    legacy = _compose_legacy_verdict(
        "treatment_initiated",
        voter=voter,
        layer_1_input=_layer_1_input("treatment_initiated", _post_index_contract()),
    )
    for key in _NEW_KEYS:
        assert key in legacy, key
    # A Layer-1 veto is a deterministic leak: the voter assigns the conventional
    # ``descendant`` role at confidence 1.0 — surfaced, not re-derived.
    assert legacy["final_role"] == "descendant"
    assert legacy["confidence"] == 1.0
    assert legacy["llm_mechanism"] is None


def test_llm_mechanism_is_the_llm_verdicts_mechanism() -> None:
    verdict = EnsembleVerdict(
        feature_name="f",
        severity="info",
        remediation="keep",
        decided_by="adversarial",
        confidence=0.4,
        final_role="confounder",
        llm_input=LLMVerdict(
            causal_role="confounder",
            mechanism="baseline severity drives both choice and persistence",
            recommended_remediation="keep_with_caveat",
        ),
    )
    legacy = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert legacy["llm_mechanism"] == "baseline severity drives both choice and persistence"
    assert legacy["final_role"] == "confounder"
    assert legacy["confidence"] == 0.4


def test_per_citation_verdicts_cross_the_legacy_boundary() -> None:
    """codex r1 MED: counts alone cannot tell a reviewer WHY a citation passed or
    failed; the serialised CitationVerdict records must survive the adapter."""
    from src.data.kg.types import CitationVerdict

    ok = CitationVerdict(
        identifier="12345678",
        identifier_kind="pmid",
        abstract_resolved=True,
        entities_found=("omalizumab", "urticaria"),
        causal_cue_found="reduces",
        overall_confidence=0.9,
    )
    bad = CitationVerdict(
        identifier="99999999",
        identifier_kind="pmid",
        abstract_resolved=False,
        error="Europe PMC: 404",
    )
    verdict = EnsembleVerdict(
        feature_name="f",
        severity="info",
        remediation="keep",
        decided_by="adversarial",
        confidence=0.4,
        verified_citations=(ok,),
        unverified_citations=(bad,),
        llm_input=LLMVerdict(
            causal_role="confounder",
            mechanism="m (PMID:12345678, PMID:99999999)",
            recommended_remediation="keep_with_caveat",
            cited_pmids=("12345678", "99999999"),
        ),
    )
    legacy = _ensemble_to_legacy_dict(verdict, adversarial_input=None)
    assert legacy["citations_checked"] == 2 and legacy["citations_verified"] == 1
    records = {r["identifier"]: r for r in legacy["citation_verdicts"]}
    assert records["12345678"]["verified"] is True
    assert records["12345678"]["entities_found"] == ["omalizumab", "urticaria"]
    assert records["12345678"]["causal_cue_found"] == "reduces"
    assert records["12345678"]["overall_confidence"] == 0.9
    assert records["99999999"]["verified"] is False
    assert records["99999999"]["abstract_resolved"] is False
    assert records["99999999"]["error"] == "Europe PMC: 404"
    from src.data.audit_sidecar_reader import _KNOWN_VERDICT_KEYS

    assert "citation_verdicts" in _KNOWN_VERDICT_KEYS


def test_bypass_paths_emit_the_keys_as_nulls() -> None:
    """Schema uniformity: the sidecar and the panel must never KeyError."""
    adv = {
        "feature": "x",
        "layer": "3",
        "severity": "info",
        "remediation": "keep",
        "evidence": "e",
        "z_score": 0.1,
        "_hblp_classified": True,
    }
    for legacy in (
        _legacy_short_circuit_verdict("x", evidence="too few rows"),
        _legacy_info_verdict("x", adversarial_input=adv, evidence="tested and passed"),
        _legacy_adversarial_alone_verdict("x", adv),
    ):
        for key in _NEW_KEYS:
            assert key in legacy, (key, legacy.get("decided_by"))
            assert legacy[key] in (None, []), (key, legacy[key])


def test_reader_knows_the_keys_and_the_schema_is_1_9() -> None:
    from src.data.audit_sidecar_reader import _KNOWN_VERDICT_KEYS, SIDECAR_SCHEMA_VERSION

    for key in _NEW_KEYS:
        assert key in _KNOWN_VERDICT_KEYS, key
    assert SIDECAR_SCHEMA_VERSION == "1.9"


def test_layer_4_dataset_context_names_the_causal_treatment_only_when_asked() -> None:
    """The prediction path is byte-identical (no treatment); the causal panel
    passes T so the classifier's role question is asked relative to it."""
    contract = FeatureContract(
        name="cci_chronic_pulmonary",
        knowable_at=KnowableAt(reference="index_date"),
        source="mart_comorbidity",
    )
    _d0, ctx0 = _build_layer_4_inputs(
        "cci_chronic_pulmonary", contract, "persistent_at_180d_g28", "optum_mart"
    )
    assert ctx0 == (
        "cohort=optum_mart; target=persistent_at_180d_g28; prediction_anchor=index_date"
    )
    _d1, ctx1 = _build_layer_4_inputs(
        "cci_chronic_pulmonary",
        contract,
        "persistent_at_180d_g28",
        "optum_mart",
        treatment="treatment_dupixent",
    )
    assert ctx1.startswith(ctx0)
    assert "treatment=treatment_dupixent" in ctx1
    assert "causal_question=effect of treatment_dupixent on persistent_at_180d_g28" in ctx1
