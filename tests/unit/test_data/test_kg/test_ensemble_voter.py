"""Unit tests for `src.data.kg.ensemble_voter`.

Covers the precedence rules, KG-signal classification, citation
verification, abstain triggers, and audit-trail invariants documented
in the module docstring.
"""

from __future__ import annotations

import pytest

from src.data.kg.ensemble_voter import (
    ADVERSARIAL_HIGH_CONFIDENCE,
    ADVERSARIAL_MODERATE_CONFIDENCE,
    KG_ONLY_CONFIDENCE,
    LAYER_1_CONFIDENCE,
    LLM_ADVERSARIAL_MODERATE_PENALTY,
    LLM_BASE_CONFIDENCE,
    LLM_CITATION_FAIL_PENALTY,
    LLM_KG_CORROBORATION_BONUS,
    LLM_NO_CITATION_PENALTY,
    EnsembleVoter,
    classify_kg_signal,
    is_citation_verified,
)
from src.data.kg.types import (
    CitationVerdict,
    EnsembleVerdict,
    KGEdge,
    LLMVerdict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_1_high(contract_source: str = "csu") -> dict:
    """Mirror of `adaptive_validity_check._layer_1_verdict` shape."""
    return {
        "feature": "feat_x",
        "layer": "1",
        "severity": "high",
        "remediation": "drop",
        "evidence": "post_index contract",
        "contract_source": contract_source,
        "contract_window_days": None,
    }


def _adversarial(severity: str, z_score: float = 6.0) -> dict:
    """Mirror of `adaptive_validity_check._build_verdict` shape."""
    return {
        "feature": "feat_x",
        "layer": "3",
        "severity": severity,
        "z_score": z_score,
        "actual_auc": 0.8,
        "null_mean": 0.5,
        "null_std": 0.05,
        "p_value": 0.001,
        "n_permutations": 200,
        "remediation": "drop" if severity == "high" else "ambiguous",
        "evidence": f"z={z_score}",
        "contract_source": None,
        "contract_window_days": None,
    }


def _kg_treats_edge(
    *,
    drug_id: str = "CHEMBL:DRUG_X",
    disease_id: str = "EFO:0000270",
    score: float = 0.9,
) -> KGEdge:
    return KGEdge(
        subject_id=drug_id,
        predicate="treats",
        object_id=disease_id,
        evidence_source="open_targets",
        subject_name="DrugX",
        object_name="DiseaseY",
        score=score,
        pmids=("12345678",),
        datasource="europepmc",
    )


def _kg_isa_edge(
    *,
    child_id: str = "C0001",
    parent_id: str = "C0002",
) -> KGEdge:
    return KGEdge(
        subject_id=child_id,
        predicate="isa",
        object_id=parent_id,
        evidence_source="umls_relations",
        subject_name="ChildConcept",
        object_name="ParentConcept",
    )


def _llm_verdict(
    role: str = "descendant",
    cited_pmids: tuple[str, ...] = ("12345678",),
    remediation: str = "drop",
) -> LLMVerdict:
    return LLMVerdict(
        causal_role=role,  # type: ignore[arg-type]
        mechanism=f"role={role} per derivation",
        recommended_remediation=remediation,  # type: ignore[arg-type]
        cited_pmids=cited_pmids,
    )


def _verified_citation(pmid: str = "12345678") -> CitationVerdict:
    return CitationVerdict(
        identifier=pmid,
        identifier_kind="pmid",
        abstract_resolved=True,
        entities_found=("DrugX", "DiseaseY"),
        causal_cue_found="treats",
        overall_confidence=1.0,
    )


def _unverified_citation(pmid: str = "99999999", *, reason: str = "missing_cue") -> CitationVerdict:
    if reason == "missing_cue":
        return CitationVerdict(
            identifier=pmid,
            identifier_kind="pmid",
            abstract_resolved=True,
            entities_found=("DrugX", "DiseaseY"),
            causal_cue_found=None,
            overall_confidence=0.5,
        )
    if reason == "missing_entity":
        return CitationVerdict(
            identifier=pmid,
            identifier_kind="pmid",
            abstract_resolved=True,
            entities_found=("DrugX",),
            causal_cue_found="treats",
            overall_confidence=0.3,
        )
    if reason == "unresolved":
        return CitationVerdict(
            identifier=pmid,
            identifier_kind="pmid",
            abstract_resolved=False,
            error="not found",
        )
    raise ValueError(f"unknown reason {reason!r}")


# ---------------------------------------------------------------------------
# is_citation_verified
# ---------------------------------------------------------------------------


def test_is_citation_verified_passes_complete_record():
    assert is_citation_verified(_verified_citation())


def test_is_citation_verified_rejects_missing_cue():
    assert not is_citation_verified(_unverified_citation(reason="missing_cue"))


def test_is_citation_verified_rejects_single_entity():
    assert not is_citation_verified(_unverified_citation(reason="missing_entity"))


def test_is_citation_verified_rejects_unresolved_abstract():
    assert not is_citation_verified(_unverified_citation(reason="unresolved"))


def test_is_citation_verified_rejects_zero_entities():
    v = CitationVerdict(
        identifier="111",
        identifier_kind="pmid",
        abstract_resolved=True,
        entities_found=(),
        causal_cue_found="treats",
    )
    assert not is_citation_verified(v)


# ---------------------------------------------------------------------------
# classify_kg_signal
# ---------------------------------------------------------------------------


def test_classify_kg_signal_drug_treats_disease():
    edge = _kg_treats_edge(drug_id="DRUG_A", disease_id="DIS_X")
    signal, edges = classify_kg_signal(
        [edge],
        feature_entity_ids=["DRUG_A"],
        target_entity_ids=["DIS_X"],
    )
    assert signal == "leak_drug_treats_disease"
    assert edges == (edge,)


def test_classify_kg_signal_taxonomic():
    edge = _kg_isa_edge(child_id="C_FEAT", parent_id="C_TARGET")
    signal, edges = classify_kg_signal(
        [edge],
        feature_entity_ids=["C_FEAT"],
        target_entity_ids=["C_TARGET"],
    )
    assert signal == "taxonomic_descendant"
    assert edges == (edge,)


def test_classify_kg_signal_inverse_isa_also_taxonomic():
    edge = KGEdge(
        subject_id="C_TARGET",
        predicate="inverse_isa",
        object_id="C_FEAT",
        evidence_source="umls_relations",
    )
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=["C_FEAT"],
        target_entity_ids=["C_TARGET"],
    )
    assert signal == "taxonomic_descendant"


def test_classify_kg_signal_contradictory_when_both():
    treats = _kg_treats_edge(drug_id="A", disease_id="B")
    isa = _kg_isa_edge(child_id="A", parent_id="B")
    signal, edges = classify_kg_signal(
        [treats, isa],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
    )
    assert signal == "contradictory"
    assert len(edges) == 2


def test_classify_kg_signal_no_signal_when_edges_dont_connect():
    edge = _kg_treats_edge(drug_id="A", disease_id="B")
    signal, edges = classify_kg_signal(
        [edge],
        feature_entity_ids=["X"],  # not A
        target_entity_ids=["Y"],  # not B
    )
    assert signal == "no_signal"
    assert edges == ()


def test_classify_kg_signal_no_signal_when_no_edges():
    signal, edges = classify_kg_signal(
        [],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
    )
    assert signal == "no_signal"
    assert edges == ()


def test_classify_kg_signal_no_signal_when_feature_set_empty():
    edge = _kg_treats_edge()
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=[],
        target_entity_ids=["EFO:0000270"],
    )
    assert signal == "no_signal"


def test_classify_kg_signal_no_signal_when_target_set_empty():
    edge = _kg_treats_edge()
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=["CHEMBL:DRUG_X"],
        target_entity_ids=[],
    )
    assert signal == "no_signal"


def test_classify_kg_signal_treats_predicate_case_insensitive():
    edge = KGEdge(
        subject_id="A",
        predicate="TREATS",
        object_id="B",
        evidence_source="open_targets",
    )
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
    )
    assert signal == "leak_drug_treats_disease"


def test_classify_kg_signal_ignores_unrelated_edges():
    treats = _kg_treats_edge(drug_id="A", disease_id="B")
    unrelated = KGEdge(
        subject_id="C",
        predicate="treats",
        object_id="D",
        evidence_source="open_targets",
    )
    signal, edges = classify_kg_signal(
        [treats, unrelated],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
    )
    assert signal == "leak_drug_treats_disease"
    assert edges == (treats,)


def test_classify_kg_signal_empty_string_ids_filtered():
    edge = _kg_treats_edge(drug_id="A", disease_id="B")
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=["", "A"],
        target_entity_ids=["", "B"],
    )
    assert signal == "leak_drug_treats_disease"


def test_classify_kg_signal_treats_predicate_only_from_open_targets():
    """A `treats` predicate from a non-open-targets source is ignored.

    Open Targets is the canonical drug-disease evidence source; an
    arbitrary "treats" predicate from elsewhere shouldn't fire the
    deterministic leak signal.
    """
    edge = KGEdge(
        subject_id="A",
        predicate="treats",
        object_id="B",
        evidence_source="manual",
    )
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
    )
    assert signal == "no_signal"


# ---------------------------------------------------------------------------
# Layer 1 deterministic veto
# ---------------------------------------------------------------------------


def test_layer_1_high_drives_drop():
    voter = EnsembleVoter()
    v = voter.vote("feat_x", layer_1_verdict=_layer_1_high())
    assert v.severity == "high"
    assert v.remediation == "drop"
    assert v.decided_by == "layer_1"
    assert v.final_role == "descendant"
    assert v.confidence == LAYER_1_CONFIDENCE


def test_layer_1_high_short_circuits_kg_and_llm():
    """Layer 1 wins even when KG and LLM exist with different opinions."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        layer_1_verdict=_layer_1_high(),
        kg_edges=[_kg_treats_edge()],
        feature_entity_ids=["CHEMBL:DRUG_X"],
        target_entity_ids=["EFO:0000270"],
        llm_verdict=_llm_verdict(role="ancestor", remediation="keep_with_caveat"),
    )
    assert v.decided_by == "layer_1"
    assert "layer_1=high but llm=ancestor" in v.disagreements


def test_layer_1_records_disagreement_when_adversarial_info():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        layer_1_verdict=_layer_1_high(),
        adversarial_verdict=_adversarial("info", z_score=1.0),
    )
    assert "layer_1=high but adversarial=info" in v.disagreements


def test_layer_1_no_disagreement_when_adversarial_high_too():
    """Both deterministic vetoes agreeing → no disagreement."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        layer_1_verdict=_layer_1_high(),
        adversarial_verdict=_adversarial("high", z_score=8.0),
    )
    assert v.decided_by == "layer_1"
    assert v.disagreements == ()


def test_layer_1_kg_signal_recorded_but_not_disagreement():
    """KG signals never count as disagreement against deterministic vetoes."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        layer_1_verdict=_layer_1_high(),
        kg_edges=[_kg_treats_edge()],
        feature_entity_ids=["CHEMBL:DRUG_X"],
        target_entity_ids=["EFO:0000270"],
    )
    assert v.decided_by == "layer_1"
    assert v.kg_signal == "leak_drug_treats_disease"
    assert v.disagreements == ()


# ---------------------------------------------------------------------------
# Adversarial deterministic veto
# ---------------------------------------------------------------------------


def test_adversarial_high_drives_drop():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("high", z_score=7.5),
    )
    assert v.severity == "high"
    assert v.remediation == "drop"
    assert v.decided_by == "adversarial"
    assert v.final_role == "descendant"
    assert v.confidence == ADVERSARIAL_HIGH_CONFIDENCE


def test_adversarial_high_records_disagreement_with_accept_llm():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("high"),
        llm_verdict=_llm_verdict(role="confounder", remediation="keep_with_caveat"),
    )
    assert v.decided_by == "adversarial"
    assert any("adversarial=high but llm=confounder" in d for d in v.disagreements)


def test_adversarial_high_no_disagreement_with_leak_llm():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("high"),
        llm_verdict=_llm_verdict(role="descendant", remediation="drop"),
    )
    assert v.decided_by == "adversarial"
    assert v.disagreements == ()


# ---------------------------------------------------------------------------
# LLM path: KG agrees / disagrees
# ---------------------------------------------------------------------------


def test_llm_descendant_with_kg_corroboration_high_confidence():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[_kg_treats_edge()],
        feature_entity_ids=["CHEMBL:DRUG_X"],
        target_entity_ids=["EFO:0000270"],
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    assert v.severity == "high"
    assert v.remediation == "drop"
    assert v.final_role == "descendant"
    assert v.kg_signal == "leak_drug_treats_disease"
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE + LLM_KG_CORROBORATION_BONUS)


def test_llm_kg_contradiction_abstains():
    """KG says leak, LLM says ancestor — abstain."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[_kg_treats_edge()],
        feature_entity_ids=["CHEMBL:DRUG_X"],
        target_entity_ids=["EFO:0000270"],
        llm_verdict=_llm_verdict(role="ancestor", remediation="keep_with_caveat"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "abstain"
    assert v.severity == "abstain"
    assert v.remediation == "review"
    assert v.final_role is None
    assert v.confidence == 0.0
    assert any("contradicts" in d.lower() or "disagrees" in d.lower() for d in v.disagreements)


def test_llm_kg_contradiction_other_direction():
    """KG taxonomic descendant + LLM confounder also contradicts."""
    edge = _kg_isa_edge(child_id="C_FEAT", parent_id="C_TGT")
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[edge],
        feature_entity_ids=["C_FEAT"],
        target_entity_ids=["C_TGT"],
        llm_verdict=_llm_verdict(role="confounder", remediation="keep_with_caveat"),
    )
    assert v.decided_by == "abstain"


def test_llm_no_kg_no_change_to_base():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE)


def test_llm_kg_self_contradictory_does_not_force_abstain():
    """KG with both treats+isa edges + LLM verdict → LLM still drives.

    Self-contradictory KG is recorded in evidence but doesn't itself
    contradict the LLM (the LLM can arbitrate).
    """
    treats = _kg_treats_edge(drug_id="A", disease_id="B")
    isa = _kg_isa_edge(child_id="A", parent_id="B")
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[treats, isa],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    assert v.kg_signal == "contradictory"


# ---------------------------------------------------------------------------
# LLM path: citation modulation
# ---------------------------------------------------------------------------


def test_llm_no_citations_penalty():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="descendant", cited_pmids=()),
    )
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE - LLM_NO_CITATION_PENALTY)


def test_llm_all_citations_failed_penalty():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[
            _unverified_citation(reason="missing_cue"),
            _unverified_citation(pmid="2", reason="unresolved"),
        ],
    )
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE - LLM_CITATION_FAIL_PENALTY)


def test_llm_partial_citation_verification_no_penalty():
    """Even one verified citation is enough to skip the fail penalty."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[
            _verified_citation(),
            _unverified_citation(pmid="2", reason="missing_cue"),
        ],
    )
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE)


def test_llm_cited_pmids_but_no_verdicts_no_penalty():
    """LLM cited PMIDs but caller didn't supply CitationVerdicts.

    This is an integration boundary case: the voter records the gap in
    evidence but doesn't penalize confidence (the caller decided not to
    check, that's not the LLM's fault).
    """
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="descendant", cited_pmids=("123",)),
        citation_verdicts=[],
    )
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE)
    assert any("no CitationVerdicts" in e for e in v.evidence)


# ---------------------------------------------------------------------------
# LLM path: adversarial moderate
# ---------------------------------------------------------------------------


def test_llm_accept_role_with_adversarial_moderate_penalty():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("moderate", z_score=4.0),
        llm_verdict=_llm_verdict(role="confounder", remediation="keep_with_caveat"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE - LLM_ADVERSARIAL_MODERATE_PENALTY)
    assert any("adversarial=moderate" in d for d in v.disagreements)


def test_llm_leak_role_with_adversarial_moderate_no_penalty():
    """Adversarial moderate corroborates LLM-leak; no penalty."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("moderate", z_score=4.0),
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    # No KG corroboration, no penalty (leak role accepts moderate)
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE)


def test_llm_confidence_clamped_to_unit_interval():
    """Stacked penalties cannot push confidence below 0."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("moderate", z_score=4.0),
        llm_verdict=_llm_verdict(role="confounder", cited_pmids=()),
        citation_verdicts=[],  # 0 cited + 0 verdicts = no_citation_penalty
    )
    # Base 0.85 - 0.15 (no citations) - 0.15 (adversarial moderate vs accept) = 0.55
    assert 0.0 <= v.confidence <= 1.0
    assert v.confidence == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# LLM remediation override
# ---------------------------------------------------------------------------


def test_llm_remediation_override_kept_when_valid():
    """LLM said `window` for a mediator; voter should respect it."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="mediator", remediation="window"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.remediation == "window"


def test_llm_remediation_override_rejected_when_inconsistent():
    """LLM said `keep_with_caveat` for a descendant; voter overrides to drop."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="descendant", remediation="keep_with_caveat"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.remediation == "drop"


def test_llm_role_to_remediation_defaults():
    """Each role has a sensible default remediation when the LLM
    suggests a remediation that isn't valid for that role."""
    voter = EnsembleVoter()
    cases = {
        "mediator": "window",
        "descendant": "drop",
        "collider": "drop",
        "ancestor": "keep_with_caveat",
        "confounder": "keep_with_caveat",
        "instrument": "keep_with_caveat",
    }
    # `review` is never valid for any role's remediation — it's
    # specifically reserved for `severity=abstain`. So it always
    # forces the default to fire.
    for role, expected in cases.items():
        v = voter.vote(
            "feat_x",
            llm_verdict=LLMVerdict(
                causal_role=role,  # type: ignore[arg-type]
                mechanism="m",
                recommended_remediation="review",
                cited_pmids=(),
            ),
        )
        assert v.remediation == expected, f"role={role}"


# ---------------------------------------------------------------------------
# KG-only path
# ---------------------------------------------------------------------------


def test_kg_only_treats_edge_drives_drop():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[_kg_treats_edge()],
        feature_entity_ids=["CHEMBL:DRUG_X"],
        target_entity_ids=["EFO:0000270"],
    )
    assert v.severity == "high"
    assert v.remediation == "drop"
    assert v.decided_by == "kg"
    assert v.final_role == "descendant"
    assert v.confidence == KG_ONLY_CONFIDENCE


def test_kg_only_taxonomic_drives_drop():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[_kg_isa_edge(child_id="C1", parent_id="C2")],
        feature_entity_ids=["C1"],
        target_entity_ids=["C2"],
    )
    assert v.decided_by == "kg"
    assert v.severity == "high"
    assert v.kg_signal == "taxonomic_descendant"


def test_kg_only_contradictory_no_llm_abstains():
    treats = _kg_treats_edge(drug_id="A", disease_id="B")
    isa = _kg_isa_edge(child_id="A", parent_id="B")
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[treats, isa],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
    )
    assert v.decided_by == "abstain"
    assert v.severity == "abstain"
    assert v.kg_signal == "contradictory"


def test_kg_no_signal_no_other_layers_abstains():
    voter = EnsembleVoter()
    v = voter.vote("feat_x")
    assert v.decided_by == "abstain"
    assert v.confidence == 0.0


# ---------------------------------------------------------------------------
# Adversarial moderate alone
# ---------------------------------------------------------------------------


def test_adversarial_moderate_alone_returns_moderate():
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("moderate", z_score=3.5),
    )
    assert v.severity == "moderate"
    assert v.remediation == "review"
    assert v.decided_by == "adversarial"
    assert v.final_role is None
    assert v.confidence == ADVERSARIAL_MODERATE_CONFIDENCE


def test_adversarial_info_alone_abstains():
    """`info` adversarial verdict has no signal worth a vote."""
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict=_adversarial("info", z_score=1.0),
    )
    assert v.decided_by == "abstain"


# ---------------------------------------------------------------------------
# EnsembleVerdict invariants
# ---------------------------------------------------------------------------


def test_verdict_is_frozen_dataclass():
    voter = EnsembleVoter()
    v = voter.vote("feat_x")
    with pytest.raises(Exception):  # noqa: PT011 - dataclass FrozenInstanceError
        v.severity = "high"  # type: ignore[misc]


def test_verdict_carries_all_inputs_in_audit_trail():
    """Audit trail preserves the upstream verdict CONTENTS.

    Per codex M5 (2026-05-08), `layer_1_input` and `adversarial_input`
    are now shallow `dict(...)` snapshots — equal in content but not
    identity. `llm_input` is a frozen dataclass and is preserved by
    identity.
    """
    voter = EnsembleVoter()
    layer_1 = _layer_1_high()
    adv = _adversarial("info", z_score=1.0)
    llm = _llm_verdict(role="ancestor", remediation="keep_with_caveat")
    v = voter.vote(
        "feat_x",
        layer_1_verdict=layer_1,
        adversarial_verdict=adv,
        llm_verdict=llm,
    )
    assert v.layer_1_input == layer_1
    assert v.adversarial_input == adv
    assert v.llm_input is llm  # frozen dataclass: identity preserved


def test_verdict_layer_1_input_isolated_from_caller_mutation():
    """Codex review MEDIUM (M5, 2026-05-08): post-vote mutation of the
    caller-owned dict must not corrupt the frozen verdict's audit trail.
    """
    voter = EnsembleVoter()
    layer_1 = _layer_1_high()
    v = voter.vote("feat_x", layer_1_verdict=layer_1)
    # Caller mutates their dict after voting
    layer_1["severity"] = "MUTATED"
    layer_1["new_key"] = "injected"
    # Verdict's snapshot is unaffected
    assert v.layer_1_input is not None
    assert v.layer_1_input["severity"] == "high"
    assert "new_key" not in v.layer_1_input


def test_verdict_adversarial_input_isolated_from_caller_mutation():
    """Same isolation guarantee for adversarial input dict."""
    voter = EnsembleVoter()
    adv = _adversarial("high", z_score=8.0)
    v = voter.vote("feat_x", adversarial_verdict=adv)
    adv["z_score"] = 0.0
    adv["severity"] = "info"
    assert v.adversarial_input is not None
    assert v.adversarial_input["z_score"] == 8.0
    assert v.adversarial_input["severity"] == "high"


def test_verdict_evidence_always_populated():
    voter = EnsembleVoter()
    # Even an abstain verdict has at least one evidence line
    v = voter.vote("feat_x")
    assert len(v.evidence) >= 1


def test_verdict_kg_edges_filtered_to_relevant_subset():
    """`kg_edges_considered` carries only the edges that classified."""
    relevant = _kg_treats_edge(drug_id="A", disease_id="B")
    irrelevant = _kg_treats_edge(drug_id="C", disease_id="D")
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[relevant, irrelevant],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
    )
    assert v.kg_edges_considered == (relevant,)


def test_verdict_citations_partitioned_correctly():
    voter = EnsembleVoter()
    good = _verified_citation(pmid="1")
    bad1 = _unverified_citation(pmid="2", reason="missing_cue")
    bad2 = _unverified_citation(pmid="3", reason="unresolved")
    v = voter.vote(
        "feat_x",
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[good, bad1, bad2],
    )
    assert v.verified_citations == (good,)
    assert v.unverified_citations == (bad1, bad2)


def test_verdict_dataclass_typing_round_trip():
    """An EnsembleVerdict can be constructed standalone (audit replay)."""
    v = EnsembleVerdict(
        feature_name="f",
        severity="abstain",
        remediation="review",
        decided_by="abstain",
        confidence=0.0,
    )
    assert v.kg_signal == "no_signal"
    assert v.kg_edges_considered == ()
    assert v.verified_citations == ()


# ---------------------------------------------------------------------------
# Integration: realistic CSU "is_basophil_count_180d a leak?" scenarios
# ---------------------------------------------------------------------------


def test_realistic_csu_journey_duration_layer_1_high():
    """Mimics the famous `journey_duration_days` case caught by Layer 1."""
    voter = EnsembleVoter()
    v = voter.vote(
        "journey_duration_days",
        layer_1_verdict={
            "feature": "journey_duration_days",
            "layer": "1",
            "severity": "high",
            "remediation": "drop",
            "evidence": "knowable_at=post_index",
            "contract_source": "csu",
            "contract_window_days": None,
        },
    )
    assert v.decided_by == "layer_1"
    assert v.severity == "high"
    assert v.confidence == 1.0


def test_realistic_age_at_index_clean_ancestor():
    """Pre-index demographic; LLM says ancestor; no leak signals."""
    voter = EnsembleVoter()
    v = voter.vote(
        "age_at_index",
        llm_verdict=_llm_verdict(role="ancestor", remediation="keep_with_caveat"),
        adversarial_verdict=_adversarial("info", z_score=0.5),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    assert v.severity == "info"
    assert v.remediation == "keep_with_caveat"
    assert v.final_role == "ancestor"


def test_realistic_drug_count_kg_corroborates_llm():
    """`drug_X_count` with KG+LLM agreement on descendant."""
    voter = EnsembleVoter()
    v = voter.vote(
        "drug_X_count",
        kg_edges=[_kg_treats_edge(drug_id="DRUG_X_CUI", disease_id="CSU_CUI")],
        feature_entity_ids=["DRUG_X_CUI"],
        target_entity_ids=["CSU_CUI"],
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    assert v.severity == "high"
    assert v.kg_signal == "leak_drug_treats_disease"
    assert v.confidence == pytest.approx(LLM_BASE_CONFIDENCE + LLM_KG_CORROBORATION_BONUS)


def test_realistic_kg_disagrees_with_llm_abstains():
    """KG says drug treats disease; LLM says ancestor — abstain.

    This is the exact pattern the EnsembleVoter must abstain on per the
    plan's design contract.
    """
    voter = EnsembleVoter()
    v = voter.vote(
        "uncertain_feat",
        kg_edges=[_kg_treats_edge(drug_id="X", disease_id="Y")],
        feature_entity_ids=["X"],
        target_entity_ids=["Y"],
        llm_verdict=_llm_verdict(role="ancestor", remediation="keep_with_caveat"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "abstain"
    assert v.severity == "abstain"
    assert v.remediation == "review"


# ---------------------------------------------------------------------------
# Codex review HIGH (H1, 2026-05-08): invalid LLM role
# ---------------------------------------------------------------------------


def test_invalid_llm_role_does_not_crash_returns_abstain():
    """LLMVerdict.causal_role outside the 6-role vocabulary used to crash
    `_role_to_remediation` with KeyError (codex review H1).

    The voter should treat the verdict as if the LLM hadn't run and the
    audit trail should record what the LLM actually said.
    """
    voter = EnsembleVoter()
    bad = LLMVerdict(
        causal_role="unknown",  # type: ignore[arg-type]
        mechanism="bad classifier output",
        recommended_remediation="review",
        cited_pmids=(),
    )
    v = voter.vote("feat_x", llm_verdict=bad)
    assert v.decided_by == "abstain"
    assert v.severity == "abstain"
    assert v.confidence == 0.0
    # Audit trail still carries the original (invalid) LLM verdict
    assert v.llm_input is bad
    assert any("unknown" in e and "vocabulary" in e for e in v.evidence)


def test_invalid_llm_role_falls_through_to_kg_only():
    """When LLM is invalid AND a strong KG signal exists, the voter
    should fall through to the KG-only path rather than crashing."""
    voter = EnsembleVoter()
    bad = LLMVerdict(
        causal_role="garbage_role",  # type: ignore[arg-type]
        mechanism="m",
        recommended_remediation="review",
    )
    v = voter.vote(
        "feat_x",
        kg_edges=[_kg_treats_edge(drug_id="A", disease_id="B")],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
        llm_verdict=bad,
    )
    assert v.decided_by == "kg"
    assert v.severity == "high"


def test_malformed_layer_1_high_without_contract_source_downgrades(caplog):
    """Codex review MEDIUM (M4, 2026-05-08): a Layer 1 verdict with
    severity=high but contract_source=None used to drive a confidence=1.0
    deterministic veto with no manifest provenance — making a malformed
    verdict indistinguishable from a verified contract veto in the audit
    trail.

    The fix downgrades it to "no signal" so the voter falls through to
    LLM/KG/abstain.
    """
    voter = EnsembleVoter()
    bad = {
        "feature": "feat_x",
        "layer": "1",
        "severity": "high",
        "contract_source": None,
        "contract_window_days": None,
    }
    import logging as _logging  # noqa: PLC0415

    with caplog.at_level(_logging.WARNING):
        v = voter.vote("feat_x", layer_1_verdict=bad)
    assert v.decided_by != "layer_1"
    assert v.decided_by == "abstain"
    assert v.layer_1_input == bad
    assert any("missing or empty" in e or "cannot honour" in e for e in v.evidence)
    assert any("malformed Layer 1" in rec.message for rec in caplog.records)


def test_malformed_layer_1_high_empty_string_contract_source_downgrades():
    """An empty-string `contract_source` is also malformed input."""
    voter = EnsembleVoter()
    bad = {"severity": "high", "contract_source": ""}
    v = voter.vote("feat_x", layer_1_verdict=bad)
    assert v.decided_by == "abstain"


def test_malformed_layer_1_high_missing_key_downgrades():
    """Missing `contract_source` key entirely → downgrade."""
    voter = EnsembleVoter()
    bad = {"severity": "high"}
    v = voter.vote("feat_x", layer_1_verdict=bad)
    assert v.decided_by == "abstain"


def test_well_formed_layer_1_high_still_drives_drop():
    """Sanity: a well-formed Layer 1 high verdict still vetoes."""
    voter = EnsembleVoter()
    good = {
        "severity": "high",
        "contract_source": "csu",
        "contract_window_days": None,
    }
    v = voter.vote("feat_x", layer_1_verdict=good)
    assert v.decided_by == "layer_1"
    assert v.severity == "high"
    assert v.confidence == LAYER_1_CONFIDENCE


def test_malformed_adversarial_high_without_z_score_downgrades(caplog):
    """Codex review MEDIUM (M3, 2026-05-08): adversarial verdict with
    severity=high but no z_score used to silently drive a confidence=0.95
    deterministic veto with `z_score=None` in the audit record. That
    breaks audit integrity (a high-confidence drop with no evidence).

    The fix downgrades the malformed verdict to "no signal" so the voter
    falls through to LLM/KG/abstain.
    """
    voter = EnsembleVoter()
    bad = {
        "feature": "feat_x",
        "layer": "3",
        "severity": "high",
        # z_score deliberately missing
    }
    import logging as _logging  # noqa: PLC0415

    with caplog.at_level(_logging.WARNING):
        v = voter.vote("feat_x", adversarial_verdict=bad)
    # The adversarial high veto should NOT have fired (z_score missing)
    assert v.decided_by != "adversarial"
    assert v.decided_by == "abstain"
    # But the malformed input is still preserved in the audit trail
    assert v.adversarial_input == bad
    # Evidence names the malformed-veto downgrade
    assert any("missing or non-finite" in e or "cannot honour" in e for e in v.evidence)
    # Operator-visible warning logged
    assert any("malformed adversarial" in rec.message for rec in caplog.records)


def test_malformed_adversarial_high_nan_z_score_also_downgrades():
    """NaN z_score is also non-finite and should not honour the veto."""
    voter = EnsembleVoter()
    bad = {
        "severity": "high",
        "z_score": float("nan"),
    }
    v = voter.vote("feat_x", adversarial_verdict=bad)
    assert v.decided_by == "abstain"


def test_malformed_adversarial_high_string_z_score_downgrades():
    """A string in `z_score` is malformed input — downgrade."""
    voter = EnsembleVoter()
    bad = {
        "severity": "high",
        "z_score": "not_a_number",
    }
    v = voter.vote("feat_x", adversarial_verdict=bad)
    assert v.decided_by == "abstain"


def test_malformed_adversarial_high_bool_z_score_downgrades():
    """`z_score=True` is the bool-as-int Python footgun; reject."""
    voter = EnsembleVoter()
    bad = {"severity": "high", "z_score": True}
    v = voter.vote("feat_x", adversarial_verdict=bad)
    assert v.decided_by == "abstain"


def test_self_loop_edge_does_not_count_as_relation():
    """Codex review MEDIUM (M2, 2026-05-08): a self-loop edge
    (subject_id == object_id) used to drive a `taxonomic_descendant`
    KG signal whenever the same CUI was both feature and target. The
    edge encodes "X is_a X", not "feature is descendant of target".
    """
    self_loop = KGEdge(
        subject_id="C1",
        predicate="isa",
        object_id="C1",
        evidence_source="umls_relations",
    )
    signal, edges = classify_kg_signal(
        [self_loop],
        feature_entity_ids=["C1"],
        target_entity_ids=["C1"],
    )
    assert signal == "no_signal"
    assert edges == ()


def test_self_loop_treats_edge_also_excluded():
    """Same defensive check for treats-style self-loops."""
    edge = KGEdge(
        subject_id="C1",
        predicate="treats",
        object_id="C1",
        evidence_source="open_targets",
    )
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=["C1"],
        target_entity_ids=["C1"],
    )
    assert signal == "no_signal"


def test_overlapping_feature_target_sets_with_real_edge_still_works():
    """Defensive case: when feature/target sets share a CUI (C1) but
    the actual edge is between two distinct CUIs (C2 → C3), the edge
    should still classify normally."""
    edge = _kg_isa_edge(child_id="C2", parent_id="C3")
    signal, _ = classify_kg_signal(
        [edge],
        feature_entity_ids=["C1", "C2"],
        target_entity_ids=["C1", "C3"],
    )
    assert signal == "taxonomic_descendant"


def test_contradictory_kg_with_accept_llm_abstains():
    """Codex review MEDIUM (M1, 2026-05-08): contradictory KG (mixed
    treats + taxonomic edges) + LLM=ancestor used to silently trust
    the LLM. The leak side of the contradictory pair was ignored.

    With the fix, contradictory KG + accept-role LLM abstains.
    """
    treats = _kg_treats_edge(drug_id="A", disease_id="B")
    isa = _kg_isa_edge(child_id="A", parent_id="B")
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[treats, isa],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
        llm_verdict=_llm_verdict(role="ancestor", remediation="keep_with_caveat"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "abstain"
    assert v.severity == "abstain"
    assert any("contradictory" in d.lower() or "disagrees" in d.lower() for d in v.disagreements)


def test_contradictory_kg_with_leak_llm_still_decides_via_llm():
    """Contradictory KG + LLM=descendant → LLM agrees there IS a leak,
    so the leak edges in the contradictory set are corroborated. Decide
    via LLM (test_llm_kg_self_contradictory_does_not_force_abstain
    above covers this; this test pins the explicit M1 scenario)."""
    treats = _kg_treats_edge(drug_id="A", disease_id="B")
    isa = _kg_isa_edge(child_id="A", parent_id="B")
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[treats, isa],
        feature_entity_ids=["A"],
        target_entity_ids=["B"],
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "llm"
    assert v.severity == "high"


def test_invalid_llm_role_does_not_block_layer_1_veto():
    """Layer 1 deterministic veto wins even with an invalid LLM verdict."""
    voter = EnsembleVoter()
    bad = LLMVerdict(
        causal_role="not_a_role",  # type: ignore[arg-type]
        mechanism="m",
        recommended_remediation="review",
    )
    v = voter.vote(
        "feat_x",
        layer_1_verdict=_layer_1_high(),
        llm_verdict=bad,
    )
    assert v.decided_by == "layer_1"
    # Disagreements should not include the invalid LLM (sanitised away)
    assert all("not_a_role" not in d for d in v.disagreements)
    # But the audit trail still records what the LLM said
    assert v.llm_input is bad


# ---------------------------------------------------------------------------
# Plan v4 Phase 1 — LLM demoted to audit-only in the voter (generalize #212).
# By default the LLM verdict NEVER decides severity/remediation; it is recorded
# for the audit trail and the decision falls through to the deterministic rules
# (KG-only / adversarial-moderate→review / abstain). The legacy decides path is
# preserved behind ADAPTIVE_LAYER4_LLM_DECIDES=1 for the ramp/back-compat.
# ---------------------------------------------------------------------------


def test_llm_audit_only_by_default_does_not_decide(monkeypatch):
    """With LLM-decides OFF (default), an LLM leak verdict that would have driven
    decided_by='llm' (severity=high/drop) is NOT used; the decision falls
    through to the deterministic adversarial-moderate review."""
    monkeypatch.delenv("ADAPTIVE_LAYER4_LLM_DECIDES", raising=False)
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict={"severity": "moderate", "z_score": 4.0},
        llm_verdict=_llm_verdict(role="descendant", remediation="drop"),
    )
    assert v.decided_by != "llm"
    assert v.decided_by == "adversarial"
    assert v.severity == "moderate"
    assert v.remediation == "review"
    # the LLM verdict is still recorded for the audit trail
    assert v.llm_input is not None
    assert any("audit" in e.lower() for e in v.evidence)


def test_llm_decides_when_flag_enabled(monkeypatch):
    """ADAPTIVE_LAYER4_LLM_DECIDES=1 restores the legacy decides path (ramp /
    back-compat): the LLM leak role drives decided_by='llm', severity=high."""
    monkeypatch.setenv("ADAPTIVE_LAYER4_LLM_DECIDES", "1")
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        adversarial_verdict={"severity": "moderate", "z_score": 4.0},
        llm_verdict=_llm_verdict(role="descendant", remediation="drop"),
    )
    assert v.decided_by == "llm"
    assert v.severity == "high"
    assert v.remediation == "drop"
    assert v.final_role == "descendant"


def test_llm_audit_only_kg_corroborated_leak_still_caught_by_kg(monkeypatch):
    """Audit-only does not weaken leak detection: a KG-corroborated leak the LLM
    would have dropped is still dropped — by the deterministic KG-only rule."""
    monkeypatch.delenv("ADAPTIVE_LAYER4_LLM_DECIDES", raising=False)
    voter = EnsembleVoter()
    v = voter.vote(
        "feat_x",
        kg_edges=[_kg_treats_edge()],
        feature_entity_ids=["CHEMBL:DRUG_X"],
        target_entity_ids=["EFO:0000270"],
        llm_verdict=_llm_verdict(role="descendant"),
        citation_verdicts=[_verified_citation()],
    )
    assert v.decided_by == "kg"  # NOT "llm"
    assert v.severity == "high"
    assert v.remediation == "drop"
