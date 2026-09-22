"""Lane E item 2 — the feature-role panel reuses the node's four layers."""

from __future__ import annotations

import json
import os

import pytest

from src.causal_engine.feature_role_panel import (
    CAUSAL_ACTIVATION_PROFILE,
    FeatureRolePanel,
    FeatureRoleRecord,
    build_feature_role_panel_sync,
)

from .conftest import OUTCOME, TREATMENT, make_panel_frame

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def _frame():
    return make_panel_frame()


@pytest.fixture
def panel(_frame, stub_lm, reviewed_optum_attestations) -> FeatureRolePanel:
    # The optum attestations are re-signed ``human`` here (conftest): the
    # deciding path needs a reviewed attestation (Lane B, spec §7 / item 5).
    return build_feature_role_panel_sync(
        _frame, manifest_source="optum", treatment=TREATMENT, outcome=OUTCOME, seed=7
    )


def test_machine_attestations_are_audit_only_on_the_real_manifest(_frame, stub_lm) -> None:
    """Lane B (spec §7 / Lane B item 5): WITHOUT the re-signing, the optum
    manifest's attestations are ``provenance="machine"`` and the structural
    decider treats them as audit-only — the attested features are NOT decided
    structurally; Layer 4 fires and informs, the ensemble abstains for human
    review. Characterises the merged behaviour on the real manifest today."""
    real = build_feature_role_panel_sync(
        _frame, manifest_source="optum", treatment=TREATMENT, outcome=OUTCOME, seed=7
    )
    for name in ("age_at_index", "dx_total_csu"):
        rec = real.records[name]
        assert rec.ensemble["decided_by"] != "structural", (name, rec.ensemble)
    rec = real.records["age_at_index"]
    assert rec.ensemble["decided_by"] == "abstain", rec.ensemble
    assert rec.ensemble["final_role"] is None
    assert rec.layer_4["fired"] is True  # informs only; the LLM never decides


def test_profile_is_per_run_and_flips_no_global(panel: FeatureRolePanel) -> None:
    """The causal activation profile turns Layer 4 + the structural decider ON
    for THIS run only; the process env and the voter's decide-flag are untouched."""
    assert CAUSAL_ACTIVATION_PROFILE == {
        "adaptive_layer4_enabled": True,
        "adaptive_structural_decider_enabled": True,
        "kg_mode": "shadow",
    }
    assert panel.activation_profile == CAUSAL_ACTIVATION_PROFILE
    assert os.environ.get("ADAPTIVE_LAYER4_LLM_DECIDES") is None
    from src.data.kg.ensemble_voter import _llm_decides_enabled

    assert _llm_decides_enabled() is False


def test_panel_covers_every_covariate_and_neither_t_nor_y(panel: FeatureRolePanel, _frame) -> None:
    expected = tuple(c for c in _frame.columns if c not in (TREATMENT, OUTCOME))
    assert panel.features == expected
    assert set(panel.records) == set(expected)
    assert TREATMENT not in panel.records and OUTCOME not in panel.records
    assert (panel.manifest_source, panel.treatment, panel.outcome) == ("optum", TREATMENT, OUTCOME)
    assert panel.n_rows == len(_frame)


def test_layer_1_post_index_is_a_leak_verdict(panel: FeatureRolePanel) -> None:
    rec = panel.records["treatment_initiated"]
    assert rec.layer_1["verdict"] == "post_index"
    assert rec.layer_1["declared_safe"] is False
    assert rec.leak_verdict is True
    assert rec.leak_source == "layer_1_post_index"
    assert rec.ensemble["decided_by"] == "layer_1"
    assert rec.ensemble["final_role"] == "descendant"
    assert rec.ensemble["confidence"] == 1.0


def test_layer_3_high_without_a_contract_is_a_leak_verdict_needing_review(
    panel: FeatureRolePanel,
) -> None:
    """Spec 3(b) excludes a Layer-3 high from adjustment; codex r1: absence of a
    contract is not evidence of timing, so the record says ``unknown`` and
    ``review_required`` rather than presenting it as proven leakage."""
    rec = panel.records["leak_probe"]
    assert rec.layer_1["verdict"] == "no_contract"
    assert rec.layer_1["temporal_status"] == "unknown"
    assert rec.layer_3["ran"] is True
    assert rec.layer_3["z_score"] > 5.0
    assert rec.leak_verdict is True
    assert rec.leak_source == "layer_3_high"
    assert rec.review_required is True
    # Proven leakage (post-index contract) needs no review; a declared-safe
    # covariate is neither.
    assert panel.records["treatment_initiated"].layer_1["temporal_status"] == "post_index"
    assert panel.records["treatment_initiated"].review_required is False
    assert panel.records["age_at_index"].layer_1["temporal_status"] == "pre_index"
    assert panel.records["age_at_index"].review_required is False
    assert panel.layer_activity["ensemble"]["review_required"] == 1


def test_declared_safe_covariates_are_never_leak_verdicts(panel: FeatureRolePanel) -> None:
    """The contract is the temporal arbiter (declared-safe immunity): a pre-index
    covariate keeps its Layer-3 statistic as EVIDENCE but is not a leak. This is
    the property that keeps prediction-era leakage out of the causal adjustment
    set unless the feature is genuinely post-index or uncontracted-and-leaking."""
    for name in ("age_at_index", "dx_total_csu", "charlson_score"):
        rec = panel.records[name]
        assert rec.layer_1["verdict"] == "pre_index"
        assert rec.layer_1["declared_safe"] is True
        assert rec.leak_verdict is False, (name, rec.ensemble)
        assert rec.leak_source is None
    # Verifier MED-2: the property is only exercised by a declared-safe feature
    # whose Layer-3 z-band actually said ``high`` — ``charlson_score`` is a
    # near-copy of Y. It is NOT a leak, and the served field says the contract
    # overruled the statistic (MED-3: the field reads pre-joint high AND declared
    # safe AND not leaked, the lever that keeps it adjustable — not the node's
    # post-joint severity, which it never leaves at ``high`` here).
    high_safe = panel.records["charlson_score"]
    assert high_safe.layer_3["ran"] is True
    assert high_safe.layer_3["severity_pre_joint_check"] == "high", high_safe.layer_3
    assert high_safe.leak_verdict is False, high_safe.ensemble
    assert high_safe.layer_3["declared_safe_immunity_applied"] is True, high_safe.layer_3
    assert panel.layer_activity["layer_3"]["declared_safe_immunity_applied"] == 1
    # And the un-immune control: the uncontracted near-copy IS the leak verdict.
    assert panel.records["leak_probe"].layer_3["severity_pre_joint_check"] == "high"
    assert panel.records["leak_probe"].layer_3["declared_safe_immunity_applied"] is False
    assert "charlson_score" not in panel.leak_features()


def test_layer_2_reports_the_committed_signal_with_its_edges(panel: FeatureRolePanel) -> None:
    rec = panel.records["dx_total_csu"]
    assert rec.layer_2["mode"] == "shadow"
    assert rec.layer_2["signal"] == "leak_drug_treats_disease"
    assert rec.layer_2["edges"], "a signal with no supporting edges is not auditable"
    edge = rec.layer_2["edges"][0]
    assert edge["predicate"] == "treats"
    assert edge["source_subject_id"].startswith("CHEMBL")
    assert edge["source_object_id"]
    # Shadow: KG informs, it does not decide a drop.
    assert rec.ensemble["severity"] == "info"
    assert rec.leak_verdict is False
    # And a feature with no KG concept says so honestly.
    assert panel.records["noise_feature"].layer_2["signal"] == "no_signal"
    assert panel.records["noise_feature"].layer_2["edges"] == []


def test_attested_features_go_to_the_structural_decider_not_the_llm(
    panel: FeatureRolePanel,
) -> None:
    """The profile turns the structural decider ON: a feature with an authored
    ``causal_structure`` is decided by ``extract_role`` and Layer 4 is never
    called for it (the node's rule — the LLM is never the decider, nor even
    consulted, for an attested feature)."""
    for name in ("age_at_index", "dx_total_csu"):
        rec = panel.records[name]
        assert rec.ensemble["decided_by"] == "structural", (name, rec.ensemble)
        assert rec.ensemble["structural_role"] == "confounder"
        assert rec.ensemble["final_role"] == "confounder"
        assert rec.layer_4["fired"] is False
    # And the Layer-3 statistic is still evidence on the record.
    assert panel.records["age_at_index"].layer_3["severity_pre_joint_check"] == "moderate"


def test_layer_4_fires_under_the_profile_but_only_informs(panel: FeatureRolePanel, stub_lm) -> None:
    rec = panel.records["moderate_probe"]
    assert rec.layer_1["verdict"] == "no_contract"
    assert rec.layer_3["severity_pre_joint_check"] == "moderate", rec.layer_3
    assert rec.layer_4["fired"] is True
    assert rec.layer_4["role"] == "confounder"
    assert "stub mechanism" in rec.layer_4["mechanism"]
    assert rec.layer_4["remediation"] == "keep_with_caveat"
    # The stub cites nothing (a cited PMID would make the node call Europe PMC
    # from a unit test); the counts must still be present and consistent.
    assert rec.layer_4["cited_pmids"] == []
    assert rec.layer_4["citations"] == {
        "checked": 0,
        "verified": 0,
        "unverified": 0,
        "verified_ids": [],
        "verdicts": [],
    }
    # Audit-only: with the Layer-3 signal joint-clamped to info and the LLM not
    # allowed to decide, no precedence rule fires and the voter ABSTAINS — the
    # honest verdict ("route to a human"), which is what the causal design
    # wants: the voters inform, the author and the reviewer decide.
    assert rec.ensemble["decided_by"] == "abstain"
    assert rec.ensemble["severity"] == "abstain"
    assert rec.ensemble["final_role"] is None
    assert rec.leak_verdict is False
    assert stub_lm.history, "the DummyLM was never called — Layer 4 did not fire"
    fired = [f for f, r in panel.records.items() if r.layer_4["fired"]]
    assert fired == ["moderate_probe"], fired


def test_layer_4_does_not_fire_without_the_profile(_frame, stub_lm) -> None:
    """Same frame, profile with Layer 4 off: not one LLM call. The profile is the
    only thing turning the voter on, per run, with no global flag flip."""
    before = len(stub_lm.history)
    panel = build_feature_role_panel_sync(
        _frame,
        manifest_source="optum",
        treatment=TREATMENT,
        outcome=OUTCOME,
        activation_profile={**CAUSAL_ACTIVATION_PROFILE, "adaptive_layer4_enabled": False},
        seed=7,
    )
    assert len(stub_lm.history) == before
    assert not any(r.layer_4["fired"] for r in panel.records.values())
    assert panel.layer_activity["layer_4"]["fired"] == 0


def test_layer_activity_counts_and_abstain_rate(panel: FeatureRolePanel) -> None:
    la = panel.layer_activity
    # Seven covariates since verifier MED-2 added ``charlson_score`` (declared
    # safe, pre-joint high): contracted 4, scored 6, immunity applied 1.
    assert la["layer_1"]["consulted"] == 7
    assert la["layer_1"]["contracted"] == 4
    assert la["layer_1"]["post_index"] == 1
    assert la["layer_2"]["cache_bound"] is True
    assert la["layer_2"]["signalled"] == 1
    assert la["layer_3"]["scored"] == 6  # the post-index column never reaches Layer 3
    assert la["layer_3"]["declared_safe_immunity_applied"] == 1
    assert la["layer_4"]["enabled"] is True
    assert la["layer_4"]["classifier_loaded"] is True
    assert la["layer_4"]["fired"] == 1
    assert la["layer_4"]["roles"] == {"confounder": 1}
    # ``charlson_score`` is attested, but an FDR-confident ``high`` is decided
    # by the adversarial layer; the node's declared-safe strip then keeps it
    # out of the leak set (immunity applied 1 above), so it is adversarial
    # here, not structural.
    assert la["ensemble"]["decided_by"] == {
        "layer_1": 1,
        "structural": 2,
        "adversarial": 3,
        "abstain": 1,
    }
    assert la["ensemble"]["leak_verdicts"] == 2
    assert la["ensemble"]["leak_sources"] == {"layer_1_post_index": 1, "layer_3_high": 1}
    assert 0.0 <= la["ensemble"]["abstain_rate"] <= 1.0
    assert panel.leak_features() == ["leak_probe", "treatment_initiated"]
    assert panel.promotion_eligibility["n_patients"] == panel.n_rows
    assert panel.promotion_eligibility["passes"] is False  # shadow: nothing promotes here


def test_panel_round_trips_through_json(panel: FeatureRolePanel) -> None:
    payload = panel.to_dict()
    text = json.dumps(payload, sort_keys=True)  # must be JSON-serialisable
    again = FeatureRolePanel.from_dict(json.loads(text))
    assert again == panel
    assert isinstance(again.records["leak_probe"], FeatureRoleRecord)
    assert again.to_dict() == payload


def test_rejects_a_frame_missing_t_or_y(_frame) -> None:
    with pytest.raises(ValueError, match="treatment"):
        build_feature_role_panel_sync(
            _frame.drop(columns=[TREATMENT]),
            manifest_source="optum",
            treatment=TREATMENT,
            outcome=OUTCOME,
        )
    with pytest.raises(ValueError, match="outcome"):
        build_feature_role_panel_sync(
            _frame, manifest_source="optum", treatment=TREATMENT, outcome="nope"
        )


def test_rejects_an_unregistered_manifest_source(_frame) -> None:
    with pytest.raises(ValueError, match="manifest"):
        build_feature_role_panel_sync(
            _frame, manifest_source="not_a_manifest", treatment=TREATMENT, outcome=OUTCOME
        )


def test_explicit_covariates_restrict_the_panel(_frame, stub_lm) -> None:
    panel = build_feature_role_panel_sync(
        _frame,
        manifest_source="optum",
        treatment=TREATMENT,
        outcome=OUTCOME,
        covariates=["age_at_index", "noise_feature"],
        seed=7,
    )
    assert panel.features == ("age_at_index", "noise_feature")
    with pytest.raises(ValueError, match="covariate"):
        build_feature_role_panel_sync(
            _frame,
            manifest_source="optum",
            treatment=TREATMENT,
            outcome=OUTCOME,
            covariates=["missing"],
        )


def test_panel_carries_a_schema_version_and_validates_strictly(panel: FeatureRolePanel) -> None:
    """codex r3: the typed parse must be an identity/consistency check, not
    coercion. ``validate_strict`` enforces the invariants and names the first
    violation; the real panel passes it."""
    from src.causal_engine.feature_role_panel import PANEL_SCHEMA_VERSION

    payload = panel.to_dict()
    assert payload["schema_version"] == PANEL_SCHEMA_VERSION == "1"
    panel.validate_strict()  # must not raise
    FeatureRolePanel.from_dict(payload).validate_strict()

    def broken(**top):
        return FeatureRolePanel.from_dict({**payload, **top})

    with pytest.raises(ValueError, match="schema_version"):
        broken(schema_version="0").validate_strict()
    with pytest.raises(ValueError, match="features"):
        broken(features=payload["features"][:-1]).validate_strict()
    with pytest.raises(ValueError, match="n_rows"):
        broken(n_rows=0).validate_strict()
    with pytest.raises(ValueError, match="manifest"):
        broken(manifest_source="nope").validate_strict()
    rec = dict(payload["records"]["age_at_index"])
    with pytest.raises(ValueError, match="feature"):
        broken(
            records={**payload["records"], "age_at_index": {**rec, "feature": "other"}}
        ).validate_strict()
    with pytest.raises(ValueError, match="leak_source"):
        broken(
            records={**payload["records"], "age_at_index": {**rec, "leak_verdict": True}}
        ).validate_strict()
    with pytest.raises(ValueError, match="leak_source"):
        broken(
            records={
                **payload["records"],
                "age_at_index": {**rec, "leak_verdict": True, "leak_source": "made_up"},
            }
        ).validate_strict()
    with pytest.raises(ValueError, match="post_index"):
        broken(
            records={
                **payload["records"],
                "age_at_index": {**rec, "leak_verdict": True, "leak_source": "layer_1_post_index"},
            }
        ).validate_strict()
    with pytest.raises(ValueError, match="layer_3"):
        broken(
            records={
                **payload["records"],
                "noise_feature": {
                    **payload["records"]["noise_feature"],
                    "leak_verdict": True,
                    "leak_source": "layer_3_high",
                    "layer_3": {**payload["records"]["noise_feature"]["layer_3"], "ran": False},
                },
            }
        ).validate_strict()
    with pytest.raises(ValueError, match="leak_verdict"):
        broken(
            records={**payload["records"], "age_at_index": {**rec, "leak_verdict": "yes"}}
        ).validate_strict()


def test_kg_decided_features_get_indication_evidence_not_a_causal_descendant(stub_lm) -> None:
    """codex r3: the voter's KG rule speaks prediction-era vocabulary — a drug
    approved for a pre-index condition is a "leak" there and yields
    final_role="descendant". In the causal contrast that edge is INDICATION
    evidence (a confounder candidate). The panel keeps the raw predictive verdict
    under its own name and withholds the causal role for the author/reviewer."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 400
    y = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "treatment_dupixent": rng.integers(0, 2, n),
            "persistent_at_180d_g28": y,
            # unattested mart flag with a committed KG `treats` edge from both drugs
            "cci_chronic_pulmonary": (rng.random(n) < 0.3).astype(int),
            "age_at_index": 40 + 10 * rng.standard_normal(n),
        }
    )
    panel = build_feature_role_panel_sync(
        frame,
        manifest_source="optum_mart",
        treatment="treatment_dupixent",
        outcome="persistent_at_180d_g28",
        seed=3,
    )
    rec = panel.records["cci_chronic_pulmonary"]
    assert rec.layer_2["signal"] == "leak_drug_treats_disease"
    assert rec.layer_2["causal_interpretation"] == "indication_evidence"
    assert rec.ensemble["decided_by"] == "kg"
    assert rec.ensemble["kg_predictive_role"] == "descendant"
    assert rec.ensemble["final_role"] is None
    assert "indication" in rec.ensemble["final_role_note"]
    assert rec.leak_verdict is False
    # A feature the KG has nothing to say about is untouched.
    assert panel.records["age_at_index"].layer_2["causal_interpretation"] is None
    assert "kg_predictive_role" in panel.records["age_at_index"].ensemble
    assert panel.records["age_at_index"].ensemble["kg_predictive_role"] is None


def test_raw_payload_validation_is_strict_not_coercive(panel: FeatureRolePanel) -> None:
    """codex r4: validate the RAW payload before any coercion — required explicit
    schema_version, exact types, no unknown fields — and enforce the invariants
    in BOTH directions."""
    from src.causal_engine.feature_role_panel import validate_panel_payload

    payload = panel.to_dict()
    validate_panel_payload(payload)  # the real panel passes

    def broken(**top):
        return {**payload, **top}

    def broken_record(name: str, **fields):
        recs = {**payload["records"], name: {**payload["records"][name], **fields}}
        return broken(records=recs)

    with pytest.raises(ValueError, match="schema_version"):
        validate_panel_payload({k: v for k, v in payload.items() if k != "schema_version"})
    with pytest.raises(ValueError, match="unknown"):
        validate_panel_payload(broken(smuggled=1))
    with pytest.raises(ValueError, match="n_rows"):
        validate_panel_payload(broken(n_rows="15209"))
    with pytest.raises(ValueError, match="features"):
        validate_panel_payload(broken(features="age_at_index"))
    with pytest.raises(ValueError, match="leak_verdict"):
        validate_panel_payload(broken_record("age_at_index", leak_verdict="yes"))
    with pytest.raises(ValueError, match="unknown"):
        validate_panel_payload(broken_record("age_at_index", extra_field=1))
    # A post-index Layer-1 verdict MUST be a leak verdict (keeping it is the
    # dangerous direction).
    with pytest.raises(ValueError, match="post_index"):
        validate_panel_payload(
            broken_record(
                "treatment_initiated", leak_verdict=False, leak_source=None, review_required=False
            )
        )
    # Layer-3-high needs an UNCONTRACTED column, Layer 3 having run, and the
    # review flag; a declared pre-index feature cannot be presented as one.
    with pytest.raises(ValueError, match="no_contract"):
        validate_panel_payload(
            broken_record(
                "age_at_index", leak_verdict=True, leak_source="layer_3_high", review_required=True
            )
        )
    with pytest.raises(ValueError, match="review_required"):
        validate_panel_payload(broken_record("leak_probe", review_required=False))
    with pytest.raises(ValueError, match="review_required"):
        validate_panel_payload(broken_record("noise_feature", review_required=True))


def test_kg_withheld_role_note_is_signal_specific() -> None:
    """codex r4: only a `leak_drug_treats_disease` signal on a declared pre-index
    covariate is indication evidence; another KG signal gets its own explanation."""
    from src.causal_engine.feature_role_panel import _ensemble_from_verdict

    v = {
        "decided_by": "kg",
        "final_role": "descendant",
        "severity": "info",
        "kg_signal": "taxonomic_descendant",
    }
    e = _ensemble_from_verdict(v, kg_causal_interpretation=None, kg_signal="taxonomic_descendant")
    assert e["final_role"] is None and e["kg_predictive_role"] == "descendant"
    assert "taxonomic_descendant" in e["final_role_note"] and "approved" not in e["final_role_note"]
    e2 = _ensemble_from_verdict(
        {**v, "kg_signal": "leak_drug_treats_disease"},
        kg_causal_interpretation="indication_evidence",
        kg_signal="leak_drug_treats_disease",
    )
    assert "indication" in e2["final_role_note"]
    e3 = _ensemble_from_verdict(
        {"decided_by": "adversarial", "final_role": None},
        kg_causal_interpretation=None,
        kg_signal="no_signal",
    )
    assert e3["kg_predictive_role"] is None and e3["final_role_note"] is None
