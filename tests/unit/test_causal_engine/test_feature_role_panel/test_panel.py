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
def panel(_frame, stub_lm) -> FeatureRolePanel:
    return build_feature_role_panel_sync(
        _frame, manifest_source="optum", treatment=TREATMENT, outcome=OUTCOME, seed=7
    )


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


def test_layer_3_high_without_a_contract_is_a_leak_verdict(panel: FeatureRolePanel) -> None:
    rec = panel.records["leak_probe"]
    assert rec.layer_1["verdict"] == "no_contract"
    assert rec.layer_3["ran"] is True
    assert rec.layer_3["z_score"] > 5.0
    assert rec.leak_verdict is True
    assert rec.leak_source == "layer_3_high"


def test_declared_safe_covariates_are_never_leak_verdicts(panel: FeatureRolePanel) -> None:
    """The contract is the temporal arbiter (declared-safe immunity): a pre-index
    covariate keeps its Layer-3 statistic as EVIDENCE but is not a leak. This is
    the property that keeps prediction-era leakage out of the causal adjustment
    set unless the feature is genuinely post-index or uncontracted-and-leaking."""
    for name in ("age_at_index", "dx_total_csu"):
        rec = panel.records[name]
        assert rec.layer_1["verdict"] == "pre_index"
        assert rec.layer_1["declared_safe"] is True
        assert rec.leak_verdict is False, (name, rec.ensemble)
        assert rec.leak_source is None


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
    assert la["layer_1"]["consulted"] == 6
    assert la["layer_1"]["contracted"] == 3
    assert la["layer_1"]["post_index"] == 1
    assert la["layer_2"]["cache_bound"] is True
    assert la["layer_2"]["signalled"] == 1
    assert la["layer_3"]["scored"] == 5  # the post-index column never reaches Layer 3
    assert la["layer_4"]["enabled"] is True
    assert la["layer_4"]["classifier_loaded"] is True
    assert la["layer_4"]["fired"] == 1
    assert la["layer_4"]["roles"] == {"confounder": 1}
    assert la["ensemble"]["decided_by"]["layer_1"] == 1
    assert la["ensemble"]["decided_by"]["structural"] == 2
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
