"""Track-2B-v3 Phase 2 — Optum-initiation structural attestations (offline structural validation).

Every pre-index (SAFE) Optum feature carries a `CausalStructureAttestation` whose
authored edges derive (via the deterministic `extract_role`) to an ACCEPT role
(ancestor/confounder/instrument). This is the no-label *structural-side* check
the activation plan calls for: on the dx-anchored, treatment-naive Optum cohort
every legitimate pre-index feature is a confounder/instrument of the
biologic-initiation decision — none is outcome-derived, so none is a leak. The
post-index FORBIDDEN columns are NOT model inputs and stay un-attested.

(T, Y) framing: T=`biologic_initiation` (the treatment decision), Y=`initiated_biologic_180d`.
Decider stays DARK — these attestations are inert until an explicit ramp.
"""

import dataclasses

from src.data.kg.ensemble_voter import ACCEPT_ROLES, LEAK_ROLES
from src.data.manifests.optum_feature_manifest import (
    OPTUM_FEATURES,
    optum_contract_for,
)
from src.ml.causal_role_dgp.extractor import derive_structural_role


def _safe_features():
    # Attestations are attached at the optum_contract_for accessor (the decider's
    # lookup path via lookup_feature_contract), keeping the OPTUM_FEATURES registry
    # statically traceable for the Layer-1 coverage guard. Validate the enriched
    # view the decider actually sees.
    return [
        optum_contract_for(c.name) for c in OPTUM_FEATURES if c.knowable_at.is_pre_or_at_index()
    ]


def _forbidden_features():
    return [c for c in OPTUM_FEATURES if not c.knowable_at.is_pre_or_at_index()]


def test_every_safe_feature_is_attested() -> None:
    missing = [c.name for c in _safe_features() if c.causal_structure is None]
    assert missing == [], f"SAFE Optum features missing causal_structure: {missing}"


def test_all_safe_attestations_derive_to_accept_and_are_classifiable() -> None:
    # Offline structural validation: every SAFE feature derives to an ACCEPT
    # role with no extractor error (0 unclassifiable).
    leaks: list[tuple[str, str]] = []
    unclassifiable: list[tuple[str, str]] = []
    for c in _safe_features():
        role, err = derive_structural_role(c)
        if err is not None or role is None:
            unclassifiable.append((c.name, err or "None role"))
            continue
        if role in LEAK_ROLES:
            leaks.append((c.name, role))
    assert unclassifiable == [], f"unclassifiable SAFE attestations: {unclassifiable}"
    assert leaks == [], f"SAFE features deriving to a LEAK role (unexpected): {leaks}"


def test_every_safe_role_is_in_accept_set() -> None:
    roles = {c.name: derive_structural_role(c)[0] for c in _safe_features()}
    bad = {n: r for n, r in roles.items() if r not in ACCEPT_ROLES}
    assert bad == {}, f"non-ACCEPT roles: {bad}"


def test_forbidden_post_index_features_are_not_attested() -> None:
    # Post-index targets/metadata are not model inputs → leave un-attested.
    attested_forbidden = [c.name for c in _forbidden_features() if c.causal_structure is not None]
    assert attested_forbidden == [], (
        f"forbidden features unexpectedly attested: {attested_forbidden}"
    )


def test_representative_roles_pin_the_two_patterns() -> None:
    # Pin the confounder (severity proxies) and instrument (access/geography/calendar) patterns.
    expected = {
        "age_at_index": "confounder",
        "primary_diagnosis_code": "confounder",
        "has_asthma": "confounder",
        "ige_total_result_last": "confounder",
        "sys_steroid_ever_filled": "confounder",
        "zip5": "instrument",
        "geographic_region": "instrument",
        "saw_allergist_flag": "instrument",
        "index_date": "instrument",
    }
    for name, want in expected.items():
        contract = optum_contract_for(name)
        assert contract is not None, f"missing contract {name}"
        role, err = derive_structural_role(contract)
        assert err is None, f"{name} unclassifiable: {err}"
        assert role == want, f"{name}: derived {role}, expected {want}"


def test_attestation_uses_consistent_treatment_outcome_labels() -> None:
    for c in _safe_features():
        att = c.causal_structure
        assert att is not None
        assert att.treatment_node == "biologic_initiation"
        assert att.outcome_node == "initiated_biologic_180d"
        assert att.feature_node == c.name


def test_attested_contracts_remain_frozen_and_replaceable() -> None:
    # Authoring uses dataclasses.replace on the frozen FeatureContract; sanity-check.
    c = optum_contract_for("age_at_index")
    assert c is not None
    again = dataclasses.replace(c)
    assert again == c
