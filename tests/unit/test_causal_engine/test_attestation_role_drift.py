"""CI consistency/drift guard (Plan v4 Layer B / Phase 2, Task 4).

For every ``FeatureContract`` that declares BOTH a ``causal_role`` AND a
``causal_structure`` attestation, assert ``extract_role(edges) == declared
causal_role``. This catches an AUTHORING-CONSISTENCY drift the moment an
attestation lands — e.g. an author who declares ``causal_role="confounder"`` but
draws collider edges (a typo).

What this is and is NOT (codex iter-3 HIGH, mandated CLAUDE.md lens):
  * It is a drift/consistency guard — true-by-construction once edges are
    authored consistently.
  * It is NOT the functional acceptance of Phase 2 (that is the
    ``decided_by="structural"`` node+voter path in
    ``test_structural_decider_node.py`` / ``test_structural_decider_voter.py``).
  * It is NOT a correctness check on the roles themselves (that is the
    non-circular literature-precision test, plan Task 8). It only asserts the
    declared role and the authored edges AGREE — not that either is "right".

Today ZERO real contracts carry ``causal_structure`` (verified), so the
real-manifest iteration is empty and passes; the fixture-logic tests below keep
the guard NON-vacuous so it cannot silently rot into a no-op.
"""

from src.data.feature_contract import CausalStructureAttestation, FeatureContract, KnowableAt
from src.ml.causal_role_dgp.extractor import derive_structural_role


def _c(role, edges):
    return FeatureContract(
        name="V",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        causal_role=role,
        causal_structure=CausalStructureAttestation(
            treatment_node="T", outcome_node="Y", feature_node="V", edges=edges
        ),
    )


def test_drift_logic_passes_on_consistent_attestation():
    # Declared 'confounder' with confounder edges (V→T, V→Y) → no drift.
    role, err = derive_structural_role(_c("confounder", (("V", "T"), ("V", "Y"))))
    assert err is None and role == "confounder"


def test_drift_logic_catches_inconsistent_attestation():
    # Declared 'confounder' but the edges encode a collider (T→V←Y) → the derived
    # role disagrees with the declaration; this is exactly the drift the real-
    # manifest assertion below would fail on.
    role, err = derive_structural_role(_c("confounder", (("T", "V"), ("Y", "V"))))
    assert err is None and role == "collider"
    assert role != "confounder"  # the drift the CI guard catches


def _all_real_contracts():
    """Every FeatureContract registered across the three manifests.

    ``CSU_FEATURES`` / ``OPTUM_FEATURES`` are ``list[FeatureContract]``;
    ``SYNTHETIC_FEATURES`` is ``dict[str, FeatureContract]`` (resolved against
    ``src/data/manifests/__init__.py`` — there is no ``all_feature_contracts``
    accessor).
    """
    from src.data.manifests import CSU_FEATURES, OPTUM_FEATURES, SYNTHETIC_FEATURES

    return [*CSU_FEATURES, *OPTUM_FEATURES, *SYNTHETIC_FEATURES.values()]


def test_real_manifest_attestations_have_no_role_drift():
    drifted = []
    for c in _all_real_contracts():
        if getattr(c, "causal_structure", None) is None or getattr(c, "causal_role", None) is None:
            continue
        role, err = derive_structural_role(c)
        if err is not None or role != c.causal_role:
            drifted.append((c.name, c.causal_role, role, err))
    assert not drifted, f"attestation role drift: {drifted}"


def test_real_manifest_iteration_is_non_empty():
    # Guard against the accessor silently returning [] (which would make the
    # drift test vacuous for a DIFFERENT reason than "no attestations yet").
    assert len(_all_real_contracts()) > 100  # ~151 today (28 CSU + 122 Optum + 1 synthetic)
