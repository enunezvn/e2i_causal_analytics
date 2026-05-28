"""EnsembleVoter structural precedence rule (Plan v4 Layer B / Phase 2, Task 2).

The deterministic authored causal role (derived from a feature's
``CausalStructureAttestation`` via ``extract_role``) is passed into the voter as
``structural_role`` / ``structural_unclassifiable`` and decides the
moderate/ambiguous band — replacing the unreliable LLM for attested features.
The rule sits BELOW the empirical statistical vetoes (Layer-1 high, adversarial
high) so an empirical leak still wins, and ABOVE the KG/LLM block so the LLM
never decides for an attested feature.
"""

from src.data.kg.ensemble_voter import EnsembleVoter


def test_structural_role_decides_benign_keep():
    v = EnsembleVoter().vote("age", structural_role="confounder")
    assert v.decided_by == "structural" and v.final_role == "confounder"
    assert v.severity == "info" and v.remediation in {"keep_with_caveat", "keep"}


def test_structural_role_decides_leak_drop():
    v = EnsembleVoter().vote("post_index_event", structural_role="collider")
    assert v.decided_by == "structural" and v.final_role == "collider"
    assert v.severity == "high" and v.remediation == "drop"


def test_structural_unclassifiable_routes_to_review():
    v = EnsembleVoter().vote("bad_attest", structural_unclassifiable=True)
    assert v.decided_by == "structural" and v.severity == "moderate"
    assert v.remediation == "review" and v.final_role is None


def test_structural_role_loses_to_adversarial_high_veto():
    # An empirically-confirmed leak (adversarial severity=high with a finite z)
    # MUST win over a benign authored role — the structural rule sits below the
    # empirical-high veto by design.
    adv = {"severity": "high", "z_score": 9.0, "_hblp_classified": True}
    v = EnsembleVoter().vote(
        "leaky_confounder", adversarial_verdict=adv, structural_role="confounder"
    )
    assert v.decided_by == "adversarial" and v.severity == "high"


def test_no_structural_input_falls_through_unchanged():
    # New kwargs default to None/False → existing paths are byte-identical.
    v = EnsembleVoter().vote("x")
    assert v.decided_by == "abstain"
