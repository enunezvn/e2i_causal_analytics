"""Tests for the offline structural-vs-empirical crosscheck (Track-2B-v3 D4).

The crosscheck is the no-label production gate: for each attested feature it
pairs the *independently-derived* structural role (from authored edges) against
the *empirical* leakage severity (Phase-1 FDR / Layer-1 on the same cohort).
The safety-critical count is ``missed_leaks`` — a feature the structural role
puts in ACCEPT while the empirical signal calls it a leak. That must be 0.
"""

from src.data.structural_empirical_crosscheck import (
    LEAK_SEVERITIES,
    compare_structural_vs_empirical,
)


def test_missed_leak_when_structural_accept_but_empirical_high() -> None:
    # structural says keep (confounder ∈ ACCEPT) but empirical flags high → DANGER.
    res = compare_structural_vs_empirical(
        structural_roles={"f": "confounder"},
        empirical_severity={"f": "high"},
    )
    assert res.missed_leaks == ("f",)
    assert not res.gate_passed


def test_critical_severity_also_counts_as_a_leak() -> None:
    # critical >= high; must not slip past the gate.
    res = compare_structural_vs_empirical(
        structural_roles={"f": "ancestor"},
        empirical_severity={"f": "critical"},
    )
    assert res.missed_leaks == ("f",)
    assert "critical" in LEAK_SEVERITIES and "high" in LEAK_SEVERITIES


def test_structural_none_is_reviewed_not_a_decision() -> None:
    # Unclassifiable authored DAG → routed to review, NOT counted as a missed leak
    # (the decider abstains; the empirical gate still governs).
    res = compare_structural_vs_empirical(
        structural_roles={"f": None},
        empirical_severity={"f": "high"},
    )
    assert res.reviewed == ("f",)
    assert res.missed_leaks == ()
    assert res.agree == ()
    assert res.disagreements == ()


def test_agree_when_both_call_it_a_leak() -> None:
    res = compare_structural_vs_empirical(
        structural_roles={"f": "mediator"},  # LEAK_ROLE
        empirical_severity={"f": "high"},
    )
    assert res.agree == ("f",)
    assert res.gate_passed


def test_agree_when_both_accept() -> None:
    res = compare_structural_vs_empirical(
        structural_roles={"f": "instrument"},  # ACCEPT_ROLE
        empirical_severity={"f": "info"},
    )
    assert res.agree == ("f",)
    assert res.gate_passed


def test_structural_overflag_is_a_disagreement_not_a_missed_leak() -> None:
    # structural says leak (collider) but empirical is quiet → conservative/safe
    # direction: a disagreement worth reviewing, but NOT a missed leak.
    res = compare_structural_vs_empirical(
        structural_roles={"f": "collider"},
        empirical_severity={"f": "info"},
    )
    assert res.disagreements == ("f",)
    assert res.missed_leaks == ()
    assert res.gate_passed


def test_moderate_is_not_a_leak_severity() -> None:
    # The empirical-high veto fires on high (not moderate); structural ACCEPT +
    # empirical moderate is NOT a missed leak.
    res = compare_structural_vs_empirical(
        structural_roles={"f": "confounder"},
        empirical_severity={"f": "moderate"},
    )
    assert res.missed_leaks == ()
    assert res.agree == ("f",)


def test_missing_empirical_severity_defaults_to_no_signal() -> None:
    # An attested feature with no empirical entry → treated as not-flagged (accept
    # bucket); a structural ACCEPT there is agreement, never a manufactured leak.
    res = compare_structural_vs_empirical(
        structural_roles={"f": "ancestor"},
        empirical_severity={},
    )
    assert res.missed_leaks == ()
    assert res.agree == ("f",)


def test_mixed_cohort_partition_is_exhaustive_and_ordered() -> None:
    res = compare_structural_vs_empirical(
        structural_roles={
            "miss": "confounder",  # ACCEPT + high → missed_leak
            "ok_leak": "descendant",  # LEAK + high → agree
            "ok_keep": "ancestor",  # ACCEPT + info → agree
            "overflag": "mediator",  # LEAK + info → disagreement
            "unsure": None,  # None → reviewed
        },
        empirical_severity={
            "miss": "high",
            "ok_leak": "high",
            "ok_keep": "info",
            "overflag": "info",
            "unsure": "high",
        },
    )
    assert res.missed_leaks == ("miss",)
    assert res.disagreements == ("overflag",)
    assert res.reviewed == ("unsure",)
    assert set(res.agree) == {"ok_leak", "ok_keep"}
    # 4-way partition is exhaustive: every feature lands in exactly one bucket.
    total = len(res.agree) + len(res.disagreements) + len(res.missed_leaks) + len(res.reviewed)
    assert total == 5
    assert not res.gate_passed  # the missed leak fails the gate


def test_unknown_role_string_is_routed_to_review_defensively() -> None:
    # A role outside the 6-role taxonomy cannot be bucketed → review, not a guess.
    res = compare_structural_vs_empirical(
        structural_roles={"f": "not_a_real_role"},
        empirical_severity={"f": "high"},
    )
    assert res.reviewed == ("f",)
    assert res.missed_leaks == ()
