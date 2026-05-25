"""Tests for src/data/causal_role_classifier_ensemble.py (Issue #242).

Multi-model worker ensemble: Sonnet 4.6 + Opus 4.7 + GPT-5, agreement-or-escalate.
The bulk of these tests drive the PURE fusion logic with hand-built
EnsembleModelVote tuples — no LM, no API, fully deterministic (red-first).
"""

from __future__ import annotations

from src.data.kg.types import EnsembleModelVote


def _vote(model, role, *, remediation=None, mechanism="", cost=None, latency=None, error=None):
    return EnsembleModelVote(
        model=model,
        causal_role=role,
        mechanism=mechanism,
        recommended_remediation=remediation,
        cost_usd=cost,
        latency_ms=latency,
        error=error,
    )


# ---------------------------------------------------------------------------
# P1 — _fuse_votes agreement-or-escalate logic
# ---------------------------------------------------------------------------


def test_fuse_full_agreement_3_of_3():
    """3/3 healthy models agree → FULL, fused role = that role."""
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote(
            "anthropic/claude-sonnet-4-6",
            "confounder",
            remediation="keep_with_caveat",
            mechanism="pre-index comorbidity",
        ),
        _vote("anthropic/claude-opus-4-7", "confounder", remediation="keep_with_caveat"),
        _vote("openai/gpt-5", "confounder", remediation="keep_with_caveat"),
    )
    clf = _fuse_votes("comorbidity_count_preindex", votes)
    assert clf.agreement == "full"
    assert clf.fused_role == "confounder"
    assert clf.healthy_votes == 3
    assert clf.fused_remediation == "keep_with_caveat"
    assert clf.fused_mechanism == "pre-index comorbidity"  # carried from a majority vote


def test_fuse_majority_2_of_3():
    """2/3 agree, 1 dissents → MAJORITY, fused = majority role; dissenter retained in votes."""
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "ancestor"),  # the dissenter (wrong)
        _vote("anthropic/claude-opus-4-7", "descendant"),
        _vote("openai/gpt-5", "descendant"),
    )
    clf = _fuse_votes("post_index_event_count", votes)
    assert clf.agreement == "majority"
    assert clf.fused_role == "descendant"
    assert clf.healthy_votes == 3
    # the dissenting model's vote is preserved for the split audit
    assert any(v.causal_role == "ancestor" for v in clf.votes)


def test_fuse_full_when_one_model_errored_other_two_agree():
    """Degrade-to-healthy: 1 model errors (non-vote), other 2 agree → FULL on 2."""
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "instrument"),
        _vote("anthropic/claude-opus-4-7", "instrument"),
        _vote("openai/gpt-5", None, error="timeout"),
    )
    clf = _fuse_votes("first_initiation_window", votes)
    assert clf.agreement == "full"
    assert clf.fused_role == "instrument"
    assert clf.healthy_votes == 2


def test_fuse_split_when_only_one_healthy_vote():
    """<=1 healthy vote → escalate (SPLIT), fused_role None."""
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "mediator"),
        _vote("anthropic/claude-opus-4-7", None, error="rate_limit"),
        _vote("openai/gpt-5", None, error="timeout"),
    )
    clf = _fuse_votes("ambiguous_feature", votes)
    assert clf.agreement == "split"
    assert clf.fused_role is None
    assert clf.healthy_votes == 1


def test_fuse_split_when_all_three_distinct():
    """All 3 disagree (1-1-1) → SPLIT, fused_role None."""
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "confounder"),
        _vote("anthropic/claude-opus-4-7", "mediator"),
        _vote("openai/gpt-5", "collider"),
    )
    clf = _fuse_votes("contested_feature", votes)
    assert clf.agreement == "split"
    assert clf.fused_role is None
    assert clf.healthy_votes == 3


def test_fuse_split_when_all_three_errored():
    """All 3 error → SPLIT, healthy_votes 0, fused_role None."""
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", None, error="timeout"),
        _vote("anthropic/claude-opus-4-7", None, error="timeout"),
        _vote("openai/gpt-5", None, error="auth"),
    )
    clf = _fuse_votes("any_feature", votes)
    assert clf.agreement == "split"
    assert clf.fused_role is None
    assert clf.healthy_votes == 0


def test_fuse_split_on_two_healthy_tie_one_errored():
    """1-1 tie among 2 healthy (3rd errored) → no majority → SPLIT (documented tie policy)."""
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "confounder"),
        _vote("anthropic/claude-opus-4-7", "mediator"),
        _vote("openai/gpt-5", None, error="timeout"),
    )
    clf = _fuse_votes("tied_feature", votes)
    assert clf.agreement == "split"
    assert clf.fused_role is None
    assert clf.healthy_votes == 2


def test_fuse_feature_name_and_votes_round_trip():
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "descendant"),
        _vote("anthropic/claude-opus-4-7", "descendant"),
        _vote("openai/gpt-5", "descendant"),
    )
    clf = _fuse_votes("feat_x", votes)
    assert clf.feature_name == "feat_x"
    assert clf.votes == votes
