"""Tests for src/data/causal_role_classifier_ensemble.py (Issue #242).

Multi-model worker ensemble: Sonnet 4.6 + Opus 4.7 + GPT-5, agreement-or-escalate.
The bulk of these tests drive the PURE fusion logic with hand-built
EnsembleModelVote tuples — no LM, no API, fully deterministic (red-first).
"""

from __future__ import annotations

import pytest

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


# ---------------------------------------------------------------------------
# P2 — per-provider cost constants + _cost_for + telemetry aggregation (#242 AC4)
# ---------------------------------------------------------------------------


def test_cost_for_sonnet_uses_sonnet_rates():
    from src.data.causal_role_classifier_ensemble import (
        SONNET_INPUT_USD_PER_MTOK,
        SONNET_OUTPUT_USD_PER_MTOK,
        _cost_for,
    )

    cost = _cost_for("anthropic/claude-sonnet-4-6", 1000, 200)
    expected = 1000 / 1e6 * SONNET_INPUT_USD_PER_MTOK + 200 / 1e6 * SONNET_OUTPUT_USD_PER_MTOK
    assert cost == pytest.approx(expected)


def test_cost_for_opus_uses_opus_rates():
    from src.data.causal_role_classifier_ensemble import (
        OPUS_INPUT_USD_PER_MTOK,
        OPUS_OUTPUT_USD_PER_MTOK,
        _cost_for,
    )

    cost = _cost_for("anthropic/claude-opus-4-7", 1000, 200)
    expected = 1000 / 1e6 * OPUS_INPUT_USD_PER_MTOK + 200 / 1e6 * OPUS_OUTPUT_USD_PER_MTOK
    assert cost == pytest.approx(expected)


def test_cost_for_gpt5_uses_gpt5_rates():
    from src.data.causal_role_classifier_ensemble import (
        GPT5_INPUT_USD_PER_MTOK,
        GPT5_OUTPUT_USD_PER_MTOK,
        _cost_for,
    )

    cost = _cost_for("openai/gpt-5", 1000, 200)
    expected = 1000 / 1e6 * GPT5_INPUT_USD_PER_MTOK + 200 / 1e6 * GPT5_OUTPUT_USD_PER_MTOK
    assert cost == pytest.approx(expected)


def test_cost_for_none_tokens_returns_none():
    from src.data.causal_role_classifier_ensemble import _cost_for

    assert _cost_for("openai/gpt-5", None, 200) is None
    assert _cost_for("openai/gpt-5", 1000, None) is None


def test_cost_for_unknown_model_returns_none():
    from src.data.causal_role_classifier_ensemble import _cost_for

    assert _cost_for("anthropic/claude-haiku-4-5", 1000, 200) is None


def test_aggregate_telemetry_sums_cost_and_takes_max_latency():
    from src.data.causal_role_classifier_ensemble import _aggregate_telemetry

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "confounder", cost=0.01, latency=500.0),
        _vote("anthropic/claude-opus-4-7", "confounder", cost=0.05, latency=900.0),
        _vote("openai/gpt-5", "confounder", cost=0.02, latency=700.0),
    )
    total_cost, max_latency = _aggregate_telemetry(votes)
    assert total_cost == pytest.approx(0.08)
    assert max_latency == 900.0


def test_aggregate_telemetry_all_none_returns_none():
    from src.data.causal_role_classifier_ensemble import _aggregate_telemetry

    votes = (
        _vote("anthropic/claude-sonnet-4-6", None, error="timeout"),
        _vote("anthropic/claude-opus-4-7", None, error="timeout"),
    )
    total_cost, max_latency = _aggregate_telemetry(votes)
    assert total_cost is None
    assert max_latency is None


def test_fuse_votes_populates_aggregate_telemetry():
    from src.data.causal_role_classifier_ensemble import _fuse_votes

    votes = (
        _vote("anthropic/claude-sonnet-4-6", "descendant", cost=0.01, latency=500.0),
        _vote("anthropic/claude-opus-4-7", "descendant", cost=0.05, latency=900.0),
        _vote("openai/gpt-5", "descendant", cost=0.02, latency=700.0),
    )
    clf = _fuse_votes("feat", votes)
    assert clf.total_cost_usd == pytest.approx(0.08)
    assert clf.max_latency_ms == 900.0


# ---------------------------------------------------------------------------
# P3 — _ensemble_to_llm_verdict adapter (#242 AC3: voter/#240-gate consume it)
# ---------------------------------------------------------------------------


def test_adapter_full_satisfied_true_no_missed():
    from src.data.causal_role_classifier_ensemble import (
        _ensemble_to_llm_verdict,
        _fuse_votes,
    )

    clf = _fuse_votes(
        "feat",
        (
            _vote(
                "anthropic/claude-sonnet-4-6",
                "confounder",
                remediation="keep_with_caveat",
                mechanism="pre-index",
            ),
            _vote("anthropic/claude-opus-4-7", "confounder", remediation="keep_with_caveat"),
            _vote("openai/gpt-5", "confounder", remediation="keep_with_caveat"),
        ),
    )
    verdict = _ensemble_to_llm_verdict(clf)
    assert verdict is not None
    assert verdict.causal_role == "confounder"
    assert verdict.recommended_remediation == "keep_with_caveat"
    assert verdict.evaluator_audit is not None
    assert verdict.evaluator_audit.satisfied is True
    assert verdict.evaluator_audit.missed_considerations == ()
    assert verdict.evaluator_audit.evaluator_model.startswith("ensemble:")


def test_adapter_majority_satisfied_false_lists_dissent():
    from src.data.causal_role_classifier_ensemble import (
        _ensemble_to_llm_verdict,
        _fuse_votes,
    )

    clf = _fuse_votes(
        "feat",
        (
            _vote("anthropic/claude-sonnet-4-6", "ancestor"),  # dissenter
            _vote("anthropic/claude-opus-4-7", "descendant"),
            _vote("openai/gpt-5", "descendant"),
        ),
    )
    verdict = _ensemble_to_llm_verdict(clf)
    assert verdict is not None
    assert verdict.causal_role == "descendant"
    assert verdict.evaluator_audit is not None
    assert verdict.evaluator_audit.satisfied is False
    # dissenting model + its role surfaced for the split audit
    missed = " ".join(verdict.evaluator_audit.missed_considerations)
    assert "ancestor" in missed
    # cap + length contract from LLMEvaluatorAudit
    assert len(verdict.evaluator_audit.missed_considerations) <= 5
    assert all(len(x) <= 80 for x in verdict.evaluator_audit.missed_considerations)


def test_adapter_split_returns_none():
    """Split = no confident verdict → adapter returns None so the voter abstains
    (escalate to review / 'unknown'), matching single-model classify_feature."""
    from src.data.causal_role_classifier_ensemble import (
        _ensemble_to_llm_verdict,
        _fuse_votes,
    )

    clf = _fuse_votes(
        "feat",
        (
            _vote("anthropic/claude-sonnet-4-6", "confounder"),
            _vote("anthropic/claude-opus-4-7", "mediator"),
            _vote("openai/gpt-5", "collider"),
        ),
    )
    assert _ensemble_to_llm_verdict(clf) is None


def test_adapter_carries_aggregate_telemetry():
    from src.data.causal_role_classifier_ensemble import (
        _ensemble_to_llm_verdict,
        _fuse_votes,
    )

    clf = _fuse_votes(
        "feat",
        (
            _vote("anthropic/claude-sonnet-4-6", "descendant", cost=0.01, latency=500.0),
            _vote("anthropic/claude-opus-4-7", "descendant", cost=0.05, latency=900.0),
            _vote("openai/gpt-5", "descendant", cost=0.02, latency=700.0),
        ),
    )
    verdict = _ensemble_to_llm_verdict(clf)
    assert verdict is not None and verdict.evaluator_audit is not None
    assert verdict.evaluator_audit.cost_usd == pytest.approx(0.08)
    assert verdict.evaluator_audit.latency_ms == 900.0


def test_adapter_evaluator_model_names_all_three_members():
    from src.data.causal_role_classifier_ensemble import (
        _ensemble_to_llm_verdict,
        _fuse_votes,
    )

    clf = _fuse_votes(
        "feat",
        (
            _vote("anthropic/claude-sonnet-4-6", "instrument"),
            _vote("anthropic/claude-opus-4-7", "instrument"),
            _vote("openai/gpt-5", "instrument"),
        ),
    )
    verdict = _ensemble_to_llm_verdict(clf)
    assert verdict is not None and verdict.evaluator_audit is not None
    model_str = verdict.evaluator_audit.evaluator_model
    assert "sonnet" in model_str and "opus" in model_str and "gpt-5" in model_str
