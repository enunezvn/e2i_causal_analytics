"""Phase 6 deterministic leakage-remediation tests.

Pre-Phase-6, leakage_remediation.py used an LLM call (`_analyze_leakage_with_llm`)
to decide which HIGH-severity features to drop. The LLM had hallucination
incidents (e.g., synthetic_v2_scenario_a_swap_close_20260507.md) where it
proposed feature names from training-corpus prior rather than the actual data.

Post-Phase-6, `_deterministic_pre_drop` handles ALL HIGH and CRITICAL findings
deterministically. The LLM is only consulted for MODERATE-severity findings
where genuine ambiguity exists.
"""

from __future__ import annotations

from src.agents.ml_foundation.data_preparer.nodes.leakage_remediation import (
    _deterministic_pre_drop,
)


def _make_finding(
    feature: str, check: str, severity: str, evidence: dict | None = None
) -> dict:
    return {
        "feature": feature,
        "check_name": check,
        "severity": severity,
        "evidence": evidence or {},
    }


def test_deterministic_drop_catches_high_single_feature_auc():
    """Pre-Phase-6: HIGH single_feature_auc was sent to LLM. Post-Phase-6: dropped."""
    context = {
        "leakage_findings": [
            _make_finding(
                "journey_duration_days",
                "single_feature_auc",
                "high",
                {"auc": 0.689, "n_samples": 7019},
            )
        ]
    }
    dropped, classifications = _deterministic_pre_drop(context)
    assert "journey_duration_days" in dropped
    assert "deterministic" in classifications["journey_duration_days"]
    assert "0.689" in classifications["journey_duration_days"]


def test_deterministic_drop_catches_high_target_correlation():
    """Pre-Phase-6: HIGH target_correlation went to LLM. Post-Phase-6: dropped."""
    context = {
        "leakage_findings": [
            _make_finding(
                "leaky_score",
                "target_correlation",
                "high",
                {"correlation": 0.78, "p_value": 1e-10},
            )
        ]
    }
    dropped, classifications = _deterministic_pre_drop(context)
    assert "leaky_score" in dropped
    assert "0.780" in classifications["leaky_score"]


def test_deterministic_drop_catches_critical_mi():
    """CRITICAL mutual_information always dropped (was already handled, regression test)."""
    context = {
        "leakage_findings": [
            _make_finding(
                "tautology",
                "mutual_information",
                "critical",
                {"mi_normalized": 0.85},
            )
        ]
    }
    dropped, classifications = _deterministic_pre_drop(context)
    assert "tautology" in dropped


def test_deterministic_skips_moderate_findings():
    """MODERATE severity is left for LLM (or downstream judgment)."""
    context = {
        "leakage_findings": [
            _make_finding(
                "marginal_feature",
                "single_feature_auc",
                "moderate",
                {"auc": 0.58},
            )
        ]
    }
    dropped, classifications = _deterministic_pre_drop(context)
    assert "marginal_feature" not in dropped
    assert dropped == []


def test_deterministic_skips_info_findings():
    """INFO severity is never auto-dropped."""
    context = {
        "leakage_findings": [
            _make_finding("benign", "single_feature_auc", "info", {"auc": 0.52})
        ]
    }
    dropped, _ = _deterministic_pre_drop(context)
    assert "benign" not in dropped


def test_deterministic_dedup_one_drop_per_feature():
    """A feature with multiple HIGH/CRITICAL findings is only dropped once."""
    context = {
        "leakage_findings": [
            _make_finding(
                "engagement_score", "single_feature_auc", "critical", {"auc": 0.99}
            ),
            _make_finding(
                "engagement_score",
                "perfect_class_separation",
                "critical",
                {"overlap": 0.0},
            ),
            _make_finding(
                "engagement_score", "logical_dependency", "high", {}
            ),
        ]
    }
    dropped, classifications = _deterministic_pre_drop(context)
    assert dropped.count("engagement_score") == 1
    # Most-severe finding (CRITICAL) should be the recorded reason.
    assert "critical" in classifications["engagement_score"]


def test_deterministic_handles_missing_feature_name():
    """Findings with empty feature names are silently skipped, not raised."""
    context = {
        "leakage_findings": [
            _make_finding("", "single_feature_auc", "high", {"auc": 0.85}),
            _make_finding(
                "real_feature", "single_feature_auc", "high", {"auc": 0.85}
            ),
        ]
    }
    dropped, _ = _deterministic_pre_drop(context)
    assert dropped == ["real_feature"]


def test_deterministic_no_findings_returns_empty():
    """No findings → no drops, no errors."""
    dropped, classifications = _deterministic_pre_drop({"leakage_findings": []})
    assert dropped == []
    assert classifications == {}
