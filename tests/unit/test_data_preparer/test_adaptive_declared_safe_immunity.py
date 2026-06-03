"""FIX 2 (full manifest immunity, user decision 2026-06-03): a feature the
cohort's FeatureContract declares pre-index (Layer-1 declared-safe) is the case
where the contract is the authoritative leakage arbiter — it is NEVER reported
as leaked, regardless of how strongly it predicts the target (legitimate signal)
or of sparsity artifacts (zero_variance / perfect_class_separation false-fire on
rare-event columns). Statistical checks remain the arbiter only for features
WITHOUT a contract.
"""

from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
    _declared_safe_immune_features,
    _severity_from_finding_dicts,
)


def test_immune_set_is_optum_declared_safe_only() -> None:
    immune = _declared_safe_immune_features(
        {"has_asthma", "charlson_score", "initiated_biologic_180d", "totally_unknown_xyz"},
        "optum",
    )
    # declared pre-index (knowable_at=index) -> immune
    assert "has_asthma" in immune
    assert "charlson_score" in immune
    # post-index target sibling -> NOT immune (still droppable)
    assert "initiated_biologic_180d" not in immune
    # no contract entry -> NOT immune (statistical layer governs it)
    assert "totally_unknown_xyz" not in immune


def test_immune_set_noop_without_manifest_source() -> None:
    # No cohort contract resolved -> opt-in safety: never grant immunity.
    assert _declared_safe_immune_features({"has_asthma"}, None) == set()


def test_severity_from_dicts_picks_highest() -> None:
    assert _severity_from_finding_dicts([{"severity": "moderate"}, {"severity": "high"}]) == "high"
    assert (
        _severity_from_finding_dicts([{"severity": "info"}, {"severity": "critical"}]) == "critical"
    )
    assert _severity_from_finding_dicts([{"severity": "info"}]) == "info"


def test_severity_from_dicts_empty_is_none() -> None:
    # The sanctioned downgrade: once immune findings are stripped and nothing
    # remains, severity is "none" -> routing skips remediation.
    assert _severity_from_finding_dicts([]) == "none"


def test_severity_from_dicts_handles_enum_severity() -> None:
    # detect_leakage findings may carry a LeakageSeverity enum rather than a str.
    from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import (
        LeakageSeverity,
    )

    assert _severity_from_finding_dicts([{"severity": LeakageSeverity.HIGH}]) == "high"


def test_remediation_companion_strips_declared_safe_from_drop_list() -> None:
    # Mirrors the leakage_remediation companion: declared-safe features the LLM
    # adds to its drop list are stripped; post-index / un-contracted features
    # remain droppable.
    drop = ["has_asthma", "charlson_score", "initiated_biologic_180d", "uncontracted_xyz"]
    immune = _declared_safe_immune_features(set(drop), "optum")
    kept = [f for f in drop if f not in immune]
    assert "has_asthma" not in kept
    assert "charlson_score" not in kept
    assert "initiated_biologic_180d" in kept  # post-index target sibling stays droppable
    assert "uncontracted_xyz" in kept  # no contract -> statistical layer governs
