"""Renaming the per-HCP trigger counts must not change drift severity (codex r1 HIGH).

Intent investigation: the high-impact set is a generic feature-name heuristic
(3e1c70cf4 initial platform; reshaped in 463d54f55 "P2 shallow output
enhancement") with no issue or policy defining which drift is high impact. The
only live features that ever matched trx_count / nrx_count are the per-HCP trigger
counts, so the rename preserves their rating; re-ranking severity is an owner
decision, not a side effect of a column rename.

These assert the IMPACT CLASS, not the whole returned sentence. ``_assess_feature_impact``
interpolates ``{feature}`` into its return value, so two different feature names
can never produce equal strings — comparing full strings would be unsatisfiable
for the equality case and a tautology for the inequality case (measured
2026-09-17; both spellings are recorded in the lane report).
"""

import inspect

from src.agents.drift_monitor.nodes import alert_aggregator as mod

# Rated high impact today via a substring match, and used here as the reference
# class rather than as an expected string.
_REFERENCE_HIGH_IMPACT = "market_share"


def _node_and_method():
    cls = next(
        c for c in vars(mod).values() if inspect.isclass(c) and "_assess_feature_impact" in vars(c)
    )
    return object.__new__(cls), cls._assess_feature_impact


def _impact_class(assessment: str) -> str:
    """The rating prefix ('HIGH IMPACT' / 'SIGNIFICANT' / ...), free of the feature name."""
    return assessment.split(":", 1)[0].strip()


def test_the_renamed_trigger_features_keep_their_high_impact_rating():
    node, assess = _node_and_method()
    expected = _impact_class(assess(node, _REFERENCE_HIGH_IMPACT, "high", "data"))
    assert expected == "HIGH IMPACT"
    for name in ("triggers_delivered_count", "triggers_accepted_count"):
        assert _impact_class(assess(node, name, "high", "data")) == expected, name


def test_a_feature_outside_the_set_is_still_not_high_impact():
    node, assess = _node_and_method()
    assert _impact_class(assess(node, "call_frequency", "high", "data")) != "HIGH IMPACT"


def test_the_rename_does_not_promote_the_total_trigger_count():
    """No re-ranking: total_rx_count was never high impact, so triggers_total_count
    must not become high impact either. Renaming a column may not change severity."""
    node, assess = _node_and_method()
    assert _impact_class(assess(node, "triggers_total_count", "high", "data")) != "HIGH IMPACT"


def test_the_legacy_names_are_gone_from_the_set():
    source = inspect.getsource(mod)
    assert '"trx_count"' not in source and '"nrx_count"' not in source
