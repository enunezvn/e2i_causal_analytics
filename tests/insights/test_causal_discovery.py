import pytest

from src.insights.causal_discovery import (
    DSPY_AVAILABLE,
    CausalDiscoveryInsightSignature,
    build_grounding,
    generate_insight,
)
from src.insights.clinical_context import format_clinical_positioning

EFFECTS = [
    {
        "treatment": "copay_card",
        "outcome": "adherence_180d",
        "ate": 0.043,
        "ate_ci_lower": 0.02,
        "ate_ci_upper": 0.066,
        "status": "proceed",
        "selected_estimator": "CausalForestDML",
    },
    {
        "treatment": "nurse_call",
        "outcome": "adherence_180d",
        "ate": 0.011,
        "ate_ci_lower": -0.01,
        "ate_ci_upper": 0.03,
        "status": "review",
        "selected_estimator": "LinearDML",
    },
]


def test_build_grounding_ranks_and_counts_gates():
    g = build_grounding("Kisqali", "patient", EFFECTS)
    assert "proceed" in g["gate_summary"] and "review" in g["gate_summary"]
    assert any(c["label"] == "Effects" and c["value"] == "2" for c in g["grounding"])


def test_generate_insight_fallback_grounded():
    g = build_grounding("Kisqali", "patient", EFFECTS)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "copay_card" in out["insight"]


# ---------------------------------------------------------------------------
# Registry context (2026-07-07 follow-up): commercial chains are OUTSIDE the
# leaderboard's estimation scope (the grain-scope guard excludes commercial
# nodes from runs), so the insight may cite them only as additional modeled
# coverage — digit-free, provenance-labeled, never as discovered effects.
# ---------------------------------------------------------------------------

DRIVERS = [
    {
        "start": "rep_detailing_frequency",
        "end": "trx_volume",
        "effect": 0.2977,
        "confidence": 0.87,
        "synthetic": True,
    },
]


def test_build_grounding_carries_digit_free_registry_context_and_chip():
    g = build_grounding("Kisqali", "patient", EFFECTS, causal_drivers=DRIVERS)
    assert "rep detailing frequency → TRx volume" in g["registry_context"]
    assert "curated synthetic" in g["registry_context"]
    assert not any(ch.isnumeric() for ch in g["registry_context"])
    assert any(c["label"] == "Registry chains" and c["value"] == "1" for c in g["grounding"])


def test_build_grounding_without_drivers_is_honest_and_chipless():
    g = build_grounding("Kisqali", "patient", EFFECTS)
    assert "no modeled causal drivers" in g["registry_context"].lower()
    assert not any(c["label"] == "Registry chains" for c in g["grounding"])


def test_fallback_appends_registry_line_only_when_present():
    g = build_grounding("Kisqali", "patient", EFFECTS, causal_drivers=DRIVERS)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert "Registry-modeled causal chains" in out["insight"]
    bare = generate_insight(build_grounding("Kisqali", "patient", EFFECTS))
    assert "Registry-modeled" not in bare["insight"]


# ---------------------------------------------------------------------------
# Clinical positioning gate (2026-07-23 frontend review): the brand's labeled
# target population + line of therapy grounds the interpretation so a modeled
# effect in a clinically off-target population is not recommended.
# ---------------------------------------------------------------------------


def test_format_clinical_positioning_is_brand_specific_and_fail_open():
    remi = format_clinical_positioning("Remibrutinib")
    assert "treatment-naive" in remi.lower()
    assert "antihistamine" in remi.lower()
    # HR+/HER2- positioning is Kisqali's, not Remibrutinib's.
    assert "her2" not in remi.lower()
    assert "her2-negative" in format_clinical_positioning("Kisqali").lower()
    # Fail-open: unknown / missing brand yields "".
    assert format_clinical_positioning("Nonexistent") == ""
    assert format_clinical_positioning(None) == ""


def test_build_grounding_carries_clinical_positioning_and_chip():
    positioning = format_clinical_positioning("Remibrutinib")
    g = build_grounding("Remibrutinib", "patient", EFFECTS, clinical_positioning=positioning)
    assert g["clinical_positioning"] == positioning
    assert any(
        c["label"] == "Clinical positioning" and c["value"] == "applied" for c in g["grounding"]
    )


def test_build_grounding_without_positioning_has_no_chip():
    g = build_grounding("Remibrutinib", "patient", EFFECTS)
    assert g["clinical_positioning"] == ""
    assert not any(c["label"] == "Clinical positioning" for c in g["grounding"])


def test_fallback_appends_positioning_only_when_present():
    positioning = format_clinical_positioning("Remibrutinib")
    out = generate_insight(
        build_grounding("Remibrutinib", "patient", EFFECTS, clinical_positioning=positioning)
    )
    assert out["is_fallback"] is True
    assert "Clinical positioning:" in out["insight"]
    assert "treatment-naive" in out["insight"].lower()
    bare = generate_insight(build_grounding("Remibrutinib", "patient", EFFECTS))
    assert "Clinical positioning:" not in bare["insight"]


@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
def test_signature_declares_positioning_and_gating_directive():
    # The clinical positioning is an explicit input field...
    assert "clinical_positioning" in CausalDiscoveryInsightSignature.input_fields
    # ...and the prompt directs the LM to GATE recommendations by it.
    doc = (CausalDiscoveryInsightSignature.__doc__ or "").lower()
    assert "clinical positioning" in doc
    assert "gate" in doc
    assert "off-target" in doc or "outside" in doc
