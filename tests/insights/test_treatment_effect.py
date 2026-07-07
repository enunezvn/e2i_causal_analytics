"""Treatment-effect insight: registry context wiring (2026-07-07 follow-up).

The registry context must stay CLEARLY SEPARATE from the estimate narrative:
digit-free (a curated effect size next to a fitted ATE reads as corroboration),
provenance-labeled, and honestly empty when the registry has nothing for the
estimated pair.

NOTE: CI's unit job does NOT run tests/insights/ — run scoped locally when
touching src/insights/treatment_effect.py.
"""

from src.insights.treatment_effect import _fallback, build_grounding, generate_insight

DRIVERS = [
    {
        "start": "rep_detailing_frequency",
        "end": "trx_volume",
        "effect": 0.2977,
        "confidence": 0.87,
        "synthetic": True,
    },
    {
        "start": "persistent_180d",
        "end": "trx_volume",
        "effect": 0.21,
        "confidence": 0.84,
        "synthetic": True,
    },
]


def _grounding(**overrides):
    kwargs = {
        "cohort": "persistence",
        "brand": "Remibrutinib",
        "treatment_var": "rep_detailing_frequency",
        "outcome_var": "trx_volume",
        "confounders": ["disease_severity", "academic_hcp"],
        "ate": 0.0412,
        "ci_lower": 0.012,
        "ci_upper": 0.071,
        "p_value": 0.004,
        "n": 3883,
        "estimator": "LinearDML",
    }
    kwargs.update(overrides)
    return build_grounding(**kwargs)


def test_build_grounding_carries_digit_free_registry_context_and_chip():
    g = _grounding(causal_drivers=DRIVERS)
    assert "rep detailing frequency → TRx volume" in g["registry_context"]
    assert "patient persistence → TRx volume" in g["registry_context"]
    assert "curated synthetic" in g["registry_context"]
    # Digit-free: a curated "+0.30" beside the fitted ATE would read as an
    # estimate — the whole reason this context is names-only.
    assert not any(ch.isnumeric() for ch in g["registry_context"])
    assert any(c["label"] == "Registry chains" and c["value"] == "2" for c in g["grounding"])


def test_build_grounding_without_drivers_is_honest_and_chipless():
    g = _grounding()
    assert "no modeled causal drivers" in g["registry_context"].lower()
    assert not any(c["label"] == "Registry chains" for c in g["grounding"])


def test_fallback_keeps_registry_context_separate_from_estimate():
    g = _grounding(causal_drivers=DRIVERS)
    out = _fallback(g)
    assert out["is_fallback"] is True
    # The estimate sentence must be intact and the registry line clearly its
    # own sentence, after the robustness caveat.
    assert "ATE +0.0412" in out["insight"]
    assert "Registry-modeled causal chains" in out["insight"]
    assert out["insight"].index("NOT validated") < out["insight"].index("Registry-modeled")


def test_fallback_without_drivers_has_no_registry_line():
    out = _fallback(_grounding())
    assert "Registry-modeled" not in out["insight"]


def test_generate_insight_fallback_carries_grounding_chips():
    g = _grounding(causal_drivers=DRIVERS)
    out = generate_insight(g)
    assert out["is_fallback"] is True
    assert any(c["label"] == "Registry chains" for c in out["grounding"])
