"""Unit tests for the HTE strategic insight (grounding + fail-closed guard).

NOTE: CI's unit job does NOT run tests/insights/ (known blind spot) — run these
scoped locally when touching src/insights/hte.py.
"""

from types import SimpleNamespace

import pytest

from src.insights import hte


def _record(**overrides):
    base = {
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "brand": "Remibrutinib",
        "confidence_level": 0.95,
        "overall_ate": 0.1106,
        "heterogeneity_score": 0.26,
        "expected_lift_pp": 0.0,
        "optimal_allocation_summary": "No reliable differential-targeting opportunity.",
        "cate_by_segment": {
            "disease_severity_band": [
                {
                    "segment_value": "high",
                    "cate_estimate": 0.1772,
                    "cate_ci_lower": 0.1267,
                    "cate_ci_upper": 0.2277,
                    "sample_size": 1385,
                    "statistical_significance": True,
                },
                {
                    "segment_value": "low",
                    "cate_estimate": 0.0338,
                    "cate_ci_lower": -0.0280,
                    "cate_ci_upper": 0.0955,
                    "sample_size": 2498,
                    "statistical_significance": False,
                },
            ],
            "age_band": [
                {
                    "segment_value": "50-65",
                    "cate_estimate": 0.1375,
                    "cate_ci_lower": 0.0796,
                    "cate_ci_upper": 0.1955,
                    "sample_size": 2015,
                    "statistical_significance": True,
                },
            ],
        },
    }
    base.update(overrides)
    return base


class TestBuildGrounding:
    def test_scope_and_effect_summary_render_real_figures(self):
        g = hte.build_grounding(_record())
        assert "treatment_arm -> persistent_180d" in g["scope"]
        assert "Remibrutinib" in g["scope"]
        assert "+11.1pp" in g["effect_summary"]
        assert "2 of 3 segments" in g["effect_summary"]
        assert g["sig_count"] == 2 and g["total_count"] == 3

    def test_cohort_n_is_per_dimension_sum_not_all_rows(self):
        # severity dim: 1385+2498=3883; age dim: 2015 -> cohort = max = 3883,
        # NOT 1385+2498+2015=5898 (dimensions partition the same cohort).
        g = hte.build_grounding(_record())
        assert "n=3,883" in g["scope"]
        assert "5,898" not in g["scope"]

    def test_segments_sorted_desc_with_pp_and_significance(self):
        g = hte.build_grounding(_record())
        lines = g["segments"].splitlines()
        assert lines[0].startswith("disease_severity_band=high: +17.7pp")
        assert "not significant" in lines[-1]
        assert "n=1,385" in lines[0]

    def test_segment_value_numerals_are_vouched(self):
        # Age band "50-65" must not trip the numeric guard.
        g = hte.build_grounding(_record())
        assert "50" in g["vouched"] and "65" in g["vouched"]

    def test_outcome_variable_digits_are_vouched(self):
        g = hte.build_grounding(_record())
        assert "180" in g["vouched"]

    def test_no_signal_when_no_cate_rows(self):
        g = hte.build_grounding(_record(cate_by_segment={}, overall_ate=None))
        assert g["has_signal"] is False


class TestGuard:
    def test_faithful_output_passes(self):
        g = hte.build_grounding(_record())
        text = (
            "The overall effect is +11.1pp and 2 of 3 segments clear zero. "
            "High severity responds strongest at +17.7pp [CI +12.7pp to +22.8pp] "
            "(n=1,385), while the low band (+3.4pp) is not distinguishable from "
            "no effect. Expected lift from differential targeting is +0.0pp."
        )
        assert hte._is_grounded(text, g) is True

    def test_fabricated_pp_rejected(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The high band gains +24.9pp of persistence.", g) is False

    def test_rederived_delta_rejected(self):
        # 17.7 - 3.4 = 14.3pp spread is a RE-DERIVED figure, never rendered.
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The spread between bands is 14.3pp.", g) is False

    def test_fabricated_count_fraction_rejected(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Fully 3 of 3 segments are significant.", g) is False

    def test_true_count_fraction_passes(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("2 of 3 segments are significant.", g) is True

    def test_comma_formatted_n_passes(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("The strongest segment holds 1,385 patients.", g) is True

    def test_unvouched_integer_rejected(self):
        g = hte.build_grounding(_record())
        assert hte._is_grounded("Roll out to 500 more HCPs.", g) is False


class TestGenerateInsight:
    def test_no_signal_returns_honest_empty_fallback(self):
        g = hte.build_grounding(_record(cate_by_segment={}, overall_ate=None))
        out = hte.generate_insight(g)
        assert out["is_fallback"] is True
        assert "no per-segment CATE results" in out["insight"]

    def test_lm_unavailable_returns_factual_fallback(self, monkeypatch):
        monkeypatch.setattr("src.insights.hte.run_signature", lambda *a, **k: None)
        g = hte.build_grounding(_record())
        out = hte.generate_insight(g)
        assert out["is_fallback"] is True
        assert "+11.1pp" in out["insight"]
        assert "Factual summary" in out["insight"]

    def test_grounded_lm_output_served(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights.hte.run_signature",
            lambda *a, **k: SimpleNamespace(
                interpretation=(
                    "Treatment lifts persistence by +11.1pp overall; high severity "
                    "(+17.7pp, n=1,385) responds most."
                ),
                key_takeaways=["2 of 3 segments clear zero", "Uniform rollout is appropriate"],
            ),
        )
        out = hte.generate_insight(hte.build_grounding(_record()))
        assert out["is_fallback"] is False
        assert "+17.7pp" in out["insight"]

    def test_ungrounded_lm_output_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights.hte.run_signature",
            lambda *a, **k: SimpleNamespace(
                interpretation="Persistence improves 42.5pp in responders.",
                key_takeaways=[],
            ),
        )
        out = hte.generate_insight(hte.build_grounding(_record()))
        assert out["is_fallback"] is True
        assert "42.5" not in out["insight"]

    def test_ungrounded_takeaway_also_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights.hte.run_signature",
            lambda *a, **k: SimpleNamespace(
                interpretation="Overall ATE is +11.1pp.",
                key_takeaways=["Target the 900 highest-value HCPs"],
            ),
        )
        out = hte.generate_insight(hte.build_grounding(_record()))
        assert out["is_fallback"] is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.1106, "+11.1pp"), (-0.028, "-2.8pp"), (None, None), (float("nan"), None)],
)
def test_pp_formatting(value, expected):
    assert hte._pp(value) == expected
