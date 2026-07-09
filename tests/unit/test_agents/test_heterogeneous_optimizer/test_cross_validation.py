"""Tests for EconML↔CausalML cross-library validation.

These state channels (library_agreement_score / validation_passed /
cross_library_validation) were scaffolded in B7.4 but never computed; the
/segment-analysis Library Validation card rendered the nulls as a fabricated
"0% / Failed". The compute function makes them real — these tests pin the
agreement semantics and the honest not-computed paths.
"""

import math

import pytest

from src.agents.heterogeneous_optimizer.cross_validation import (
    AGREEMENT_THRESHOLD,
    MIN_SEGMENTS_FOR_VALIDATION,
    compute_cross_library_validation,
    serialize_validation_for_llm,
)
from src.agents.heterogeneous_optimizer.nodes.uplift_analyzer import UpliftAnalyzerNode


def _cate(dim: str, value: str, estimate: float) -> dict:
    return {
        "segment_name": dim,
        "segment_value": value,
        "cate_estimate": estimate,
        "cate_ci_lower": estimate - 0.05,
        "cate_ci_upper": estimate + 0.05,
        "sample_size": 500,
        "statistical_significance": True,
    }


def _uplift(dim: str, value: str, score: float) -> dict:
    return {
        "segment_name": dim,
        "segment_value": value,
        "mean_uplift_score": score,
        "uplift_score_std": 0.02,
        "auuc": None,
        "qini_coefficient": None,
        "top_10_pct_lift": 0.1,
        "sample_size": 500,
    }


class TestComputeCrossLibraryValidation:
    def test_identical_ranking_and_direction_passes_with_full_agreement(self):
        cate = {
            "sev": [
                _cate("sev", "high", 0.35),
                _cate("sev", "med", 0.20),
                _cate("sev", "low", 0.09),
            ]
        }
        uplift = {
            "sev": [
                _uplift("sev", "high", 0.30),
                _uplift("sev", "med", 0.25),
                _uplift("sev", "low", 0.10),
            ]
        }

        out = compute_cross_library_validation(cate, uplift, "random_forest")

        assert out["library_agreement_score"] == pytest.approx(1.0)
        assert out["econml_causalml_agreement"] == pytest.approx(1.0)
        assert out["validation_passed"] is True
        detail = out["cross_library_validation"]
        assert detail["computed"] is True
        assert detail["n_segments_compared"] == 3
        assert detail["spearman_rho"] == pytest.approx(1.0)
        assert detail["sign_agreement"] == pytest.approx(1.0)
        assert detail["uplift_model"] == "random_forest"

    def test_reversed_ranking_fails(self):
        # Same direction everywhere but perfectly OPPOSITE ranking: the rank
        # component contributes 0 (rho clamped at 0), sign contributes 0.5.
        cate = {"sev": [_cate("sev", "a", 0.30), _cate("sev", "b", 0.20), _cate("sev", "c", 0.10)]}
        uplift = {
            "sev": [_uplift("sev", "a", 0.10), _uplift("sev", "b", 0.20), _uplift("sev", "c", 0.30)]
        }

        out = compute_cross_library_validation(cate, uplift, "random_forest")

        assert out["library_agreement_score"] == pytest.approx(0.5)
        assert out["validation_passed"] is False
        assert out["cross_library_validation"]["spearman_rho"] == pytest.approx(-1.0)

    def test_direction_disagreement_lowers_score(self):
        # One library says harmful where the other says beneficial.
        cate = {"sev": [_cate("sev", "a", 0.30), _cate("sev", "b", 0.20), _cate("sev", "c", -0.10)]}
        uplift = {
            "sev": [_uplift("sev", "a", 0.30), _uplift("sev", "b", 0.20), _uplift("sev", "c", 0.10)]
        }

        out = compute_cross_library_validation(cate, uplift, "random_forest")

        assert out["cross_library_validation"]["sign_agreement"] == pytest.approx(2 / 3)
        # rho is 1.0 (same ranking) so score = 0.5*(2/3) + 0.5*1.0
        assert out["library_agreement_score"] == pytest.approx(0.5 * (2 / 3) + 0.5)

    def test_too_few_pairs_is_honestly_not_computed(self):
        cate = {"sev": [_cate("sev", "a", 0.3), _cate("sev", "b", 0.2)]}
        uplift = {"sev": [_uplift("sev", "a", 0.3), _uplift("sev", "b", 0.2)]}
        assert MIN_SEGMENTS_FOR_VALIDATION > 2

        out = compute_cross_library_validation(cate, uplift, "random_forest")

        assert out["cross_library_validation"]["computed"] is False
        assert "2 segment(s)" in out["cross_library_validation"]["reason"]
        # No fabricated verdict on the not-computed path.
        assert "library_agreement_score" not in out
        assert "validation_passed" not in out

    def test_missing_uplift_is_honestly_not_computed(self):
        cate = {"sev": [_cate("sev", "a", 0.3)]}

        out = compute_cross_library_validation(cate, None, None)

        assert out["cross_library_validation"]["computed"] is False
        assert "uplift" in out["cross_library_validation"]["reason"]
        assert "validation_passed" not in out

    def test_constant_uplift_scores_fall_back_to_sign_agreement(self):
        # Zero variance on one side -> Spearman undefined (nan) -> sign-only.
        cate = {"sev": [_cate("sev", "a", 0.3), _cate("sev", "b", 0.2), _cate("sev", "c", 0.1)]}
        uplift = {
            "sev": [_uplift("sev", "a", 0.25), _uplift("sev", "b", 0.25), _uplift("sev", "c", 0.25)]
        }

        out = compute_cross_library_validation(cate, uplift, "random_forest")

        detail = out["cross_library_validation"]
        assert detail["spearman_rho"] is None
        assert "sign_agreement only" in detail["method"]
        assert out["library_agreement_score"] == pytest.approx(1.0)

    def test_non_finite_estimates_are_excluded_from_pairing(self):
        cate = {
            "sev": [
                _cate("sev", "a", 0.30),
                _cate("sev", "b", 0.20),
                _cate("sev", "c", 0.10),
                _cate("sev", "d", math.nan),
            ]
        }
        uplift = {
            "sev": [
                _uplift("sev", "a", 0.30),
                _uplift("sev", "b", 0.20),
                _uplift("sev", "c", 0.10),
                _uplift("sev", "d", 0.40),
            ]
        }

        out = compute_cross_library_validation(cate, uplift, "random_forest")

        assert out["cross_library_validation"]["n_segments_compared"] == 3
        assert out["validation_passed"] is True

    def test_pairs_matched_within_dimension_only(self):
        # A segment value present in CATE dim 'sev' must not pair with an
        # uplift score from dim 'age' even if the value strings collide.
        cate = {"sev": [_cate("sev", "x", 0.3), _cate("sev", "y", 0.2), _cate("sev", "z", 0.1)]}
        uplift = {
            "age": [_uplift("age", "x", 0.3), _uplift("age", "y", 0.2), _uplift("age", "z", 0.1)]
        }

        out = compute_cross_library_validation(cate, uplift, "random_forest")

        assert out["cross_library_validation"]["computed"] is False


class TestUpliftNodeCrossLibraryUpdate:
    def test_failed_validation_carries_a_warning(self):
        node = UpliftAnalyzerNode()
        state = {
            "cate_by_segment": {
                "sev": [_cate("sev", "a", 0.30), _cate("sev", "b", 0.20), _cate("sev", "c", 0.10)]
            }
        }
        uplift = {
            "sev": [_uplift("sev", "a", 0.10), _uplift("sev", "b", 0.20), _uplift("sev", "c", 0.30)]
        }

        update = node._cross_library_update(state, uplift, "random_forest")

        assert update["validation_passed"] is False
        assert any("Cross-library validation FAILED" in w for w in update["warnings"])

    def test_passed_validation_has_no_warning(self):
        node = UpliftAnalyzerNode()
        state = {
            "cate_by_segment": {
                "sev": [_cate("sev", "a", 0.30), _cate("sev", "b", 0.20), _cate("sev", "c", 0.10)]
            }
        }
        uplift = {
            "sev": [_uplift("sev", "a", 0.30), _uplift("sev", "b", 0.20), _uplift("sev", "c", 0.10)]
        }

        update = node._cross_library_update(state, uplift, "random_forest")

        assert update["validation_passed"] is True
        assert "warnings" not in update

    def test_validation_error_never_raises(self, monkeypatch):
        node = UpliftAnalyzerNode()
        monkeypatch.setattr(
            "src.agents.heterogeneous_optimizer.nodes.uplift_analyzer.compute_cross_library_validation",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        update = node._cross_library_update({"cate_by_segment": {}}, {}, "random_forest")

        assert update["cross_library_validation"]["computed"] is False
        assert "boom" in update["cross_library_validation"]["reason"]


class TestProfileGeneratorFeedsValidationToLLM:
    @pytest.mark.asyncio
    async def test_llm_receives_the_validation_verdict(self, monkeypatch):
        from src.agents.heterogeneous_optimizer.nodes.profile_generator import (
            ProfileGeneratorNode,
        )

        received = {}

        def _fake_runner(**kwargs):
            received.update(kwargs)
            return {
                "executive_summary": "s",
                "interpretation": "i",
                "key_insights": ["k"],
                "high_responder_description": "h",
                "low_responder_description": "l",
            }

        monkeypatch.setattr(
            "src.agents.heterogeneous_optimizer.dspy_integration.generate_cate_interpretation",
            _fake_runner,
        )

        state = {
            "overall_ate": 0.2,
            "heterogeneity_score": 0.2,
            "expected_lift_pp": 0.02,
            "optimal_allocation_summary": "summary",
            "feature_importance": {},
            "cate_by_segment": {"sev": [_cate("sev", "high", 0.3)]},
            "high_responders": [],
            "low_responders": [],
            "errors": [],
            "warnings": [],
            "status": "optimizing",
            "library_agreement_score": 0.42,
            "validation_passed": False,
            "cross_library_validation": {
                "computed": True,
                "n_segments_compared": 5,
                "spearman_rho": -0.2,
                "sign_agreement": 0.8,
            },
        }

        await ProfileGeneratorNode().execute(state)  # type: ignore[arg-type]

        text = received["cross_library_validation_text"]
        assert text.startswith("FAILED")
        assert "42%" in text


class TestSerializeValidationForLLM:
    def test_not_computed_serializes_reason(self):
        state = {
            "cross_library_validation": {"computed": False, "reason": "no uplift results"},
        }
        assert serialize_validation_for_llm(state) == "not computed (no uplift results)"

    def test_absent_state_is_not_computed(self):
        assert serialize_validation_for_llm({}) == "not computed"

    def test_passed_verdict_serializes_components(self):
        state = {
            "library_agreement_score": 0.756,
            "validation_passed": True,
            "cross_library_validation": {
                "computed": True,
                "n_segments_compared": 14,
                "spearman_rho": 0.512,
                "sign_agreement": 1.0,
            },
        }
        text = serialize_validation_for_llm(state)
        assert text.startswith("PASSED")
        assert "76%" in text
        assert "14 segments" in text
        assert "0.51" in text

    def test_failed_verdict_says_failed(self):
        state = {
            "library_agreement_score": 0.4,
            "validation_passed": False,
            "cross_library_validation": {"computed": True, "n_segments_compared": 5},
        }
        assert serialize_validation_for_llm(state).startswith("FAILED")

    def test_threshold_is_sane(self):
        assert 0.5 < AGREEMENT_THRESHOLD < 1.0
