"""Tests for rationale_generator.py selection explanation logic."""

import pytest

from src.agents.ml_foundation.model_selector.nodes.rationale_generator import (
    _build_rationale_text,
    _check_constraint_compliance,
    _describe_alternatives,
    _explain_why_not_selected,
    _generate_primary_reason,
    _generate_supporting_factors,
    generate_rationale,
)


class TestGeneratePrimaryReason:
    """Test primary reason generation."""

    def test_causal_forest_reason(self):
        """CausalForest should have heterogeneous treatment effects reason."""
        candidate = {
            "name": "CausalForest",
            "family": "causal_ml",
            "strengths": ["heterogeneous_effects"],
        }
        reason = _generate_primary_reason(candidate, "binary_classification")
        assert "heterogeneous treatment effects" in reason.lower()

    def test_linear_dml_reason(self):
        """LinearDML should have fast/interpretable reason."""
        candidate = {
            "name": "LinearDML",
            "family": "causal_ml",
            "strengths": ["fast", "interpretable"],
        }
        reason = _generate_primary_reason(candidate, "binary_classification")
        assert "fast" in reason.lower() or "interpretable" in reason.lower()

    def test_xgboost_reason(self):
        """XGBoost should have accuracy/feature importance reason."""
        candidate = {
            "name": "XGBoost",
            "family": "gradient_boosting",
            "strengths": ["accuracy"],
        }
        reason = _generate_primary_reason(candidate, "binary_classification")
        assert "accuracy" in reason.lower()

    def test_lightgbm_reason(self):
        """LightGBM should have speed/efficiency reason."""
        candidate = {
            "name": "LightGBM",
            "family": "gradient_boosting",
            "strengths": ["speed"],
        }
        reason = _generate_primary_reason(candidate, "binary_classification")
        assert "speed" in reason.lower() or "efficiency" in reason.lower()

    def test_random_forest_reason(self):
        """RandomForest should have ensemble/robust reason."""
        candidate = {
            "name": "RandomForest",
            "family": "ensemble",
            "strengths": ["robust"],
        }
        reason = _generate_primary_reason(candidate, "binary_classification")
        assert "robust" in reason.lower() or "ensemble" in reason.lower()

    def test_linear_model_reason(self):
        """Linear models should have interpretable baseline reason."""
        candidate = {
            "name": "LogisticRegression",
            "family": "linear",
            "strengths": ["interpretable"],
        }
        reason = _generate_primary_reason(candidate, "binary_classification")
        assert "interpretable" in reason.lower()
        assert "baseline" in reason.lower()

    def test_default_reason_includes_problem_type(self):
        """Default reason should include problem type."""
        candidate = {
            "name": "UnknownAlgo",
            "family": "unknown",
            "strengths": [],
        }
        reason = _generate_primary_reason(candidate, "regression")
        assert "regression" in reason.lower()


class TestGenerateSupportingFactors:
    """Test supporting factor generation."""

    def test_strength_based_factors_included(self):
        """Should include factors based on algorithm strengths."""
        candidate = {
            "name": "CausalForest",
            "inference_latency_ms": 50,
            "memory_gb": 4.0,
            "interpretability_score": 0.7,
        }
        strengths = ["heterogeneous_effects", "interpretability"]

        factors = _generate_supporting_factors(candidate, strengths)

        # Should include strength descriptions
        assert any("subgroup-specific" in f.lower() for f in factors)
        assert any("interpretable" in f.lower() for f in factors)

    def test_very_low_latency_factor(self):
        """Should include very low latency factor (<10ms)."""
        candidate = {
            "name": "LinearDML",
            "inference_latency_ms": 5,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
        }
        factors = _generate_supporting_factors(candidate, [])

        assert any("very low inference latency" in f.lower() for f in factors)
        assert any("5ms" in f for f in factors)

    def test_low_latency_factor(self):
        """Should include low latency factor (10-50ms)."""
        candidate = {
            "name": "XGBoost",
            "inference_latency_ms": 20,
            "memory_gb": 2.0,
            "interpretability_score": 0.6,
        }
        factors = _generate_supporting_factors(candidate, [])

        assert any("low inference latency" in f.lower() for f in factors)
        assert any("20ms" in f for f in factors)

    def test_minimal_memory_factor(self):
        """Should include minimal memory factor (<1GB)."""
        candidate = {
            "name": "LogisticRegression",
            "inference_latency_ms": 1,
            "memory_gb": 0.1,
            "interpretability_score": 1.0,
        }
        factors = _generate_supporting_factors(candidate, [])

        assert any("minimal memory" in f.lower() for f in factors)
        assert any("0.1gb" in f.lower() for f in factors)

    def test_moderate_memory_factor(self):
        """Should include moderate memory factor (1-3GB)."""
        candidate = {
            "name": "XGBoost",
            "inference_latency_ms": 20,
            "memory_gb": 2.0,
            "interpretability_score": 0.6,
        }
        factors = _generate_supporting_factors(candidate, [])

        assert any("moderate memory" in f.lower() for f in factors)

    def test_high_interpretability_factor(self):
        """Should include high interpretability factor (>=0.8)."""
        candidate = {
            "name": "LinearDML",
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
        }
        factors = _generate_supporting_factors(candidate, [])

        assert any("high interpretability" in f.lower() for f in factors)

    def test_multiple_strengths_all_included(self):
        """Should include all applicable strength-based factors."""
        candidate = {
            "name": "LinearDML",
            "inference_latency_ms": 5,
            "memory_gb": 0.5,
            "interpretability_score": 0.9,
        }
        strengths = ["fast", "interpretable", "low_variance", "causal_inference"]

        factors = _generate_supporting_factors(candidate, strengths)

        # Should have multiple factors
        assert len(factors) >= 4


class TestDescribeAlternatives:
    """Test alternative candidate descriptions."""

    def test_describe_alternatives_includes_all_alternatives(self):
        """Should describe all alternative candidates."""
        primary = {
            "name": "Primary",
            "selection_score": 0.90,
            "inference_latency_ms": 10,
            "memory_gb": 1.0,
            "interpretability_score": 0.9,
        }
        alternatives = [
            {
                "name": "Alt1",
                "selection_score": 0.85,
                "inference_latency_ms": 20,
                "memory_gb": 2.0,
                "interpretability_score": 0.8,
            },
            {
                "name": "Alt2",
                "selection_score": 0.75,
                "inference_latency_ms": 50,
                "memory_gb": 4.0,
                "interpretability_score": 0.6,
            },
        ]

        described = _describe_alternatives(alternatives, primary)

        assert len(described) == 2
        assert described[0]["algorithm_name"] == "Alt1"
        assert described[1]["algorithm_name"] == "Alt2"

    def test_alternative_descriptions_include_required_fields(self):
        """Each alternative should include required fields."""
        primary = {"name": "Primary", "selection_score": 0.90}
        alternatives = [{"name": "Alt", "selection_score": 0.80}]

        described = _describe_alternatives(alternatives, primary)

        assert "algorithm_name" in described[0]
        assert "selection_score" in described[0]
        assert "score_difference" in described[0]
        assert "reason_not_selected" in described[0]

    def test_score_difference_calculated_correctly(self):
        """Score difference should be primary - alternative."""
        primary = {"name": "Primary", "selection_score": 0.90}
        alternatives = [{"name": "Alt", "selection_score": 0.75}]

        described = _describe_alternatives(alternatives, primary)

        # 0.90 - 0.75 = 0.15
        assert abs(described[0]["score_difference"] - 0.15) < 0.001


class TestExplainWhyNotSelected:
    """Test alternative rejection reasons."""

    def test_high_latency_reason(self):
        """Should explain high latency if >2x primary."""
        alternative = {
            "inference_latency_ms": 100,
            "memory_gb": 2.0,
            "interpretability_score": 0.8,
        }
        primary = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.8,
        }

        reason = _explain_why_not_selected(alternative, primary, 0.05)

        assert "latency" in reason.lower()

    def test_high_memory_reason(self):
        """Should explain high memory if >1.5x primary."""
        alternative = {
            "inference_latency_ms": 10,
            "memory_gb": 6.0,
            "interpretability_score": 0.8,
        }
        primary = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.8,
        }

        reason = _explain_why_not_selected(alternative, primary, 0.05)

        assert "memory" in reason.lower()

    def test_low_interpretability_reason(self):
        """Should explain low interpretability if <0.2 difference."""
        alternative = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.4,
        }
        primary = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.9,
        }

        reason = _explain_why_not_selected(alternative, primary, 0.05)

        assert "interpretability" in reason.lower()

    def test_lower_score_reason(self):
        """Should explain lower overall score if >0.1 difference."""
        alternative = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.8,
        }
        primary = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.8,
        }

        reason = _explain_why_not_selected(alternative, primary, 0.15)

        assert "lower overall" in reason.lower() or "lower" in reason.lower()

    def test_slightly_lower_score_default(self):
        """Should use default reason for small score differences."""
        alternative = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.8,
        }
        primary = {
            "inference_latency_ms": 10,
            "memory_gb": 2.0,
            "interpretability_score": 0.8,
        }

        reason = _explain_why_not_selected(alternative, primary, 0.05)

        assert "slightly lower" in reason.lower()


class TestCheckConstraintCompliance:
    """Test constraint compliance checking."""

    def test_latency_constraint_pass(self):
        """Should pass latency constraint if within limit."""
        candidate = {"inference_latency_ms": 20, "memory_gb": 2.0}
        constraints = ["inference_latency_<50ms"]

        compliance = _check_constraint_compliance(candidate, constraints)

        assert compliance["inference_latency_<50ms"] is True

    def test_latency_constraint_fail(self):
        """Should fail latency constraint if exceeds limit."""
        candidate = {"inference_latency_ms": 100, "memory_gb": 2.0}
        constraints = ["inference_latency_<50ms"]

        compliance = _check_constraint_compliance(candidate, constraints)

        assert compliance["inference_latency_<50ms"] is False

    def test_memory_constraint_pass(self):
        """Should pass memory constraint if within limit."""
        candidate = {"inference_latency_ms": 20, "memory_gb": 2.0}
        constraints = ["memory_<4gb"]

        compliance = _check_constraint_compliance(candidate, constraints)

        assert compliance["memory_<4gb"] is True

    def test_memory_constraint_fail(self):
        """Should fail memory constraint if exceeds limit."""
        candidate = {"inference_latency_ms": 20, "memory_gb": 8.0}
        constraints = ["memory_<4gb"]

        compliance = _check_constraint_compliance(candidate, constraints)

        assert compliance["memory_<4gb"] is False

    def test_multiple_constraints_checked(self):
        """Should check all constraints."""
        candidate = {"inference_latency_ms": 20, "memory_gb": 2.0}
        constraints = ["inference_latency_<50ms", "memory_<4gb"]

        compliance = _check_constraint_compliance(candidate, constraints)

        assert len(compliance) == 2
        assert compliance["inference_latency_<50ms"] is True
        assert compliance["memory_<4gb"] is True

    def test_malformed_constraint_passes_by_default(self):
        """Malformed constraints should default to pass."""
        candidate = {"inference_latency_ms": 20, "memory_gb": 2.0}
        constraints = ["invalid_constraint_format"]

        compliance = _check_constraint_compliance(candidate, constraints)

        assert compliance["invalid_constraint_format"] is True

    def test_unknown_constraint_passes(self):
        """Unknown constraints should default to pass."""
        candidate = {"inference_latency_ms": 20, "memory_gb": 2.0}
        constraints = ["unknown_constraint"]

        compliance = _check_constraint_compliance(candidate, constraints)

        assert compliance["unknown_constraint"] is True


class TestBuildRationaleText:
    """Test rationale text formatting."""

    def test_rationale_includes_algorithm_name_and_score(self):
        """Rationale should include algorithm name and score."""
        text = _build_rationale_text("XGBoost", "High accuracy", [], 0.85, [])

        assert "XGBoost" in text
        assert "0.850" in text

    def test_rationale_includes_primary_reason(self):
        """Rationale should include primary reason."""
        text = _build_rationale_text("XGBoost", "High accuracy", [], 0.85, [])

        assert "Primary Reason: High accuracy" in text

    def test_rationale_includes_supporting_factors(self):
        """Rationale should include all supporting factors."""
        factors = ["Fast training", "Low memory", "High interpretability"]
        text = _build_rationale_text("LinearDML", "Best for speed", factors, 0.90, [])

        assert "Supporting Factors:" in text
        assert "Fast training" in text
        assert "Low memory" in text
        assert "High interpretability" in text

    def test_rationale_includes_alternatives(self):
        """Rationale should include alternative candidates."""
        alternatives = [
            {
                "algorithm_name": "XGBoost",
                "selection_score": 0.80,
                "reason_not_selected": "Higher latency",
            },
            {
                "algorithm_name": "RandomForest",
                "selection_score": 0.75,
                "reason_not_selected": "Lower accuracy",
            },
        ]
        text = _build_rationale_text("CausalForest", "Best overall", [], 0.90, alternatives)

        assert "Alternatives Considered:" in text
        assert "XGBoost" in text
        assert "Higher latency" in text
        assert "RandomForest" in text

    def test_rationale_limits_alternatives_to_3(self):
        """Rationale should show max 3 alternatives."""
        alternatives = [
            {
                "algorithm_name": f"Alt{i}",
                "selection_score": 0.8 - i * 0.1,
                "reason_not_selected": "Lower score",
            }
            for i in range(5)
        ]
        text = _build_rationale_text("Primary", "Best", [], 0.90, alternatives)

        # Should only include first 3
        assert "Alt0" in text
        assert "Alt1" in text
        assert "Alt2" in text
        assert "Alt3" not in text


@pytest.mark.asyncio
class TestGenerateRationale:
    """Test complete rationale generation."""

    async def test_generate_rationale_complete_flow(self):
        """Should generate complete rationale with all components."""
        state = {
            "primary_candidate": {
                "name": "CausalForest",
                "family": "causal_ml",
                "strengths": ["heterogeneous_effects", "interpretability"],
                "selection_score": 0.90,
                "inference_latency_ms": 50,
                "memory_gb": 4.0,
                "interpretability_score": 0.7,
            },
            "alternative_candidates": [
                {
                    "name": "LinearDML",
                    "selection_score": 0.85,
                    "inference_latency_ms": 10,
                    "memory_gb": 1.0,
                    "interpretability_score": 0.9,
                },
                {
                    "name": "XGBoost",
                    "selection_score": 0.75,
                    "inference_latency_ms": 20,
                    "memory_gb": 2.0,
                    "interpretability_score": 0.6,
                },
            ],
            "technical_constraints": ["inference_latency_<100ms", "memory_<8gb"],
            "problem_type": "binary_classification",
        }

        result = await generate_rationale(state)

        # Verify all output fields present
        assert "selection_rationale" in result
        assert "primary_reason" in result
        assert "supporting_factors" in result
        assert "alternatives_considered" in result
        assert "constraint_compliance" in result

        # Verify rationale text structure
        rationale = result["selection_rationale"]
        assert "CausalForest" in rationale
        assert "Primary Reason:" in rationale
        assert "Supporting Factors:" in rationale
        assert "Alternatives Considered:" in rationale

        # Verify constraint compliance
        compliance = result["constraint_compliance"]
        assert compliance["inference_latency_<100ms"] is True
        assert compliance["memory_<8gb"] is True

    async def test_generate_rationale_no_primary_candidate_error(self):
        """Should return error if no primary candidate."""
        state = {"primary_candidate": {}}

        result = await generate_rationale(state)

        assert "error" in result
        assert result["error_type"] == "missing_primary_candidate_error"

    async def test_generate_rationale_no_alternatives(self):
        """Should work with no alternatives."""
        state = {
            "primary_candidate": {
                "name": "OnlyOne",
                "family": "causal_ml",
                "strengths": ["fast"],
                "selection_score": 0.90,
                "inference_latency_ms": 10,
                "memory_gb": 1.0,
                "interpretability_score": 0.9,
            },
            "alternative_candidates": [],
            "technical_constraints": [],
            "problem_type": "binary_classification",
        }

        result = await generate_rationale(state)

        assert "selection_rationale" in result
        assert len(result["alternatives_considered"]) == 0
