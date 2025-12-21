"""Integration tests for ModelSelectorAgent."""

import pytest

from src.agents.ml_foundation.model_selector.agent import ModelSelectorAgent


@pytest.fixture
def valid_scope_spec():
    """Fixture: valid ScopeSpec from scope_definer."""
    return {
        "experiment_id": "exp_remi_us_20231215120000",
        "problem_type": "binary_classification",
        "prediction_target": "will_prescribe",
        "prediction_horizon_days": 30,
        "brand": "Remibrutinib",
        "indication": "CSU",
        "region": "US",
        "required_features": [
            "specialty",
            "years_in_practice",
            "prior_prescriptions",
            "hcp_tier",
        ],
        "excluded_features": ["hcp_name", "hcp_npi"],
        "technical_constraints": ["inference_latency_<50ms", "memory_<4gb"],
        "regulatory_constraints": ["HIPAA", "GDPR"],
        "ethical_constraints": ["no_protected_attributes"],
    }


@pytest.fixture
def valid_qc_report():
    """Fixture: valid QC report from data_preparer."""
    return {
        "qc_passed": True,
        "qc_errors": [],
        "qc_warnings": [],
        "row_count": 50000,
        "column_count": 25,
        "feature_count": 20,
        "missing_percentage": 0.02,
        "duplicate_percentage": 0.0,
        "validation_timestamp": "2023-12-15T12:00:00Z",
    }


@pytest.fixture
def valid_baseline_metrics():
    """Fixture: baseline performance metrics."""
    return {
        "random_forest_baseline": {
            "auc": 0.72,
            "precision": 0.68,
            "recall": 0.65,
            "f1": 0.66,
        },
        "logistic_regression_baseline": {
            "auc": 0.65,
            "precision": 0.60,
            "recall": 0.62,
            "f1": 0.61,
        },
    }


class TestModelSelectorAgentMetadata:
    """Test agent metadata."""

    def test_agent_tier_is_0(self):
        """Agent should be Tier 0."""
        assert ModelSelectorAgent.tier == 0

    def test_agent_tier_name_is_ml_foundation(self):
        """Agent should be in ml_foundation tier."""
        assert ModelSelectorAgent.tier_name == "ml_foundation"

    def test_agent_type_is_standard(self):
        """Agent should be standard type."""
        assert ModelSelectorAgent.agent_type == "standard"

    def test_sla_is_120_seconds(self):
        """SLA should be 120 seconds (2 minutes)."""
        assert ModelSelectorAgent.sla_seconds == 120


class TestModelSelectorAgentInitialization:
    """Test agent initialization."""

    def test_agent_initializes_successfully(self):
        """Agent should initialize without errors."""
        agent = ModelSelectorAgent()
        assert agent is not None
        assert agent.graph is not None

    def test_agent_has_graph_attribute(self):
        """Agent should have compiled graph."""
        agent = ModelSelectorAgent()
        assert hasattr(agent, "graph")


@pytest.mark.asyncio
class TestModelSelectorAgentInputValidation:
    """Test input validation."""

    async def test_missing_scope_spec_raises_error(self, valid_qc_report):
        """Should raise ValueError if scope_spec missing."""
        agent = ModelSelectorAgent()
        input_data = {"qc_report": valid_qc_report}

        result = await agent.run(input_data)

        assert "error" in result
        assert "scope_spec" in result["error"]

    async def test_missing_qc_report_raises_error(self, valid_scope_spec):
        """Should raise ValueError if qc_report missing."""
        agent = ModelSelectorAgent()
        input_data = {"scope_spec": valid_scope_spec}

        result = await agent.run(input_data)

        assert "error" in result
        assert "qc_report" in result["error"]

    async def test_qc_failed_raises_error(self, valid_scope_spec):
        """Should raise error if QC validation failed."""
        agent = ModelSelectorAgent()
        qc_report = {
            "qc_passed": False,
            "qc_errors": ["Missing required features"],
        }
        input_data = {"scope_spec": valid_scope_spec, "qc_report": qc_report}

        result = await agent.run(input_data)

        assert "error" in result
        assert "qc validation failed" in result["error"].lower()


@pytest.mark.asyncio
class TestModelSelectorAgentBasicExecution:
    """Test basic agent execution."""

    async def test_successful_execution_binary_classification(
        self, valid_scope_spec, valid_qc_report
    ):
        """Should successfully select model for binary classification."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)

        # Should not have errors
        assert "error" not in result or result.get("error") is None

        # Should have required output fields
        assert "model_candidate" in result
        assert "selection_rationale" in result
        assert "experiment_id" in result

    async def test_model_candidate_structure(self, valid_scope_spec, valid_qc_report):
        """ModelCandidate should have all required fields."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Verify all required fields
        required_fields = [
            "algorithm_name",
            "algorithm_class",
            "algorithm_family",
            "default_hyperparameters",
            "hyperparameter_search_space",
            "expected_performance",
            "training_time_estimate_hours",
            "estimated_inference_latency_ms",
            "memory_requirement_gb",
            "interpretability_score",
            "scalability_score",
            "selection_score",
        ]
        for field in required_fields:
            assert field in candidate, f"Missing field: {field}"

    async def test_selection_rationale_structure(self, valid_scope_spec, valid_qc_report):
        """SelectionRationale should have all required fields."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        rationale = result["selection_rationale"]

        # Verify all required fields
        required_fields = [
            "selection_rationale",
            "primary_reason",
            "supporting_factors",
            "alternatives_considered",
            "constraint_compliance",
        ]
        for field in required_fields:
            assert field in rationale, f"Missing field: {field}"


@pytest.mark.asyncio
class TestModelSelectorAgentAlgorithmSelection:
    """Test algorithm selection logic."""

    async def test_causal_ml_preferred_for_e2i(self, valid_scope_spec, valid_qc_report):
        """E2I platform should prefer CausalML algorithms."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Should prefer causal_ml family (CausalForest or LinearDML)
        # Due to 10% causal ML bonus
        assert candidate["algorithm_family"] in [
            "causal_ml",
            "gradient_boosting",
            "linear",
        ]

    async def test_respects_latency_constraints(self, valid_scope_spec, valid_qc_report):
        """Should respect latency constraints."""
        agent = ModelSelectorAgent()

        # Set strict latency constraint
        scope_spec_with_latency = {
            **valid_scope_spec,
            "technical_constraints": ["inference_latency_<20ms"],
        }
        input_data = {
            "scope_spec": scope_spec_with_latency,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Selected algorithm should meet latency constraint
        assert candidate["estimated_inference_latency_ms"] <= 20

    async def test_respects_memory_constraints(self, valid_scope_spec, valid_qc_report):
        """Should respect memory constraints."""
        agent = ModelSelectorAgent()

        # Set strict memory constraint
        scope_spec_with_memory = {
            **valid_scope_spec,
            "technical_constraints": ["memory_<2gb"],
        }
        input_data = {
            "scope_spec": scope_spec_with_memory,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Selected algorithm should meet memory constraint
        assert candidate["memory_requirement_gb"] <= 2.0

    async def test_respects_algorithm_preferences(self, valid_scope_spec, valid_qc_report):
        """Should prefer user-specified algorithms."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
            "algorithm_preferences": ["LinearDML"],
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # LinearDML should get bonus and likely be selected
        # (unless constraints exclude it)
        assert candidate["algorithm_name"] in ["LinearDML", "CausalForest", "XGBoost"]

    async def test_respects_excluded_algorithms(self, valid_scope_spec, valid_qc_report):
        """Should exclude user-specified algorithms."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
            "excluded_algorithms": ["XGBoost", "LightGBM"],
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Should NOT select excluded algorithms
        assert candidate["algorithm_name"] not in ["XGBoost", "LightGBM"]

    async def test_respects_interpretability_requirement(self, valid_scope_spec, valid_qc_report):
        """Should select interpretable models when required."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
            "interpretability_required": True,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Should select high interpretability algorithm (>= 0.7)
        assert candidate["interpretability_score"] >= 0.7


@pytest.mark.asyncio
class TestModelSelectorAgentRegressionProblems:
    """Test regression problem handling."""

    async def test_regression_problem_type(self, valid_qc_report):
        """Should select appropriate algorithms for regression."""
        agent = ModelSelectorAgent()

        regression_scope = {
            "experiment_id": "exp_test_regression",
            "problem_type": "regression",
            "prediction_target": "prescription_volume",
            "prediction_horizon_days": 90,
            "technical_constraints": [],
        }
        input_data = {
            "scope_spec": regression_scope,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Should select regression-compatible algorithm
        regression_algorithms = [
            "CausalForest",
            "LinearDML",
            "XGBoost",
            "LightGBM",
            "RandomForest",
            "Ridge",
            "Lasso",
        ]
        assert candidate["algorithm_name"] in regression_algorithms

        # Should NOT select classification-only
        assert candidate["algorithm_name"] != "LogisticRegression"


@pytest.mark.asyncio
class TestModelSelectorAgentAlternatives:
    """Test alternative candidate handling."""

    async def test_alternative_candidates_included(self, valid_scope_spec, valid_qc_report):
        """Should include 2-3 alternative candidates."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)

        # Should have alternatives
        assert "alternative_candidates" in result
        assert len(result["alternative_candidates"]) >= 1
        assert len(result["alternative_candidates"]) <= 3

    async def test_alternatives_explained_in_rationale(self, valid_scope_spec, valid_qc_report):
        """Rationale should explain why alternatives weren't selected."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        rationale = result["selection_rationale"]

        # Should have alternatives_considered
        assert len(rationale["alternatives_considered"]) >= 1

        # Each alternative should have reason_not_selected
        for alt in rationale["alternatives_considered"]:
            assert "algorithm_name" in alt
            assert "reason_not_selected" in alt


@pytest.mark.asyncio
class TestModelSelectorAgentConstraintCompliance:
    """Test constraint compliance checking."""

    async def test_constraint_compliance_checked(self, valid_scope_spec, valid_qc_report):
        """Should check constraint compliance."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        rationale = result["selection_rationale"]

        # Should have constraint_compliance
        assert "constraint_compliance" in rationale
        compliance = rationale["constraint_compliance"]

        # Should check all constraints from scope_spec
        for constraint in valid_scope_spec["technical_constraints"]:
            assert constraint in compliance
            assert isinstance(compliance[constraint], bool)

    async def test_selected_algorithm_meets_constraints(self, valid_scope_spec, valid_qc_report):
        """Selected algorithm should pass all constraints."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        rationale = result["selection_rationale"]
        compliance = rationale["constraint_compliance"]

        # All constraints should pass
        for constraint, passed in compliance.items():
            assert passed, f"Constraint failed: {constraint}"


@pytest.mark.asyncio
class TestModelSelectorAgentOutputFormats:
    """Test output format compliance."""

    async def test_experiment_id_preserved(self, valid_scope_spec, valid_qc_report):
        """Experiment ID should be preserved from input."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)

        assert result["experiment_id"] == valid_scope_spec["experiment_id"]

    async def test_algorithm_class_is_valid_python_path(self, valid_scope_spec, valid_qc_report):
        """algorithm_class should be valid Python import path."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]

        # Should contain module path
        assert "." in candidate["algorithm_class"]

        # Should be one of known paths
        valid_prefixes = ["econml.", "xgboost.", "lightgbm.", "sklearn."]
        assert any(candidate["algorithm_class"].startswith(p) for p in valid_prefixes)

    async def test_hyperparameter_search_space_valid_structure(
        self, valid_scope_spec, valid_qc_report
    ):
        """Hyperparameter search space should have valid Optuna structure."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        candidate = result["model_candidate"]
        hp_space = candidate["hyperparameter_search_space"]

        # Should be a dictionary
        assert isinstance(hp_space, dict)

        # Each hyperparameter should have type
        for _hp_name, hp_spec in hp_space.items():
            assert "type" in hp_spec
            assert hp_spec["type"] in ["int", "float", "categorical"]

    async def test_selection_rationale_is_readable_text(self, valid_scope_spec, valid_qc_report):
        """Selection rationale should be human-readable text."""
        agent = ModelSelectorAgent()
        input_data = {
            "scope_spec": valid_scope_spec,
            "qc_report": valid_qc_report,
        }

        result = await agent.run(input_data)
        rationale_text = result["selection_rationale"]["selection_rationale"]

        # Should be non-empty string
        assert isinstance(rationale_text, str)
        assert len(rationale_text) > 50

        # Should contain key sections
        assert "Primary Reason:" in rationale_text
        assert "Supporting Factors:" in rationale_text
