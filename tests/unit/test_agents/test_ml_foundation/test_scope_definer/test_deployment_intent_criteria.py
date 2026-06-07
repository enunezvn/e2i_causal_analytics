"""Deployment-intent (clinical | commercial) success-criteria tests.

The deployment-intent axis is ORTHOGONAL to the data-difficulty ``regime``
axis. It recalibrates the deployment bar to the use case: clinical-decision
models keep the literature floor (AUC 0.75; Vickers 2019 / Cook 2007), while
COMMERCIAL targeting/propensity models use a lower, separately-cited floor
(AUC 0.65; Hosmer-Lemeshow 2013; advertising-propensity distribution median
0.76, range 0.60-0.95) plus prevalence-aware operating gates (recall 0.50,
MCC 0.10 per Chen-2024 deflation, net-benefit p_t 0.05 ≈ c_FP:c_FN 1:19 for
low FP cost — the deployed model must still clear NB>0 at that p_t on its merits).

Default intent is "clinical" — the flag NEVER silently loosens the bar.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
    adaptive_success_criteria,
    define_success_criteria,
)


class TestDefineSuccessCriteriaStampsDeploymentIntent:
    """define_success_criteria must stamp deployment_intent at the TOP LEVEL of
    success_criteria (not only inside the path-dependent _adaptive_inputs stash),
    so the evaluator's commercial recall-constrained operating point + sigmoid
    calibration default + the deployer gates can read it reliably."""

    @pytest.mark.parametrize("intent", ["clinical", "commercial"])
    def test_top_level_deployment_intent_stamped(self, intent: str) -> None:
        state = {
            "inferred_problem_type": "binary_classification",
            "experiment_id": "exp_test",
            "deployment_intent": intent,
            "n_samples": 40000,
            "prevalence": 0.0232,
            "feature_count": 19,
            "regime": "default",
        }
        result = asyncio.run(define_success_criteria(state))
        sc = result["success_criteria"]
        assert sc["deployment_intent"] == intent

    def test_missing_deployment_intent_defaults_clinical(self) -> None:
        state = {
            "inferred_problem_type": "binary_classification",
            "experiment_id": "exp_test",
            "n_samples": 40000,
            "prevalence": 0.0232,
            "feature_count": 19,
            "regime": "default",
        }
        result = asyncio.run(define_success_criteria(state))
        assert result["success_criteria"]["deployment_intent"] == "clinical"

    def test_scope_state_schema_retains_deployment_intent(self) -> None:
        """Regression: the pydantic ScopeDefinerState MUST declare
        deployment_intent, else the agent silently drops it and the whole intent
        chain defaults to clinical regardless of --deployment-intent commercial."""
        from src.agents.ml_foundation.scope_definer.state import ScopeDefinerState

        # The field must be DECLARED on the schema; otherwise pydantic drops the
        # agent's forwarded value (extra fields are not retained).
        assert "deployment_intent" in ScopeDefinerState.model_fields


class TestAdaptiveSuccessCriteriaDeploymentIntent:
    def test_clinical_clean_unchanged(self) -> None:
        """Default/clinical clean regime keeps the literature floor 0.75."""
        thr, skipped = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="clean",
            deployment_intent="clinical",
        )
        assert thr["minimum_auc"] == 0.75
        assert thr["minimum_recall"] == 0.65
        assert thr["minimum_mcc"] == 0.45

    def test_default_deployment_intent_is_clinical(self) -> None:
        """Omitting deployment_intent reproduces clinical exactly (back-compat)."""
        clinical, _ = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="clean",
            deployment_intent="clinical",
        )
        default, _ = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="clean",
        )
        assert default == clinical

    def test_commercial_clean_lowers_auc_floor_to_060(self) -> None:
        thr, skipped = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="clean",
            deployment_intent="commercial",
        )
        # Commercial = literature minimum-useful-targeting floor (owner-ratified).
        assert thr["minimum_auc"] == 0.60
        assert thr["minimum_lift_over_baseline"] == 0.08
        assert "minimum_auc" not in skipped

    def test_commercial_relaxes_operating_gates(self) -> None:
        thr, _ = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="clean",
            deployment_intent="commercial",
        )
        assert thr["minimum_recall"] == 0.50
        assert thr["minimum_mcc"] == 0.10

    def test_commercial_fires_auc_even_in_default_regime(self) -> None:
        """Clinical 'default' regime skips the AUC gate (rubric-stress), but a
        commercial run must still carry a real discrimination floor."""
        thr, skipped = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="default",
            deployment_intent="commercial",
        )
        assert "minimum_auc" not in skipped
        assert thr["minimum_auc"] == 0.60

    def test_clinical_default_still_skips_auc(self) -> None:
        thr, skipped = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="default",
            deployment_intent="clinical",
        )
        assert "minimum_auc" in skipped
        assert "minimum_auc" not in thr

    def test_commercial_auc_is_baseline_aware(self) -> None:
        """When the dummy baseline is unusually strong the floor still rises."""
        thr, _ = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.60,
            feature_count=64,
            regime="clean",
            deployment_intent="commercial",
        )
        assert thr["minimum_auc"] == pytest.approx(0.65)  # max(0.60, 0.60+0.05)

    def test_invalid_intent_falls_back_to_clinical(self) -> None:
        thr, _ = adaptive_success_criteria(
            n_samples=9125,
            prevalence=0.125,
            baseline_auc=0.5,
            feature_count=64,
            regime="clean",
            deployment_intent="garbage",
        )
        assert thr["minimum_auc"] == 0.75


class TestDefineSuccessCriteriaThreadsIntent:
    def test_stash_carries_deployment_intent_and_top_level(self) -> None:
        from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
            define_success_criteria,
        )

        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {"min_auc": 0.65},
            "n_samples": 9125,
            "prevalence": 0.125,
            "feature_count": 64,
            "regime": "default",
            "deployment_intent": "commercial",
        }
        out = asyncio.run(define_success_criteria(state))
        sc = out["success_criteria"]
        assert sc["deployment_intent"] == "commercial"
        assert sc["_adaptive_inputs"]["deployment_intent"] == "commercial"

    def test_default_deployment_intent_clinical_when_absent(self) -> None:
        from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
            define_success_criteria,
        )

        state = {
            "inferred_problem_type": "binary_classification",
            "performance_requirements": {"min_auc": 0.65},
            "n_samples": 9125,
            "prevalence": 0.125,
            "feature_count": 64,
            "regime": "default",
        }
        out = asyncio.run(define_success_criteria(state))
        assert out["success_criteria"]["deployment_intent"] == "clinical"
