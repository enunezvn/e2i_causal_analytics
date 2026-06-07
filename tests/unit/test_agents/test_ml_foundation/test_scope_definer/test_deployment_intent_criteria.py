"""Deployment-intent (clinical | commercial) success-criteria tests.

The deployment-intent axis is ORTHOGONAL to the data-difficulty ``regime``
axis. It recalibrates the deployment bar to the use case: clinical-decision
models keep the literature floor (AUC 0.75; Vickers 2019 / Cook 2007), while
COMMERCIAL targeting/propensity models use a lower, separately-cited floor
(AUC 0.65; Hosmer-Lemeshow 2013; advertising-propensity distribution median
0.76, range 0.60-0.95) plus prevalence-aware operating gates (recall 0.50,
MCC 0.10 per Chen-2024 deflation, net-benefit p_t 0.10 for low FP cost).

Default intent is "clinical" — the flag NEVER silently loosens the bar.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.ml_foundation.scope_definer.nodes.criteria_validator import (
    adaptive_success_criteria,
)


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
