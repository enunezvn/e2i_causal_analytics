# tests/unit/test_agents/test_causal_impact/test_refutation_efficiency.py
"""#1188 codex iter-1 fixes: refutation must critique the SAME model the
estimation reported.

HIGH: an efficiency run fits the selected covariate estimator with baselines
as X=W, but refutation reconstructed it with common_causes=[] — refuting a
DIFFERENT (unadjusted) model, or failing closed on the ATE-mismatch guard.
MED: DRLearner's reconstruction never mirrored the selector's nuisance /
final-stage models (now GB nuisances + StatsModelsLinearRegression final).
"""

from src.agents.causal_impact.nodes.refutation import (
    _effective_reconstruction_common_causes,
    _reconstruction_nuisance_init_params,
)


class TestEffectiveReconstructionCommonCauses:
    def test_efficiency_dml_run_threads_baselines(self):
        """The reconstructed DML/DR/forest model must condition on the SAME
        baseline columns the reported estimator used."""
        est = {
            "adjustment_type": "efficiency",
            "baseline_covariates_adjusted": ["disease_severity", "age_at_diagnosis"],
            "selected_estimator": "linear_dml",
            "method": "LinearDML",
        }
        assert _effective_reconstruction_common_causes([], est) == [
            "disease_severity",
            "age_at_diagnosis",
        ]

    def test_efficiency_ols_run_stays_unadjusted(self):
        """An OLS-selected efficiency run reported the UNADJUSTED contrast —
        reconstructing it with baselines would fit a different (ANCOVA) model
        and could trip the mismatch guard on the chance-imbalance correction."""
        est = {
            "adjustment_type": "efficiency",
            "baseline_covariates_adjusted": ["disease_severity"],
            "selected_estimator": "ols",
            "method": "linear_regression",
        }
        assert _effective_reconstruction_common_causes([], est) == []

    def test_confounding_run_keeps_confounders(self):
        est = {
            "adjustment_type": "confounding",
            "baseline_covariates_adjusted": [],
            "selected_estimator": "causal_forest",
            "method": "CausalForestDML",
        }
        assert _effective_reconstruction_common_causes(["disease_severity"], est) == [
            "disease_severity"
        ]

    def test_legacy_result_without_labels_keeps_confounders(self):
        assert _effective_reconstruction_common_causes(["x1"], {"method": "LinearDML"}) == ["x1"]


class TestDrLearnerNuisanceMirror:
    def test_drlearner_reconstruction_mirrors_selector_models(self):
        """DoWhy's DRLearner rebuild must use the selector's EXACT models:
        GradientBoosting nuisances + StatsModelsLinearRegression final (the
        honest-CI final stage) — otherwise a DR winner refutes a differently
        fit surface or fails closed."""
        from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        params = _reconstruction_nuisance_init_params(
            "backdoor.econml.dr.DRLearner", discrete_treatment=True
        )
        assert isinstance(params.get("model_regression"), GradientBoostingRegressor)
        assert isinstance(params.get("model_propensity"), GradientBoostingClassifier)
        assert isinstance(params.get("model_final"), StatsModelsLinearRegression)
        assert params["model_regression"].n_estimators == 50
        assert params["model_regression"].random_state == 42
        assert params["model_propensity"].n_estimators == 50
