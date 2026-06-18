"""Fast (non-slow) unit tests for refutation-node helpers.

``test_refutation.py`` is module-marked ``slow`` (it drives the real DoWhy
refutation suite, minutes per test, off-PR). These pure-logic helper tests have
no DoWhy dependency, so they live here to run in the on-PR backend lane.
"""

from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.agents.causal_impact.nodes.refutation import _scaled_nuisance_init_params


class TestScaledNuisanceInitParams:
    """Perf-fix: the reconstructed estimator's nuisance models are scaled
    (StandardScaler) so the logistic propensity converges fast instead of
    grinding lbfgs to max_iter (~40s/refit) on mixed-scale covariates. Applied
    ONLY to the lbfgs-prone estimators (LinearDML / DRLearner); forest-based
    (CausalForestDML) and plain linear methods are left untouched."""

    def test_lineardml_discrete_scaled_classifier_propensity(self):
        p = _scaled_nuisance_init_params("backdoor.econml.dml.LinearDML", discrete_treatment=True)
        assert set(p) == {"model_y", "model_t"}
        for m in p.values():
            assert isinstance(m, Pipeline)
            assert isinstance(m.steps[0][1], StandardScaler)
        # discrete treatment -> propensity model_t is a classifier; model_y a regressor
        assert is_classifier(p["model_t"])
        assert not is_classifier(p["model_y"])

    def test_lineardml_continuous_treatment_uses_regressor(self):
        p = _scaled_nuisance_init_params("backdoor.econml.dml.LinearDML", discrete_treatment=False)
        assert not is_classifier(p["model_t"])

    def test_drlearner_scaled_propensity_and_regression(self):
        p = _scaled_nuisance_init_params("backdoor.econml.dr.DRLearner", discrete_treatment=True)
        assert set(p) == {"model_regression", "model_propensity"}
        assert is_classifier(p["model_propensity"])  # DRLearner propensity is always a classifier
        assert isinstance(p["model_propensity"].steps[0][1], StandardScaler)

    def test_forest_and_linear_methods_get_no_override(self):
        # CausalForestDML uses scale-invariant forest nuisance -> no scaling needed.
        assert (
            _scaled_nuisance_init_params(
                "backdoor.econml.dml.CausalForestDML", discrete_treatment=True
            )
            == {}
        )
        # Plain linear regression has no iterative nuisance to converge.
        assert (
            _scaled_nuisance_init_params("backdoor.linear_regression", discrete_treatment=True)
            == {}
        )
