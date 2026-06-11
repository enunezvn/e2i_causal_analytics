"""Issue #868: final-fit degeneracy guard for HPO best params.

Measured defect (synthetic-CSU persistence cohort): Optuna selected
``{'C': 0.0019517..., 'penalty': 'l1'}`` with best value
0.6499999999999999 — which is EXACTLY ``0.7*0.5 + 0.3*1.0``, the blended
roc_auc objective rewarding a degenerate all-positive constant
classifier (val AUC 0.5, minority recall 1.0). The final fitted model at
that C crosses the L1 coefficient-zeroing cliff: intercept-only model,
near-constant probabilities, test AUC exactly 0.500 — while the same
study's next-best trial (healthy C) delivers real discrimination.

The guard (``_apply_degeneracy_guard`` in
``nodes/hyperparameter_tuner.py``) re-fits the HPO best params
final-fit-style, detects collapse on the validation split, and falls
back to the next-best DISTINCT trial params. Conservative: it only
fires when the HPO promise was real (best_value >= 0.58) AND the fit is
genuinely collapsed (near-constant probabilities or chance-level val
AUC). Healthy best params are byte-identical pass-through.
"""

from unittest.mock import patch

import numpy as np
import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import create_trial
from sklearn.linear_model import LogisticRegression

from src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner import (
    DEGENERACY_GUARD_PROB_STD_FLOOR,
    DEGENERACY_GUARD_PROMISE_FLOOR,
    DEGENERACY_GUARD_RECALIBRATION_SLOPE_CEILING,
    DEGENERACY_GUARD_VAL_AUC_FLOOR,
    _apply_degeneracy_guard,
    _detect_final_fit_collapse,
    tune_hyperparameters,
    validate_hpo_output,
)

# ---------------------------------------------------------------------------
# Fixtures: linearly-separable-ish binary data where L1 LR at the cliff C
# (1e-4) zeroes ALL coefficients (verified: prob std ~1e-16, val AUC 0.5)
# while healthy C=1.0 discriminates (val AUC ~0.97).
# ---------------------------------------------------------------------------

CLIFF_C = 1e-4
HEALTHY_C = 1.0

LR_FIXED = {"random_state": 42, "max_iter": 1000, "solver": "saga"}

_DISTS = {
    "C": FloatDistribution(1e-5, 10.0, log=True),
    "penalty": CategoricalDistribution(("l1", "l2")),
}


@pytest.fixture()
def lr_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5))
    logit = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
    y = (logit + rng.normal(scale=1.0, size=400) > 0).astype(int)
    return X[:300], y[:300], X[300:], y[300:]


def _make_study(trials):
    """Build a real (in-memory) Optuna study from [(params, value), ...]."""
    study = optuna.create_study(direction="maximize")
    for params, value in trials:
        study.add_trial(
            create_trial(
                params=params,
                distributions={k: _DISTS[k] for k in params},
                value=value,
            )
        )
    return study


def _results_from_study(study):
    """Mirror of OptunaOptimizer.optimize()'s results contract subset."""
    return {
        "best_params": dict(study.best_params),
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "study_name": study.study_name,
    }


def _run_guard(study, results, lr_data, problem_type="binary_classification"):
    X_train, y_train, X_val, y_val = lr_data
    return _apply_degeneracy_guard(
        study=study,
        results=results,
        model_class=LogisticRegression,
        algorithm_name="LogisticRegression",
        default_hyperparameters={},
        fixed_params=dict(LR_FIXED),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        problem_type=problem_type,
    )


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------


class TestDetectFinalFitCollapse:
    def test_intercept_only_l1_model_is_collapsed(self, lr_data):
        """The measured #868 mode: cliff C zeroes coefficients -> constant probs."""
        X_train, y_train, X_val, y_val = lr_data
        model = LogisticRegression(C=CLIFF_C, penalty="l1", **LR_FIXED).fit(X_train, y_train)
        collapsed, reason, diagnostics = _detect_final_fit_collapse(model, X_val, y_val)
        assert collapsed is True
        assert diagnostics["val_prob_std"] < DEGENERACY_GUARD_PROB_STD_FLOOR
        assert "constant" in reason

    def test_healthy_model_is_not_collapsed(self, lr_data):
        X_train, y_train, X_val, y_val = lr_data
        model = LogisticRegression(C=HEALTHY_C, penalty="l1", **LR_FIXED).fit(X_train, y_train)
        collapsed, reason, diagnostics = _detect_final_fit_collapse(model, X_val, y_val)
        assert collapsed is False
        assert diagnostics["val_prob_std"] >= DEGENERACY_GUARD_PROB_STD_FLOOR
        assert diagnostics["val_auc"] > DEGENERACY_GUARD_VAL_AUC_FLOOR

    def test_chance_level_auc_is_collapsed_even_with_prob_spread(self):
        """AUC <= 0.52 with non-constant probs is still a collapse.

        Deterministic construction: probabilities strictly increasing with
        labels alternating 0,1,0,1,... -> AUC = 0.505 exactly (positives
        occupy ranks 2,4,...,200: (10100 - 5050) / 10000).
        """

        class _ChanceModel:
            def predict_proba(self, X):
                p = np.linspace(0.05, 0.95, len(X))
                return np.column_stack([1 - p, p])

        X_val = np.zeros((200, 3))
        y_val = np.tile([0, 1], 100)
        collapsed, reason, diagnostics = _detect_final_fit_collapse(_ChanceModel(), X_val, y_val)
        assert collapsed is True
        assert diagnostics["val_auc"] == pytest.approx(0.505)
        assert diagnostics["val_auc"] <= DEGENERACY_GUARD_VAL_AUC_FLOOR
        assert "AUC" in reason

    def test_single_class_predictions_without_predict_proba(self):
        class _HardClassifier:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        X_val = np.zeros((50, 2))
        y_val = np.array([0, 1] * 25)
        collapsed, reason, _ = _detect_final_fit_collapse(_HardClassifier(), X_val, y_val)
        assert collapsed is True
        assert "single-class" in reason

    def test_probability_crushed_model_is_calibration_collapsed(self, lr_data):
        """Acceptance-run iteration: an over-regularized L2 fit keeps ranking
        (AUC 0.97) but crushes probabilities (std 0.105, raw recalibration
        slope 11.9 on this fixture) -> >=2x underconfident -> collapse."""
        X_train, y_train, X_val, y_val = lr_data
        model = LogisticRegression(C=0.005, penalty="l2", **LR_FIXED).fit(X_train, y_train)
        collapsed, reason, diagnostics = _detect_final_fit_collapse(model, X_val, y_val)
        assert collapsed is True
        assert diagnostics["val_auc"] > DEGENERACY_GUARD_VAL_AUC_FLOOR  # ranking intact
        assert (
            diagnostics["val_recalibration_slope"] >= DEGENERACY_GUARD_RECALIBRATION_SLOPE_CEILING
        )
        assert "underconfident" in reason

    def test_healthy_model_records_recalibration_slope_below_ceiling(self, lr_data):
        """C=1.0 measures raw slope ~1.14 on this fixture — no fire."""
        X_train, y_train, X_val, y_val = lr_data
        model = LogisticRegression(C=HEALTHY_C, penalty="l1", **LR_FIXED).fit(X_train, y_train)
        collapsed, _, diagnostics = _detect_final_fit_collapse(model, X_val, y_val)
        assert collapsed is False
        assert diagnostics["val_recalibration_slope"] < DEGENERACY_GUARD_RECALIBRATION_SLOPE_CEILING

    def test_slope_criterion_skipped_when_unstable(self):
        """n_pos < 30 -> the van Calster helper returns NaN -> criterion
        skipped (fail-safe to healthy), never a false fire."""

        class _MildModel:
            def predict_proba(self, X):
                p = np.linspace(0.2, 0.8, len(X))
                return np.column_stack([1 - p, p])

        X_val = np.zeros((40, 2))
        # 20 positives < 30 -> slope is NaN; AUC here is healthy (positives
        # concentrated at high p): y sorted with p ascending.
        y_val = np.array([0] * 20 + [1] * 20)
        collapsed, _, diagnostics = _detect_final_fit_collapse(_MildModel(), X_val, y_val)
        assert collapsed is False
        assert "val_recalibration_slope" not in diagnostics


# ---------------------------------------------------------------------------
# Guard behavior: fallback on collapse
# ---------------------------------------------------------------------------


class TestDegeneracyGuardFallback:
    def test_falls_back_to_next_best_distinct_trial_on_collapse(self, lr_data):
        """Best trial is the L1 cliff (promise 0.65); trial 1 is healthy."""
        study = _make_study(
            [
                ({"C": CLIFF_C, "penalty": "l1"}, 0.65),  # degenerate "best"
                ({"C": HEALTHY_C, "penalty": "l1"}, 0.62),  # healthy runner-up
            ]
        )
        results = _results_from_study(study)
        meta = _run_guard(study, results, lr_data)

        assert meta is not None
        assert meta["fired"] is True
        assert meta["fallback_adopted"] is True
        assert meta["adopted_trial"] == 1
        assert results["best_params"] == {"C": HEALTHY_C, "penalty": "l1"}
        assert results["best_trial_number"] == 1
        assert results["best_value"] == pytest.approx(0.62)
        # The rejected degenerate config is recorded for the log trail.
        assert meta["original_params"] == {"C": CLIFF_C, "penalty": "l1"}
        assert any(r["params"] == {"C": CLIFF_C, "penalty": "l1"} for r in meta["rejected"])

    def test_skips_param_sets_equal_to_ones_already_tried(self, lr_data):
        """A duplicate of the degenerate best params must not be re-fit."""
        fit_log = []

        class _StubModel:
            def __init__(self, **params):
                self.params = params

            def fit(self, X, y):
                fit_log.append(dict(self.params))
                return self

            def predict_proba(self, X):
                X = np.asarray(X)
                if self.params.get("C", 1.0) < 0.01:  # cliff -> constant probs
                    return np.tile([0.3, 0.7], (len(X), 1))
                p = 1.0 / (1.0 + np.exp(-X[:, 0]))
                return np.column_stack([1 - p, p])

        rng = np.random.default_rng(3)
        X_train = rng.normal(size=(100, 3))
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = rng.normal(size=(200, 3))
        # Labels drawn FROM the stub's own healthy probabilities so the
        # healthy branch is well-calibrated (true recalibration slope = 1);
        # deterministic labels would be perfectly separable and blow the
        # recalibration slope past the crush ceiling.
        p_val = 1.0 / (1.0 + np.exp(-X_val[:, 0]))
        y_val = (rng.uniform(size=200) < p_val).astype(int)

        study = _make_study(
            [
                ({"C": CLIFF_C, "penalty": "l1"}, 0.65),  # degenerate best
                ({"C": CLIFF_C, "penalty": "l1"}, 0.64),  # duplicate of best
                ({"C": HEALTHY_C, "penalty": "l1"}, 0.60),  # healthy
            ]
        )
        results = _results_from_study(study)
        meta = _apply_degeneracy_guard(
            study=study,
            results=results,
            model_class=_StubModel,
            algorithm_name="LogisticRegression",
            default_hyperparameters={},
            fixed_params=dict(LR_FIXED),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            problem_type="binary_classification",
        )

        assert meta is not None
        assert meta["fallback_adopted"] is True
        assert meta["adopted_trial"] == 2
        assert results["best_params"] == {"C": HEALTHY_C, "penalty": "l1"}
        # Exactly 2 fits: the degenerate best verification + the healthy
        # fallback. The duplicate param set (trial 1) was skipped, not re-fit.
        cliff_fits = [p for p in fit_log if p.get("C") == CLIFF_C]
        assert len(cliff_fits) == 1
        assert len(fit_log) == 2

    def test_walks_past_crushed_trial_to_calibrated_trial(self, lr_data):
        """Acceptance-run iteration: the first fallback by value desc is a
        probability-crushed l2 config (still underconfident -> would fail the
        deployment slope gate); the guard must keep walking to the genuinely
        calibrated trial."""
        study = _make_study(
            [
                ({"C": CLIFF_C, "penalty": "l1"}, 0.65),  # intercept-only best
                ({"C": 0.005, "penalty": "l2"}, 0.63),  # crushed: raw slope ~11.9
                ({"C": HEALTHY_C, "penalty": "l2"}, 0.62),  # calibrated: slope ~1.27
            ]
        )
        results = _results_from_study(study)
        meta = _run_guard(study, results, lr_data)

        assert meta is not None
        assert meta["fallback_adopted"] is True
        assert meta["adopted_trial"] == 2
        assert results["best_params"] == {"C": HEALTHY_C, "penalty": "l2"}
        rejected_trials = [r["trial"] for r in meta["rejected"]]
        assert rejected_trials == [0, 1]
        assert "underconfident" in meta["rejected"][1]["reason"]

    def test_adopts_highest_verification_val_auc_not_first_by_hpo_value(self):
        """Acceptance-run iteration 2 (measured on the real persistence
        cohort): the HPO blend (0.7*AUC + 0.3*recall) ranks crushed configs
        FIRST (all-positive predictions collect the recall bonus), so
        'first non-collapsed by value desc' re-adopts the crush family
        (v2 run: trial 8 C=0.005 adopted, conformal test slope dev 0.6098,
        still blocked). Pure verification val AUC ranks them last (measured:
        crushed 0.6022/0.6037 vs healthy 0.6066). The guard must adopt the
        non-collapsed trial with the HIGHEST verification val AUC."""
        rng = np.random.default_rng(11)
        X_train = rng.normal(size=(100, 2))
        y_train = (X_train[:, 0] > 0).astype(int)
        X_val = rng.normal(size=(200, 2))
        p_true = 1.0 / (1.0 + np.exp(-X_val[:, 0]))
        y_val = (rng.uniform(size=200) < p_true).astype(int)

        class _RankStub:
            def __init__(self, **params):
                self.C = params.get("C")

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                X = np.asarray(X)
                if self.C == CLIFF_C:  # the cliff: constant probabilities
                    return np.tile([0.4, 0.6], (len(X), 1))
                if self.C == 0.5:  # weaker discriminator (noise-diluted)
                    raw = X[:, 0] + 2.0 * X[:, 1]
                else:  # C == 1.0: clean discriminator, calibrated
                    raw = X[:, 0]
                p = 1.0 / (1.0 + np.exp(-raw))
                return np.column_stack([1 - p, p])

        study = _make_study(
            [
                ({"C": CLIFF_C, "penalty": "l1"}, 0.70),  # collapsed best
                ({"C": 0.5, "penalty": "l2"}, 0.65),  # non-collapsed, weaker AUC
                ({"C": HEALTHY_C, "penalty": "l2"}, 0.60),  # non-collapsed, best AUC
            ]
        )
        results = _results_from_study(study)
        meta = _apply_degeneracy_guard(
            study=study,
            results=results,
            model_class=_RankStub,
            algorithm_name="LogisticRegression",
            default_hyperparameters={},
            fixed_params=dict(LR_FIXED),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            problem_type="binary_classification",
        )

        assert meta is not None
        assert meta["fired"] is True
        assert meta["fallback_adopted"] is True
        # NOT trial 1 (higher HPO value) — trial 2 has the higher
        # verification val AUC.
        assert meta["adopted_trial"] == 2
        assert results["best_params"] == {"C": HEALTHY_C, "penalty": "l2"}
        # Both non-collapsed candidates were considered and recorded.
        considered = {c["trial"]: c for c in meta["considered"]}
        assert set(considered) == {1, 2}
        assert considered[2]["val_auc"] > considered[1]["val_auc"]

    def test_keeps_original_best_when_all_trials_degenerate(self, lr_data):
        """Fail-closed: never fabricate — keep the honest best and flag."""
        study = _make_study(
            [
                ({"C": CLIFF_C, "penalty": "l1"}, 0.65),
                ({"C": CLIFF_C * 2, "penalty": "l1"}, 0.63),  # also below the cliff
            ]
        )
        results = _results_from_study(study)
        original_params = dict(results["best_params"])
        meta = _run_guard(study, results, lr_data)

        assert meta is not None
        assert meta["fired"] is True
        assert meta["fallback_adopted"] is False
        assert meta["all_degenerate"] is True
        # Original best kept — downstream deployment gates judge it honestly.
        assert results["best_params"] == original_params
        assert results["best_trial_number"] == 0
        assert results["best_value"] == pytest.approx(0.65)
        assert len(meta["rejected"]) == 2  # best + the one distinct fallback


# ---------------------------------------------------------------------------
# Guard conservatism: healthy / out-of-scope cases are no-ops
# ---------------------------------------------------------------------------


class TestDegeneracyGuardConservatism:
    def test_does_not_fire_on_healthy_best_params(self, lr_data):
        """Healthy best -> params, trial number and value byte-identical."""
        study = _make_study(
            [
                ({"C": HEALTHY_C, "penalty": "l1"}, 0.65),
                ({"C": 0.5, "penalty": "l2"}, 0.62),
            ]
        )
        results = _results_from_study(study)
        snapshot = {k: (dict(v) if isinstance(v, dict) else v) for k, v in results.items()}
        meta = _run_guard(study, results, lr_data)

        assert meta is not None
        assert meta["fired"] is False
        assert results == snapshot

    def test_skipped_when_promise_below_floor(self, lr_data):
        """No real discrimination promised -> collapse-vs-promise undefined."""
        study = _make_study(
            [
                ({"C": CLIFF_C, "penalty": "l1"}, DEGENERACY_GUARD_PROMISE_FLOOR - 0.01),
            ]
        )
        results = _results_from_study(study)
        snapshot = dict(results["best_params"])
        meta = _run_guard(study, results, lr_data)

        assert meta is None
        assert results["best_params"] == snapshot

    def test_skipped_for_non_binary_problem_type(self, lr_data):
        study = _make_study([({"C": CLIFF_C, "penalty": "l1"}, 0.65)])
        results = _results_from_study(study)
        meta = _run_guard(study, results, lr_data, problem_type="regression")
        assert meta is None


# ---------------------------------------------------------------------------
# Call-site wiring through tune_hyperparameters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTuneHyperparametersWiring:
    async def test_guard_adoption_flows_into_best_hyperparameters(self):
        """When the guard swaps results['best_params'], the merged
        best_hyperparameters and the hpo_degeneracy_guard output reflect it."""
        sentinel_meta = {"checked": True, "fired": True, "fallback_adopted": True}

        def _fake_guard(**kwargs):
            kwargs["results"]["best_params"] = {"n_estimators": 77}
            kwargs["results"]["best_trial_number"] = 3
            return sentinel_meta

        state = {
            "enable_hpo": True,
            "hpo_trials": 3,
            "algorithm_name": "RandomForest",
            "problem_type": "binary_classification",
            "experiment_id": "test_868",
            "default_hyperparameters": {"n_estimators": 100},
            "hyperparameter_search_space": {
                "n_estimators": {"type": "int", "low": 50, "high": 200}
            },
            "X_train_preprocessed": np.random.rand(80, 4),
            "X_validation_preprocessed": np.random.rand(30, 4),
            "train_data": {"y": np.random.randint(0, 2, 80)},
            "validation_data": {"y": np.random.randint(0, 2, 30)},
        }

        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner."
            "_apply_degeneracy_guard",
            side_effect=_fake_guard,
        ) as mock_guard:
            result = await tune_hyperparameters(state)

        assert result["hpo_completed"] is True
        mock_guard.assert_called_once()
        assert result["best_hyperparameters"]["n_estimators"] == 77
        assert result["hpo_best_trial"] == 3
        assert result["hpo_degeneracy_guard"] == sentinel_meta

    async def test_guard_exception_is_non_fatal(self):
        """A crashing guard must not break HPO (keep the study best)."""
        state = {
            "enable_hpo": True,
            "hpo_trials": 3,
            "algorithm_name": "RandomForest",
            "problem_type": "binary_classification",
            "experiment_id": "test_868_err",
            "default_hyperparameters": {"n_estimators": 100},
            "hyperparameter_search_space": {
                "n_estimators": {"type": "int", "low": 50, "high": 200}
            },
            "X_train_preprocessed": np.random.rand(80, 4),
            "X_validation_preprocessed": np.random.rand(30, 4),
            "train_data": {"y": np.random.randint(0, 2, 80)},
            "validation_data": {"y": np.random.randint(0, 2, 30)},
        }

        with patch(
            "src.agents.ml_foundation.model_trainer.nodes.hyperparameter_tuner."
            "_apply_degeneracy_guard",
            side_effect=RuntimeError("boom"),
        ):
            result = await tune_hyperparameters(state)

        assert result["hpo_completed"] is True
        assert "best_hyperparameters" in result
        assert "hpo_degeneracy_guard" not in result

    async def test_validate_hpo_output_accepts_guard_field(self):
        output = {
            "hpo_completed": True,
            "best_hyperparameters": {"n_estimators": 150},
            "hpo_best_trial": 5,
            "hpo_trials_run": 10,
            "hpo_best_value": 0.95,
            "hpo_study_name": "test_study",
            "hpo_degeneracy_guard": {"checked": True, "fired": False},
        }
        is_valid, errors = validate_hpo_output(output)
        assert is_valid is True
        assert errors == []
