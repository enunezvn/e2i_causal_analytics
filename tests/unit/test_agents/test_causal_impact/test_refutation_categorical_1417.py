"""#1417: refutation reconstruction must survive categorical confounders.

The estimation node one-hot encodes categorical covariates before fitting
(``_encode_categorical_covariates``), so the reported ATE comes from the
encoded design matrix. The DoWhy reconstruction in the refutation node used
to pass the RAW passthrough frame with RAW categorical names as
``common_causes``/``effect_modifiers``; DoWhy internally dummifies its copy
(dropping the raw categorical columns) and its EconML wrapper then selects
effect modifiers by the raw names — ``KeyError: "['delivery_channel',
'trigger_type', 'priority'] not in index"`` live (deployed 0c3d75fa,
2026-07-31), zero causal_validations rows, turn fails closed.

Numeric-only fixtures elsewhere in this suite can never trip this, hence
these tests. Live repro + fix-shape validation ran in the deployed container
before this file was written (issue #1417).
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.refutation import (
    RefutationError,
    _effective_reconstruction_common_causes,
    _reconstruct_dowhy_artifacts,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _make_categorical_estimation_data(
    n: int = 400,
    seed: int = 42,
    true_ate: float = 0.5,
    with_identifier: bool = False,
) -> pd.DataFrame:
    """Frame with one categorical + one numeric confounder (raw, un-encoded).

    Mirrors the live substrate shape: the #1351 resolver binds string driver
    columns (trigger_type, delivery_channel, ...) into the adjustment set,
    and the passthrough frame reaches refutation UN-encoded.
    """
    rng = np.random.default_rng(seed)
    channel = rng.choice(["email", "call", "event"], n)
    num1 = rng.normal(0, 1, n)
    treatment = rng.binomial(1, 0.5, n).astype(float)
    outcome = (
        true_ate * treatment
        + 0.2 * (channel == "email").astype(float)
        + 0.1 * num1
        + rng.normal(0, 0.3, n)
    )
    frame = {
        "delivery_channel": channel,
        "lead_time_days": num1,
        "accepted": treatment,
        "converted": outcome,
    }
    if with_identifier:
        # High-cardinality identifier in the PASSTHROUGH but NOT in the
        # adjustment set — encoding the whole frame (instead of only the
        # common-cause columns) would trip the cardinality guard on it.
        frame["hcp_id"] = [f"hcp_{i:05d}" for i in range(n)]
    return pd.DataFrame(frame)


def _estimation_result(ate: float = 0.5) -> dict:
    return {
        "method": "LinearDML",
        "ate": ate,
        "ate_ci_lower": ate - 0.1,
        "ate_ci_upper": ate + 0.1,
        "effect_size": "medium",
        "statistical_significance": True,
        "p_value": 0.01,
        "sample_size": 400,
        "covariates_adjusted": ["delivery_channel", "lead_time_days"],
        "heterogeneity_detected": False,
    }


class TestCategoricalReconstruction:
    """#1417 red-first: raw categorical names crashed DoWhy's EconML path."""

    def test_categorical_confounder_reconstruction_succeeds(self):
        data = _make_categorical_estimation_data()
        model, estimand, estimate = _reconstruct_dowhy_artifacts(
            data=data,
            treatment="accepted",
            outcome="converted",
            common_causes=["delivery_channel", "lead_time_days"],
            estimation_result=_estimation_result(ate=0.5),
        )
        assert model is not None
        assert estimand is not None
        # Tolerance guard inside the reconstruction already validated the
        # ATE; assert the estimate is numeric and in the right neighborhood.
        assert abs(float(estimate.value) - 0.5) <= max(0.5 * 0.20, 0.10)

    def test_passthrough_identifier_outside_adjustment_set_is_not_encoded(self):
        """Only the adjustment-set columns may be encoded.

        A whole-frame encode would raise the cardinality guard on hcp_id
        (400 uniques > 50) even though it is NOT a confounder here.
        """
        data = _make_categorical_estimation_data(with_identifier=True)
        model, _estimand, estimate = _reconstruct_dowhy_artifacts(
            data=data,
            treatment="accepted",
            outcome="converted",
            common_causes=["delivery_channel", "lead_time_days"],
            estimation_result=_estimation_result(ate=0.5),
        )
        assert model is not None
        assert abs(float(estimate.value) - 0.5) <= max(0.5 * 0.20, 0.10)

    def test_identifier_inside_adjustment_set_still_fails_closed(self):
        """The cardinality guard must still fire when the identifier IS in
        the adjustment set — encoding it would explode the design matrix."""
        data = _make_categorical_estimation_data(with_identifier=True)
        with pytest.raises(RefutationError):
            _reconstruct_dowhy_artifacts(
                data=data,
                treatment="accepted",
                outcome="converted",
                common_causes=["delivery_channel", "lead_time_days", "hcp_id"],
                estimation_result=_estimation_result(ate=0.5),
            )

    def test_efficiency_run_categorical_baselines_reconstruct(self):
        """Codex #1417 residual: an efficiency run threads CATEGORICAL
        baselines through ``_effective_reconstruction_common_causes`` into the
        reconstruction — those must be encoded exactly like ordinary
        confounders."""
        data = _make_categorical_estimation_data()
        est = _estimation_result(ate=0.5)
        est["adjustment_type"] = "efficiency"
        est["covariates_adjusted"] = []
        est["baseline_covariates_adjusted"] = ["delivery_channel", "lead_time_days"]
        effective = _effective_reconstruction_common_causes([], est)
        assert effective == ["delivery_channel", "lead_time_days"]
        model, _estimand, estimate = _reconstruct_dowhy_artifacts(
            data=data,
            treatment="accepted",
            outcome="converted",
            common_causes=effective,
            estimation_result=est,
        )
        assert model is not None
        assert abs(float(estimate.value) - 0.5) <= max(0.5 * 0.20, 0.10)

    def test_ate_tolerance_guard_still_enforced_on_encoded_frame(self):
        """The fix must not bypass the reconstructed-vs-reported guard."""
        data = _make_categorical_estimation_data()
        with pytest.raises(RefutationError) as excinfo:
            _reconstruct_dowhy_artifacts(
                data=data,
                treatment="accepted",
                outcome="converted",
                common_causes=["delivery_channel", "lead_time_days"],
                estimation_result=_estimation_result(ate=5.0),
            )
        assert excinfo.value.details.get("reason") == "reconstructed_ate_mismatch"
