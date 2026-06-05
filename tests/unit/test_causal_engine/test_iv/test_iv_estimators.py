"""Tests for IV Estimators (2SLS, LIML, Fuller)."""

import numpy as np
import pytest

from src.causal_engine.iv import (
    FullerEstimator,
    InstrumentStrength,
    IVConfig,
    IVEstimatorType,
    LIMLEstimator,
    TwoStageLSEstimator,
)


class TestTwoStageLSEstimator:
    """Tests for Two-Stage Least Squares estimator."""

    @pytest.fixture
    def estimator(self):
        """Create 2SLS estimator."""
        return TwoStageLSEstimator()

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic IV data with known true effect."""
        np.random.seed(42)
        n = 1000

        # True effect
        beta_true = 0.5

        # Instrument (exogenous)
        Z = np.random.normal(0, 1, n)

        # Unobserved confounder
        U = np.random.normal(0, 1, n)

        # Endogenous treatment (correlated with U)
        D = 0.8 * Z + 0.6 * U + np.random.normal(0, 0.5, n)

        # Outcome
        Y = beta_true * D + 0.5 * U + np.random.normal(0, 0.5, n)

        return {
            "outcome": Y,
            "treatment": D,
            "instruments": Z,
            "beta_true": beta_true,
        }

    def test_estimator_type(self, estimator):
        """Test estimator returns correct type."""
        assert estimator.estimator_type == IVEstimatorType.TWO_STAGE_LS

    def test_fit_returns_result(self, estimator, synthetic_data):
        """Test fit returns IVResult."""
        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
        )

        assert result.success is True
        assert result.coefficient is not None
        assert result.std_error is not None
        assert result.p_value is not None

    def test_coefficient_close_to_true(self, estimator, synthetic_data):
        """Test 2SLS recovers approximately correct coefficient."""
        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
        )

        # Should be within ~0.15 of true effect (0.5)
        assert abs(result.coefficient - synthetic_data["beta_true"]) < 0.15

    def test_confidence_interval_contains_true(self, estimator, synthetic_data):
        """Test CI contains true parameter."""
        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
        )

        beta_true = synthetic_data["beta_true"]
        assert result.ci_lower <= beta_true <= result.ci_upper

    def test_first_stage_f_statistic(self, estimator, synthetic_data):
        """Test first-stage F-stat is computed."""
        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
        )

        # Strong instruments should have F > 10
        assert result.diagnostics.first_stage_f_stat > 10
        assert result.diagnostics.instrument_strength == InstrumentStrength.STRONG

    def test_with_covariates(self, estimator, synthetic_data):
        """Test 2SLS works with covariates."""
        np.random.seed(42)
        n = len(synthetic_data["outcome"])
        X = np.random.normal(0, 1, (n, 2))

        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
            covariates=X,
        )

        assert result.success is True
        assert result.n_covariates == 2

    def test_multiple_instruments(self, estimator):
        """Test 2SLS with multiple instruments."""
        np.random.seed(42)
        n = 500

        # Two instruments
        Z = np.random.normal(0, 1, (n, 2))
        U = np.random.normal(0, 1, n)
        D = 0.5 * Z[:, 0] + 0.4 * Z[:, 1] + 0.5 * U + np.random.normal(0, 0.5, n)
        Y = 0.3 * D + 0.4 * U + np.random.normal(0, 0.5, n)

        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        assert result.n_instruments == 2
        # Sargan test should be computed for overidentified model
        assert result.diagnostics.sargan_stat is not None

    def test_robust_standard_errors(self):
        """Test robust standard errors option."""
        config = IVConfig(robust_std_errors=True)
        estimator = TwoStageLSEstimator(config)

        np.random.seed(42)
        n = 500
        Z = np.random.normal(0, 1, n)
        D = 0.7 * Z + np.random.normal(0, 0.5, n)
        Y = 0.4 * D + np.random.normal(0, 0.5, n)

        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        assert result.std_error is not None

    def test_weak_instrument_detection(self):
        """Test detection of weak instruments."""
        estimator = TwoStageLSEstimator()

        np.random.seed(42)
        n = 500

        # Weak instrument (small first-stage coefficient)
        Z = np.random.normal(0, 1, n)
        U = np.random.normal(0, 1, n)
        D = 0.1 * Z + 0.9 * U + np.random.normal(0, 0.5, n)  # Z barely affects D
        Y = 0.5 * D + 0.4 * U + np.random.normal(0, 0.5, n)

        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        # F-stat should be low
        assert result.diagnostics.first_stage_f_stat < 10
        assert result.diagnostics.instrument_strength in [
            InstrumentStrength.WEAK,
            InstrumentStrength.VERY_WEAK,
            InstrumentStrength.MODERATE,
        ]

    def test_invalid_input_shapes(self, estimator):
        """Test error handling for mismatched input shapes."""
        Y = np.random.normal(0, 1, 100)
        D = np.random.normal(0, 1, 50)  # Different length
        Z = np.random.normal(0, 1, 100)

        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is False
        assert result.error_message is not None


class TestLIMLEstimator:
    """Tests for Limited Information Maximum Likelihood estimator."""

    @pytest.fixture
    def estimator(self):
        """Create LIML estimator."""
        return LIMLEstimator()

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic IV data."""
        np.random.seed(123)
        n = 1000
        beta_true = 0.4

        Z = np.random.normal(0, 1, n)
        U = np.random.normal(0, 1, n)
        D = 0.7 * Z + 0.5 * U + np.random.normal(0, 0.5, n)
        Y = beta_true * D + 0.4 * U + np.random.normal(0, 0.5, n)

        return {
            "outcome": Y,
            "treatment": D,
            "instruments": Z,
            "beta_true": beta_true,
        }

    def test_estimator_type(self, estimator):
        """Test estimator returns correct type."""
        assert estimator.estimator_type == IVEstimatorType.LIML

    def test_fit_returns_result(self, estimator, synthetic_data):
        """Test LIML fit returns IVResult."""
        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
        )

        assert result.success is True
        assert result.coefficient is not None
        assert result.raw_estimate is not None
        assert "kappa" in result.raw_estimate

    def test_coefficient_close_to_true(self, estimator, synthetic_data):
        """Test LIML recovers approximately correct coefficient."""
        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
        )

        # LIML should be within ~0.2 of true effect (allowing for sampling variation)
        assert abs(result.coefficient - synthetic_data["beta_true"]) < 0.20

    def test_liml_kappa_just_identified_strong_is_one(self, estimator, synthetic_data):
        """Just-identified strong instruments → LIML κ ≈ 1.0 (LIML ≈ 2SLS).

        C2 regression: the generalized-eigenvalue arguments were swapped
        (``eigvalsh(WtMzW, WtM0W)``), so ``min`` returned a canonical-correlation
        quantity ≈0.70 instead of the k-class κ. The LIML κ is the smallest root
        of ``det(W'M0 W − κ W'MZ W)=0`` = ``min eigvalsh(WtM0W, WtMzW)``, which is
        ≥ 1 (=1 exactly-identified). The old loose ``0 < κ < 2`` assertion on this
        just-identified, near-exogenous DGP could not catch the inversion.
        """
        result = estimator.fit(
            outcome=synthetic_data["outcome"],
            treatment=synthetic_data["treatment"],
            instruments=synthetic_data["instruments"],
        )
        kappa = result.raw_estimate["kappa"]
        assert kappa == pytest.approx(1.0, abs=0.02), (
            f"just-identified κ must be ≈1.0 (LIML≈2SLS), got {kappa}"
        )

    def test_liml_kappa_over_identified_endogenous_at_least_one(self, estimator):
        """Over-identified (k≥3), genuine endogeneity → LIML κ strictly > 1.

        C2: with over-identifying instruments the smallest root of the LIML
        pencil exceeds 1. RED with the swapped arguments: ≈0.39.
        """
        np.random.seed(7)
        n = 4000
        # 3 instruments (over-identified for a single endogenous treatment)
        Z = np.random.normal(0, 1, (n, 3))
        U = np.random.normal(0, 1, n)  # unobserved confounder → endogeneity
        D = Z @ np.array([0.6, 0.5, 0.4]) + 0.7 * U + np.random.normal(0, 0.5, n)
        Y = 0.5 * D + 0.8 * U + np.random.normal(0, 0.5, n)

        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)
        kappa = result.raw_estimate["kappa"]
        assert kappa >= 1.0 - 1e-6, f"over-identified κ must be ≥ 1, got {kappa}"
        assert kappa > 1.0, f"over-identified κ should be strictly > 1, got {kappa}"

    def test_fuller_modification(self):
        """Test Fuller modification reduces bias."""
        config = IVConfig(fuller_k=1.0)
        liml = LIMLEstimator(config)

        np.random.seed(42)
        n = 500
        Z = np.random.normal(0, 1, n)
        U = np.random.normal(0, 1, n)
        D = 0.6 * Z + 0.5 * U + np.random.normal(0, 0.5, n)
        Y = 0.3 * D + 0.4 * U + np.random.normal(0, 0.5, n)

        result = liml.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        assert result.raw_estimate["fuller_k"] == 1.0

    def test_liml_vs_2sls_similar_strong_instruments(self):
        """Test LIML and 2SLS give similar results with strong instruments."""
        np.random.seed(42)
        n = 1000

        Z = np.random.normal(0, 1, n)
        D = 0.8 * Z + np.random.normal(0, 0.3, n)  # Strong first stage
        Y = 0.5 * D + np.random.normal(0, 0.5, n)

        liml = LIMLEstimator()
        tsls = TwoStageLSEstimator()

        result_liml = liml.fit(outcome=Y, treatment=D, instruments=Z)
        result_tsls = tsls.fit(outcome=Y, treatment=D, instruments=Z)

        # Coefficients should be very close
        assert abs(result_liml.coefficient - result_tsls.coefficient) < 0.05


class TestFullerEstimator:
    """Tests for Fuller estimator."""

    def test_estimator_type(self):
        """Test Fuller estimator type."""
        estimator = FullerEstimator()
        assert estimator.estimator_type == IVEstimatorType.FULLER

    def test_default_fuller_k(self):
        """Test default Fuller k = 1."""
        estimator = FullerEstimator()
        assert estimator.config.fuller_k == 1.0

    def test_custom_fuller_k(self):
        """Test custom Fuller k parameter."""
        estimator = FullerEstimator(fuller_k=4.0)
        assert estimator.config.fuller_k == 4.0

    def test_fuller_fit(self):
        """Test Fuller estimator fits data."""
        np.random.seed(42)
        n = 500

        Z = np.random.normal(0, 1, n)
        D = 0.7 * Z + np.random.normal(0, 0.4, n)
        Y = 0.4 * D + np.random.normal(0, 0.5, n)

        estimator = FullerEstimator(fuller_k=1.0)
        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        assert result.coefficient is not None


class TestIVConfig:
    """Tests for IV configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IVConfig()

        assert config.confidence_level == 0.95
        assert config.robust_std_errors is True
        assert config.weak_iv_robust is True
        assert config.weak_iv_threshold == 10.0
        assert config.fuller_k is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = IVConfig(
            confidence_level=0.90,
            robust_std_errors=False,
            fuller_k=1.0,
            bootstrap_iterations=100,
        )

        assert config.confidence_level == 0.90
        assert config.robust_std_errors is False
        assert config.fuller_k == 1.0
        assert config.bootstrap_iterations == 100


class TestIVEdgeCases:
    """Test edge cases for IV estimation."""

    def test_perfect_instrument(self):
        """Test with perfect instrument (no first-stage error)."""
        np.random.seed(42)
        n = 500

        Z = np.random.normal(0, 1, n)
        D = Z  # Perfect instrument
        Y = 0.5 * D + np.random.normal(0, 0.5, n)

        estimator = TwoStageLSEstimator()
        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        assert result.diagnostics.first_stage_f_stat > 100

    def test_small_sample(self):
        """Test with small sample size."""
        np.random.seed(42)
        n = 50  # Small sample

        Z = np.random.normal(0, 1, n)
        D = 0.7 * Z + np.random.normal(0, 0.4, n)
        Y = 0.4 * D + np.random.normal(0, 0.5, n)

        estimator = TwoStageLSEstimator()
        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        # SE should be larger with small sample
        assert result.std_error > 0

    def test_many_instruments(self):
        """Test with many instruments (potential weak instrument bias)."""
        np.random.seed(42)
        n = 500
        k = 10  # Many instruments

        Z = np.random.normal(0, 1, (n, k))
        # Only first few instruments are relevant
        D = 0.3 * Z[:, 0] + 0.2 * Z[:, 1] + np.random.normal(0, 0.5, n)
        Y = 0.4 * D + np.random.normal(0, 0.5, n)

        estimator = TwoStageLSEstimator()
        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        assert result.success is True
        assert result.n_instruments == k


class TestHausmanEndogeneity:
    """H4 — Hausman endogeneity test must use COEFFICIENT variances.

    The test statistic is (β_IV − β_OLS)² / (Var(β_IV) − Var(β_OLS)) where the
    variances are of the coefficient estimates (σ̂²·[(W'W)⁻¹]₀₀, O(1/n)). The
    buggy code used the residual variances σ̂² (O(1), outcome-scale); since
    σ̂²_IV ≈ σ̂²_OLS the denominator was tiny/sign-unstable and the
    ``var_diff ≤ 0 → stat=0, p=1.0`` branch silently declared an endogenous
    treatment EXOGENOUS.
    """

    def test_rejects_strong_endogeneity(self):
        est = TwoStageLSEstimator()
        np.random.seed(3)
        n = 4000
        Z = np.random.normal(0, 1, n)
        U = np.random.normal(0, 1, n)  # unobserved confounder
        D = 0.8 * Z + 0.9 * U + np.random.normal(0, 0.3, n)  # endogenous
        Y = 0.5 * D + 1.0 * U + np.random.normal(0, 0.5, n)
        res = est._hausman_test(Y, D, Z, None, n, 1, 0)
        stat, pvalue = res[0], res[1]
        assert stat > 3.84, f"strong endogeneity must yield a large Hausman stat, got {stat}"
        assert pvalue < 0.01, f"strong endogeneity must reject exogeneity, got p={pvalue}"

    def test_does_not_reject_exogenous(self):
        est = TwoStageLSEstimator()
        np.random.seed(11)
        n = 4000
        Z = np.random.normal(0, 1, n)
        D = 0.8 * Z + np.random.normal(0, 0.5, n)  # no shared confounder
        Y = 0.5 * D + np.random.normal(0, 0.5, n)
        res = est._hausman_test(Y, D, Z, None, n, 1, 0)
        pvalue = res[1]
        assert pvalue > 0.05, f"exogenous DGP must fail to reject, got p={pvalue}"

    def test_surfaces_validity_flag(self):
        est = TwoStageLSEstimator()
        np.random.seed(3)
        n = 2000
        Z = np.random.normal(0, 1, n)
        U = np.random.normal(0, 1, n)
        D = 0.8 * Z + 0.7 * U + np.random.normal(0, 0.4, n)
        Y = 0.5 * D + 0.8 * U + np.random.normal(0, 0.5, n)
        res = est._hausman_test(Y, D, Z, None, n, 1, 0)
        assert len(res) == 3, "Hausman must surface a validity flag (stat, pvalue, valid)"
        assert res[2] is True, "a well-conditioned Hausman test is valid"
