"""
Instrumental Variable Diagnostic Tests

Comprehensive suite of diagnostic tests for IV estimation:
- Weak instrument tests (Cragg-Donald, Kleibergen-Paap)
- Endogeneity tests (Hausman, Durbin-Wu-Hausman)
- Overidentification tests (Sargan, Hansen J)
- Instrument relevance tests (partial R², partial F)

References:
    - Stock & Yogo (2005) "Testing for Weak Instruments"
    - Kleibergen & Paap (2006) "Generalized reduced rank tests"
    - Baum, Schaffer & Stillman (2007) "Enhanced routines for IV/GMM estimation"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class WeakInstrumentTest:
    """Results from weak instrument tests."""

    test_name: str
    statistic: float
    critical_values: dict[str, float]  # e.g., {"10%": 16.38, "15%": 8.96}
    is_weak: bool
    message: str


@dataclass
class OveridentificationTest:
    """Results from overidentification tests."""

    test_name: str
    statistic: float
    df: int
    p_value: float
    rejects_validity: bool
    message: str


@dataclass
class EndogeneityTest:
    """Results from endogeneity tests."""

    test_name: str
    statistic: float
    df: int
    p_value: float
    is_endogenous: bool
    message: str


@dataclass
class IVDiagnosticReport:
    """Complete diagnostic report for IV estimation."""

    # Weak instrument tests
    cragg_donald: Optional[WeakInstrumentTest] = None
    kleibergen_paap: Optional[WeakInstrumentTest] = None

    # Overidentification tests
    sargan: Optional[OveridentificationTest] = None
    hansen_j: Optional[OveridentificationTest] = None

    # Endogeneity tests
    hausman: Optional[EndogeneityTest] = None
    durbin_wu_hausman: Optional[EndogeneityTest] = None

    # Summary
    recommendation: str = ""

    def is_valid_estimation(self, alpha: float = 0.05) -> bool:
        """Check if IV estimation appears valid."""
        # Check weak instruments
        if self.cragg_donald and self.cragg_donald.is_weak:
            return False
        if self.kleibergen_paap and self.kleibergen_paap.is_weak:
            return False

        # Check overidentification
        if self.sargan and self.sargan.rejects_validity:
            return False
        if self.hansen_j and self.hansen_j.rejects_validity:
            return False

        return True

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"recommendation": self.recommendation}

        if self.cragg_donald:
            result["cragg_donald"] = {
                "statistic": self.cragg_donald.statistic,
                "is_weak": self.cragg_donald.is_weak,
                "critical_values": self.cragg_donald.critical_values,
            }

        if self.sargan:
            result["sargan"] = {
                "statistic": self.sargan.statistic,
                "p_value": self.sargan.p_value,
                "rejects_validity": self.sargan.rejects_validity,
            }

        if self.hausman:
            result["hausman"] = {
                "statistic": self.hausman.statistic,
                "p_value": self.hausman.p_value,
                "is_endogenous": self.hausman.is_endogenous,
            }

        return result


def cragg_donald_test(
    treatment: NDArray[np.float64],
    instruments: NDArray[np.float64],
    covariates: Optional[NDArray[np.float64]] = None,
) -> WeakInstrumentTest:
    """
    Cragg-Donald F-statistic for weak instrument detection.

    This is the minimum eigenvalue of the concentration matrix,
    scaled as an F-statistic. Compare to Stock-Yogo critical values.

    Args:
        treatment: Endogenous variable (n,)
        instruments: Instruments (n, k)
        covariates: Exogenous controls (n, p) or None

    Returns:
        WeakInstrumentTest with Cragg-Donald statistic
    """
    n = len(treatment)
    D = treatment.flatten()
    Z = instruments.reshape(-1, 1) if instruments.ndim == 1 else instruments
    k = Z.shape[1]

    # Partial out covariates if present
    if covariates is not None:
        X = covariates.reshape(-1, 1) if covariates.ndim == 1 else covariates
        p = X.shape[1]

        # Residual maker for X
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        M_X = np.eye(n) - X @ XtX_inv @ X.T
        D_tilde = M_X @ D
        Z_tilde = M_X @ Z
    else:
        p = 0
        D_tilde = D - np.mean(D)
        Z_tilde = Z - np.mean(Z, axis=0)

    # First-stage regression: D on Z
    coef = np.linalg.lstsq(Z_tilde, D_tilde, rcond=None)[0]
    D_hat = Z_tilde @ coef
    residuals = D_tilde - D_hat

    # Cragg-Donald statistic (for single endogenous variable = first-stage F)
    ss_reg = np.sum((D_hat - np.mean(D_tilde)) ** 2)
    ss_res = np.sum(residuals**2)

    df1 = k
    df2 = n - k - p - 1

    if ss_res > 0 and df2 > 0:
        cd_stat = (ss_reg / df1) / (ss_res / df2)
    else:
        cd_stat = 0.0

    # Stock-Yogo critical values for 2SLS, single endogenous variable
    # Based on 10%, 15%, 20%, 25% maximal IV size
    critical_values = {
        "10%": 16.38,
        "15%": 8.96,
        "20%": 6.66,
        "25%": 5.53,
    }

    is_weak = cd_stat < critical_values["10%"]

    if is_weak:
        message = f"Cragg-Donald F = {cd_stat:.2f} < 16.38: WEAK instruments detected"
    else:
        message = f"Cragg-Donald F = {cd_stat:.2f} >= 16.38: Instruments appear strong"

    return WeakInstrumentTest(
        test_name="Cragg-Donald",
        statistic=float(cd_stat),
        critical_values=critical_values,
        is_weak=is_weak,
        message=message,
    )


def partial_r_squared(
    treatment: NDArray[np.float64],
    instruments: NDArray[np.float64],
    covariates: Optional[NDArray[np.float64]] = None,
) -> float:
    """
    Compute partial R-squared of instruments in first-stage regression.

    Measures the incremental explanatory power of instruments
    after controlling for covariates.

    Args:
        treatment: Endogenous variable (n,)
        instruments: Instruments (n, k)
        covariates: Exogenous controls (n, p) or None

    Returns:
        Partial R-squared value [0, 1]
    """
    len(treatment)
    D = treatment.flatten()
    Z = instruments.reshape(-1, 1) if instruments.ndim == 1 else instruments

    if covariates is not None:
        X = covariates.reshape(-1, 1) if covariates.ndim == 1 else covariates

        # R² from full model (Z + X)
        full = np.column_stack([Z, X])
        coef_full = np.linalg.lstsq(full, D, rcond=None)[0]
        D_hat_full = full @ coef_full
        ss_res_full = np.sum((D - D_hat_full) ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0.0

        # R² from restricted model (X only)
        coef_x = np.linalg.lstsq(X, D, rcond=None)[0]
        D_hat_x = X @ coef_x
        ss_res_x = np.sum((D - D_hat_x) ** 2)
        r2_x = 1 - ss_res_x / ss_tot if ss_tot > 0 else 0.0

        # Partial R²
        partial_r2 = (r2_full - r2_x) / (1 - r2_x) if r2_x < 1 else 0.0
    else:
        # Simple R² of instruments
        coef = np.linalg.lstsq(Z, D, rcond=None)[0]
        D_hat = Z @ coef
        ss_res = np.sum((D - D_hat) ** 2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        partial_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(max(0.0, min(1.0, partial_r2)))


def sargan_test(
    residuals: NDArray[np.float64],
    instruments: NDArray[np.float64],
    n_endogenous: int = 1,
) -> OveridentificationTest:
    """
    Sargan test for overidentifying restrictions.

    Under the null hypothesis that instruments are valid,
    the test statistic follows chi-squared(k - g) where
    k = number of instruments, g = number of endogenous variables.

    Args:
        residuals: Structural equation residuals (n,)
        instruments: Instruments (n, k)
        n_endogenous: Number of endogenous variables

    Returns:
        OveridentificationTest with Sargan statistic
    """
    n = len(residuals)
    e = residuals.flatten()
    Z = instruments.reshape(-1, 1) if instruments.ndim == 1 else instruments
    k = Z.shape[1]

    # Degrees of freedom (overidentification)
    df = k - n_endogenous

    if df <= 0:
        return OveridentificationTest(
            test_name="Sargan",
            statistic=0.0,
            df=0,
            p_value=1.0,
            rejects_validity=False,
            message="Model is exactly identified - Sargan test not applicable",
        )

    # Sargan statistic: n * R² from regressing residuals on instruments
    try:
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        ZtZ_inv = np.linalg.pinv(Z.T @ Z)

    # Project residuals onto instrument space
    P_Z = Z @ ZtZ_inv @ Z.T
    e_hat = P_Z @ e

    ss_reg = np.sum(e_hat**2)
    ss_tot = np.sum(e**2)

    r_squared = ss_reg / ss_tot if ss_tot > 0 else 0.0
    sargan_stat = n * r_squared

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(sargan_stat, df)
    rejects = p_value < 0.05

    if rejects:
        message = (
            f"Sargan χ²({df}) = {sargan_stat:.2f}, p = {p_value:.4f}: REJECTS instrument validity"
        )
    else:
        message = f"Sargan χ²({df}) = {sargan_stat:.2f}, p = {p_value:.4f}: Cannot reject validity"

    return OveridentificationTest(
        test_name="Sargan",
        statistic=float(sargan_stat),
        df=df,
        p_value=float(p_value),
        rejects_validity=rejects,
        message=message,
    )


def anderson_rubin_test(
    outcome: NDArray[np.float64],
    treatment: NDArray[np.float64],
    instruments: NDArray[np.float64],
    beta_null: float = 0.0,
    covariates: Optional[NDArray[np.float64]] = None,
) -> EndogeneityTest:
    """
    Anderson-Rubin test for coefficient significance.

    This test is robust to weak instruments and tests H0: β = β_null.
    Under the null, the statistic follows F(k, n-k-p).

    Args:
        outcome: Dependent variable (n,)
        treatment: Endogenous variable (n,)
        instruments: Instruments (n, k)
        beta_null: Null hypothesis value for β (default 0)
        covariates: Exogenous controls (n, p) or None

    Returns:
        EndogeneityTest with Anderson-Rubin statistic
    """
    n = len(outcome)
    Y = outcome.flatten()
    D = treatment.flatten()
    Z = instruments.reshape(-1, 1) if instruments.ndim == 1 else instruments
    k = Z.shape[1]

    # Create Y - β_null * D
    Y_adj = Y - beta_null * D

    # Partial out covariates
    if covariates is not None:
        X = covariates.reshape(-1, 1) if covariates.ndim == 1 else covariates
        p = X.shape[1]

        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        M_X = np.eye(n) - X @ XtX_inv @ X.T
        Y_adj_tilde = M_X @ Y_adj
        Z_tilde = M_X @ Z
    else:
        p = 0
        Y_adj_tilde = Y_adj - np.mean(Y_adj)
        Z_tilde = Z - np.mean(Z, axis=0)

    # Regress Y_adj on Z
    coef = np.linalg.lstsq(Z_tilde, Y_adj_tilde, rcond=None)[0]
    Y_hat = Z_tilde @ coef
    residuals = Y_adj_tilde - Y_hat

    # F-statistic
    ss_reg = np.sum((Y_hat - np.mean(Y_adj_tilde)) ** 2)
    ss_res = np.sum(residuals**2)

    df1 = k
    df2 = n - k - p - 1

    if ss_res > 0 and df2 > 0:
        ar_stat = (ss_reg / df1) / (ss_res / df2)
        p_value = 1 - stats.f.cdf(ar_stat, df1, df2)
    else:
        ar_stat = 0.0
        p_value = 1.0

    is_significant = p_value < 0.05

    if is_significant:
        message = f"AR F({df1},{df2}) = {ar_stat:.2f}, p = {p_value:.4f}: REJECTS β = {beta_null}"
    else:
        message = (
            f"AR F({df1},{df2}) = {ar_stat:.2f}, p = {p_value:.4f}: Cannot reject β = {beta_null}"
        )

    return EndogeneityTest(
        test_name="Anderson-Rubin",
        statistic=float(ar_stat),
        df=df1,
        p_value=float(p_value),
        is_endogenous=is_significant,  # In this context, tests if β ≠ 0
        message=message,
    )


def durbin_wu_hausman_test(
    outcome: NDArray[np.float64],
    treatment: NDArray[np.float64],
    instruments: NDArray[np.float64],
    covariates: Optional[NDArray[np.float64]] = None,
) -> EndogeneityTest:
    """
    Durbin-Wu-Hausman test for endogeneity.

    Tests whether OLS and IV estimates are significantly different.
    Rejection implies treatment is endogenous and IV is needed.

    Procedure:
    1. Regress D on Z (and X) to get first-stage residuals ν̂
    2. Include ν̂ in OLS regression: Y = D*β + ν̂*ρ + X*γ + ε
    3. Test H0: ρ = 0

    Args:
        outcome: Dependent variable (n,)
        treatment: Endogenous variable (n,)
        instruments: Instruments (n, k)
        covariates: Exogenous controls (n, p) or None

    Returns:
        EndogeneityTest with DWH statistic
    """
    n = len(outcome)
    Y = outcome.flatten()
    D = treatment.flatten()
    Z = instruments.reshape(-1, 1) if instruments.ndim == 1 else instruments
    Z.shape[1]

    # Step 1: First-stage regression to get residuals
    if covariates is not None:
        X = covariates.reshape(-1, 1) if covariates.ndim == 1 else covariates
        X.shape[1]
        first_stage = np.column_stack([Z, X])
    else:
        first_stage = Z

    pi = np.linalg.lstsq(first_stage, D, rcond=None)[0]
    D_hat = first_stage @ pi
    nu_hat = D - D_hat  # First-stage residuals

    # Step 2: Augmented OLS regression
    if covariates is not None:
        aug_design = np.column_stack([D, nu_hat, X, np.ones(n)])
    else:
        aug_design = np.column_stack([D, nu_hat, np.ones(n)])

    aug_coef = np.linalg.lstsq(aug_design, Y, rcond=None)[0]
    rho = aug_coef[1]  # Coefficient on first-stage residuals

    # Compute standard error of ρ
    Y_hat = aug_design @ aug_coef
    residuals = Y - Y_hat
    sigma_sq = np.sum(residuals**2) / (n - aug_design.shape[1])

    try:
        var_coef = sigma_sq * np.linalg.inv(aug_design.T @ aug_design)
        se_rho = np.sqrt(var_coef[1, 1])
    except np.linalg.LinAlgError:
        var_coef = sigma_sq * np.linalg.pinv(aug_design.T @ aug_design)
        se_rho = np.sqrt(var_coef[1, 1])

    # t-test for ρ = 0
    df = n - aug_design.shape[1]
    if se_rho > 0:
        t_stat = rho / se_rho
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    else:
        t_stat = 0.0
        p_value = 1.0

    # Chi-squared statistic (t² ~ χ²(1) asymptotically)
    dwh_stat = t_stat**2
    is_endogenous = p_value < 0.05

    if is_endogenous:
        message = f"DWH χ²(1) = {dwh_stat:.2f}, p = {p_value:.4f}: Treatment is ENDOGENOUS (use IV)"
    else:
        message = f"DWH χ²(1) = {dwh_stat:.2f}, p = {p_value:.4f}: Cannot reject exogeneity (OLS may be OK)"

    return EndogeneityTest(
        test_name="Durbin-Wu-Hausman",
        statistic=float(dwh_stat),
        df=1,
        p_value=float(p_value),
        is_endogenous=is_endogenous,
        message=message,
    )


def run_all_diagnostics(
    outcome: NDArray[np.float64],
    treatment: NDArray[np.float64],
    instruments: NDArray[np.float64],
    residuals: NDArray[np.float64],
    covariates: Optional[NDArray[np.float64]] = None,
) -> IVDiagnosticReport:
    """
    Run comprehensive IV diagnostic tests.

    Args:
        outcome: Dependent variable (n,)
        treatment: Endogenous variable (n,)
        instruments: Instruments (n, k)
        residuals: Structural equation residuals (n,)
        covariates: Exogenous controls (n, p) or None

    Returns:
        IVDiagnosticReport with all test results
    """
    report = IVDiagnosticReport()

    # Weak instrument test
    try:
        report.cragg_donald = cragg_donald_test(treatment, instruments, covariates)
    except Exception as e:
        logger.warning(f"Cragg-Donald test failed: {e}")

    # Overidentification test
    try:
        report.sargan = sargan_test(residuals, instruments)
    except Exception as e:
        logger.warning(f"Sargan test failed: {e}")

    # Endogeneity tests
    try:
        report.durbin_wu_hausman = durbin_wu_hausman_test(
            outcome, treatment, instruments, covariates
        )
    except Exception as e:
        logger.warning(f"DWH test failed: {e}")

    # Generate recommendation
    recommendations = []

    if report.cragg_donald and report.cragg_donald.is_weak:
        recommendations.append("WEAK INSTRUMENTS: Consider LIML or Fuller estimator")

    if report.sargan and report.sargan.rejects_validity:
        recommendations.append("OVERID REJECTED: Some instruments may be invalid")

    if report.durbin_wu_hausman:
        if report.durbin_wu_hausman.is_endogenous:
            recommendations.append("ENDOGENEITY CONFIRMED: IV estimation is appropriate")
        else:
            recommendations.append("NO ENDOGENEITY: OLS may be consistent")

    if not recommendations:
        recommendations.append("IV estimation appears valid")

    report.recommendation = "; ".join(recommendations)

    return report


def stock_yogo_critical_values(
    n_instruments: int,
    n_endogenous: int = 1,
    bias_level: float = 0.10,
) -> float:
    """
    Look up Stock-Yogo critical values for weak instrument test.

    Critical values for 2SLS bias. If F < critical value,
    then IV bias could exceed bias_level * OLS bias.

    Args:
        n_instruments: Number of instruments (k)
        n_endogenous: Number of endogenous variables (default 1)
        bias_level: Acceptable bias level (0.05, 0.10, 0.20, 0.30)

    Returns:
        Critical value for weak instrument test
    """
    # Stock-Yogo (2005) Table 5.2 - 2SLS bias
    # For single endogenous variable, various instrument counts
    critical_values_2sls = {
        # k: {bias: critical_value}
        1: {0.05: 13.91, 0.10: 9.08, 0.20: 6.46, 0.30: 5.39},
        2: {0.05: 16.85, 0.10: 10.27, 0.20: 6.71, 0.30: 5.34},
        3: {0.05: 22.30, 0.10: 13.43, 0.20: 8.68, 0.30: 6.84},
        4: {0.05: 24.58, 0.10: 14.31, 0.20: 9.02, 0.30: 7.02},
        5: {0.05: 26.87, 0.10: 15.09, 0.20: 9.31, 0.30: 7.16},
        6: {0.05: 28.83, 0.10: 15.72, 0.20: 9.54, 0.30: 7.27},
        7: {0.05: 30.53, 0.10: 16.23, 0.20: 9.73, 0.30: 7.35},
        8: {0.05: 32.02, 0.10: 16.66, 0.20: 9.88, 0.30: 7.42},
    }

    if n_endogenous != 1:
        # For multiple endogenous variables, use conservative approximation
        logger.warning(
            f"Stock-Yogo tables for {n_endogenous} endogenous vars not implemented. "
            "Using conservative single-variable values."
        )

    # Clamp instrument count to available values
    k = min(max(n_instruments, 1), 8)

    # Find closest bias level
    available_biases = [0.05, 0.10, 0.20, 0.30]
    closest_bias = min(available_biases, key=lambda x: abs(x - bias_level))

    return critical_values_2sls[k][closest_bias]
