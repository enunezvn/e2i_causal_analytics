"""
Two-Stage Least Squares (2SLS) Estimator

Implements 2SLS for instrumental variable estimation:
1. First stage: Regress treatment D on instruments Z and covariates X
2. Second stage: Regress outcome Y on predicted treatment D̂ and covariates X

The 2SLS estimator is consistent under:
- Relevance: Cov(Z, D) ≠ 0 (instruments predict treatment)
- Exclusion: Cov(Z, ε) = 0 (instruments only affect Y through D)

Reference:
    Angrist & Pischke (2009) "Mostly Harmless Econometrics", Chapter 4
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .base import (
    BaseIVEstimator,
    IVDiagnostics,
    IVEstimatorType,
    IVResult,
)

logger = logging.getLogger(__name__)


class TwoStageLSEstimator(BaseIVEstimator):
    """
    Two-Stage Least Squares (2SLS) IV Estimator.

    Usage:
        estimator = TwoStageLSEstimator()
        result = estimator.fit(
            outcome=Y,
            treatment=D,
            instruments=Z,
            covariates=X
        )
        print(f"LATE: {result.coefficient:.4f} ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
    """

    @property
    def estimator_type(self) -> IVEstimatorType:
        return IVEstimatorType.TWO_STAGE_LS

    def fit(
        self,
        outcome: NDArray[np.float64],
        treatment: NDArray[np.float64],
        instruments: NDArray[np.float64],
        covariates: Optional[NDArray[np.float64]] = None,
        **kwargs,
    ) -> IVResult:
        """
        Fit 2SLS estimator.

        Args:
            outcome: Y - observed outcomes (n,)
            treatment: D - endogenous treatment (n,)
            instruments: Z - instruments (n, k)
            covariates: X - exogenous controls (n, p) or None

        Returns:
            IVResult with 2SLS estimates
        """
        start_time = time.perf_counter()

        try:
            self._validate_inputs(outcome, treatment, instruments, covariates)

            # Ensure proper shapes
            Y = outcome.flatten()
            D = treatment.flatten()
            Z = instruments.reshape(-1, 1) if instruments.ndim == 1 else instruments
            n = len(Y)
            k = Z.shape[1]  # Number of instruments

            # Build regressor matrix: [Z, X, constant]
            if covariates is not None:
                X = covariates.reshape(-1, 1) if covariates.ndim == 1 else covariates
                W = np.column_stack([Z, X, np.ones(n)])  # First stage regressors
                p = X.shape[1]
            else:
                X = None
                W = np.column_stack([Z, np.ones(n)])
                p = 0

            # ============ First Stage ============
            # Regress D on Z (and X)
            # D = Z*π + X*γ + ν
            first_stage_result = self._first_stage(D, W, n, k)
            D_hat = first_stage_result["D_hat"]
            first_stage_coef = first_stage_result["coef"]
            first_stage_result["residuals"]
            first_stage_r_squared = first_stage_result["r_squared"]

            # First-stage F-statistic for instrument relevance
            f_stat, f_pvalue = self._first_stage_f_test(D, Z, X, n, k, p)

            # ============ Second Stage ============
            # Regress Y on D̂ (and X)
            # Y = D̂*β + X*δ + ε
            if X is not None:
                W2 = np.column_stack([D_hat, X, np.ones(n)])
            else:
                W2 = np.column_stack([D_hat, np.ones(n)])

            second_stage_result = self._second_stage(Y, W2, D, D_hat, n)
            beta = second_stage_result["beta"]  # IV coefficient
            residuals = second_stage_result["residuals"]

            # ============ Standard Errors ============
            # Use proper 2SLS standard errors (not OLS on second stage)
            std_error = self._compute_2sls_std_error(Y, D, D_hat, W2, residuals, n, k, p)

            # Degrees of freedom
            df = n - (1 + p + 1)  # n - (treatment + covariates + constant)

            # t-statistic and p-value
            t_stat = beta / std_error if std_error > 0 else 0.0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            # Confidence interval
            ci_lower, ci_upper = self._compute_confidence_interval(beta, std_error, df)

            # ============ Diagnostics ============
            diagnostics = self._compute_diagnostics(
                Y, D, Z, X, residuals, f_stat, f_pvalue, n, k, p
            )

            elapsed = (time.perf_counter() - start_time) * 1000

            return IVResult(
                estimator_type=self.estimator_type,
                success=True,
                coefficient=float(beta),
                std_error=float(std_error),
                ci_lower=float(ci_lower),
                ci_upper=float(ci_upper),
                t_stat=float(t_stat),
                p_value=float(p_value),
                first_stage_coef=first_stage_coef[:k],  # Instrument coefficients only
                first_stage_r_squared=float(first_stage_r_squared),
                diagnostics=diagnostics,
                n_observations=n,
                n_instruments=k,
                n_covariates=p,
                estimation_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"2SLS estimation failed: {e}")
            return IVResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )

    def _first_stage(
        self,
        D: NDArray[np.float64],
        W: NDArray[np.float64],
        n: int,
        k: int,
    ) -> dict:
        """Run first-stage regression: D on Z (and X)."""
        # OLS: D = W*π + ν
        # π = (W'W)^{-1} W'D
        WtW = W.T @ W
        WtD = W.T @ D

        try:
            coef = np.linalg.solve(WtW, WtD)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(W, D, rcond=None)[0]

        D_hat = W @ coef
        residuals = D - D_hat

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((D - np.mean(D)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "coef": coef,
            "D_hat": D_hat,
            "residuals": residuals,
            "r_squared": r_squared,
        }

    def _second_stage(
        self,
        Y: NDArray[np.float64],
        W2: NDArray[np.float64],
        D: NDArray[np.float64],
        D_hat: NDArray[np.float64],
        n: int,
    ) -> dict:
        """Run second-stage regression: Y on D̂ (and X)."""
        # OLS: Y = W2*β + ε where W2 includes D_hat
        WtW = W2.T @ W2
        WtY = W2.T @ Y

        try:
            coef = np.linalg.solve(WtW, WtY)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(W2, Y, rcond=None)[0]

        beta = coef[0]  # Treatment coefficient

        # Use ACTUAL treatment D (not D_hat) for residuals
        # This is important for correct standard errors
        W2 @ coef
        # But replace D_hat with D for residual calculation
        W2_actual = W2.copy()
        W2_actual[:, 0] = D
        Y_hat_actual = W2_actual @ coef
        residuals = Y - Y_hat_actual

        return {
            "beta": beta,
            "coef": coef,
            "residuals": residuals,
        }

    def _compute_2sls_std_error(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        D_hat: NDArray[np.float64],
        W2: NDArray[np.float64],
        residuals: NDArray[np.float64],
        n: int,
        k: int,
        p: int,
    ) -> float:
        """
        Compute correct 2SLS standard errors.

        Note: Cannot use OLS standard errors from second stage directly.
        Must account for first-stage estimation.
        """
        # Number of parameters in second stage
        q = 1 + p + 1  # treatment + covariates + constant

        # Variance of residuals (using degrees of freedom correction)
        sigma_sq = np.sum(residuals**2) / (n - q)

        # (W2'W2)^{-1} using predicted treatment
        try:
            WtW_inv = np.linalg.inv(W2.T @ W2)
        except np.linalg.LinAlgError:
            WtW_inv = np.linalg.pinv(W2.T @ W2)

        # Variance-covariance matrix
        var_cov = sigma_sq * WtW_inv

        # Standard error of treatment coefficient
        std_error = np.sqrt(var_cov[0, 0])

        # Apply heteroskedasticity-robust correction if requested
        if self.config.robust_std_errors:
            std_error = self._robust_std_error(W2, residuals, n, q)

        return float(std_error)

    def _robust_std_error(
        self,
        W: NDArray[np.float64],
        residuals: NDArray[np.float64],
        n: int,
        q: int,
    ) -> float:
        """Compute heteroskedasticity-robust (HC1) standard errors."""
        try:
            WtW_inv = np.linalg.inv(W.T @ W)
        except np.linalg.LinAlgError:
            WtW_inv = np.linalg.pinv(W.T @ W)

        # HC1 correction factor
        correction = n / (n - q)

        # Meat matrix: W' diag(e^2) W
        e_sq = residuals**2
        meat = (W.T * e_sq) @ W

        # Sandwich estimator: (W'W)^{-1} * meat * (W'W)^{-1}
        var_cov = correction * WtW_inv @ meat @ WtW_inv

        return float(np.sqrt(var_cov[0, 0]))

    def _first_stage_f_test(
        self,
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        n: int,
        k: int,
        p: int,
    ) -> tuple[float, float]:
        """
        Compute first-stage F-statistic for instrument relevance.

        Tests H0: π = 0 (instruments are irrelevant)
        """
        # Full model: D = Z*π + X*γ + c + ν
        if X is not None:
            W_full = np.column_stack([Z, X, np.ones(n)])
        else:
            W_full = np.column_stack([Z, np.ones(n)])

        # Restricted model: D = X*γ + c + ν (no instruments)
        if X is not None:
            W_restricted = np.column_stack([X, np.ones(n)])
        else:
            W_restricted = np.ones((n, 1))

        # Full model residuals
        try:
            coef_full = np.linalg.solve(W_full.T @ W_full, W_full.T @ D)
        except np.linalg.LinAlgError:
            coef_full = np.linalg.lstsq(W_full, D, rcond=None)[0]
        resid_full = D - W_full @ coef_full
        ss_full = np.sum(resid_full**2)

        # Restricted model residuals
        try:
            coef_restr = np.linalg.solve(W_restricted.T @ W_restricted, W_restricted.T @ D)
        except np.linalg.LinAlgError:
            coef_restr = np.linalg.lstsq(W_restricted, D, rcond=None)[0]
        resid_restr = D - W_restricted @ coef_restr
        ss_restr = np.sum(resid_restr**2)

        # F-statistic
        df1 = k  # Number of instruments
        df2 = n - k - p - 1  # Degrees of freedom

        if ss_full > 0 and df2 > 0:
            f_stat = ((ss_restr - ss_full) / df1) / (ss_full / df2)
            f_pvalue = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            f_stat = 0.0
            f_pvalue = 1.0

        return float(f_stat), float(f_pvalue)

    def _compute_diagnostics(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        f_stat: float,
        f_pvalue: float,
        n: int,
        k: int,
        p: int,
    ) -> IVDiagnostics:
        """Compute IV diagnostics."""
        diagnostics = IVDiagnostics(
            first_stage_f_stat=f_stat,
            first_stage_f_pvalue=f_pvalue,
            instrument_strength=self._classify_instrument_strength(f_stat),
        )

        # Partial R-squared of instruments
        diagnostics.partial_r_squared = self._partial_r_squared(D, Z, X, n)

        if self.config.run_diagnostics:
            # Anderson-Rubin test (weak-instrument robust)
            ar_stat, ar_pvalue = self._anderson_rubin_test(Y, D, Z, X, n, k, p)
            diagnostics.anderson_rubin_stat = ar_stat
            diagnostics.anderson_rubin_pvalue = ar_pvalue

            # Overidentification test (if k > 1)
            if k > 1 and self.config.run_overid_test:
                sargan_stat, sargan_pvalue = self._sargan_test(residuals, Z, X, n, k, p)
                diagnostics.sargan_stat = sargan_stat
                diagnostics.sargan_pvalue = sargan_pvalue

            # Hausman test for endogeneity
            if self.config.run_hausman_test:
                hausman_stat, hausman_pvalue = self._hausman_test(Y, D, Z, X, n, k, p)
                diagnostics.hausman_stat = hausman_stat
                diagnostics.hausman_pvalue = hausman_pvalue

        return diagnostics

    def _partial_r_squared(
        self,
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        n: int,
    ) -> float:
        """Compute partial R-squared of instruments."""
        # Full model: D = Z*π + X*γ + c + ν
        if X is not None:
            W_full = np.column_stack([Z, X, np.ones(n)])
            W_restricted = np.column_stack([X, np.ones(n)])
        else:
            W_full = np.column_stack([Z, np.ones(n)])
            W_restricted = np.ones((n, 1))

        # Full model R-squared
        coef_full = np.linalg.lstsq(W_full, D, rcond=None)[0]
        ss_res_full = np.sum((D - W_full @ coef_full) ** 2)

        # Restricted model R-squared
        coef_restr = np.linalg.lstsq(W_restricted, D, rcond=None)[0]
        ss_res_restr = np.sum((D - W_restricted @ coef_restr) ** 2)

        # Partial R-squared
        if ss_res_restr > 0:
            partial_r2 = (ss_res_restr - ss_res_full) / ss_res_restr
        else:
            partial_r2 = 0.0

        return float(partial_r2)

    def _anderson_rubin_test(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        n: int,
        k: int,
        p: int,
    ) -> tuple[float, float]:
        """
        Anderson-Rubin test for coefficient significance.

        Robust to weak instruments. Tests H0: β = 0.
        """
        # Under H0 (β=0): Y = X*δ + ε, test if Z explains Y after partialling out X
        if X is not None:
            # Partial out X from Y and Z
            Y_res = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
            Z_res = Z - X @ np.linalg.lstsq(X, Z, rcond=None)[0]
        else:
            Y_res = Y - np.mean(Y)
            Z_res = Z - np.mean(Z, axis=0)

        # Regress Y_res on Z_res
        W = np.column_stack([Z_res, np.ones(n)])
        coef = np.linalg.lstsq(W, Y_res, rcond=None)[0]
        residuals = Y_res - W @ coef

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum(Y_res**2)

        # F-statistic
        df1 = k
        df2 = n - k - p - 1

        if ss_res > 0 and df2 > 0 and ss_tot > ss_res:
            ar_stat = ((ss_tot - ss_res) / df1) / (ss_res / df2)
            ar_pvalue = 1 - stats.f.cdf(ar_stat, df1, df2)
        else:
            ar_stat = 0.0
            ar_pvalue = 1.0

        return float(ar_stat), float(ar_pvalue)

    def _sargan_test(
        self,
        residuals: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        n: int,
        k: int,
        p: int,
    ) -> tuple[float, float]:
        """
        Sargan test for overidentifying restrictions.

        H0: All instruments are valid (uncorrelated with error)
        Only valid when k > 1 (overidentified).
        """
        if k <= 1:
            return 0.0, 1.0  # Exactly identified, cannot test

        # Regress 2SLS residuals on all instruments and covariates
        if X is not None:
            W = np.column_stack([Z, X, np.ones(n)])
        else:
            W = np.column_stack([Z, np.ones(n)])

        coef = np.linalg.lstsq(W, residuals, rcond=None)[0]
        fitted = W @ coef

        # n * R^2 ~ chi^2(k-1)
        r_squared = np.sum(fitted**2) / np.sum(residuals**2)
        sargan_stat = n * r_squared
        sargan_pvalue = 1 - stats.chi2.cdf(sargan_stat, k - 1)

        return float(sargan_stat), float(sargan_pvalue)

    def _hausman_test(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        X: Optional[NDArray[np.float64]],
        n: int,
        k: int,
        p: int,
    ) -> tuple[float, float]:
        """
        Hausman test for endogeneity.

        H0: Treatment is exogenous (OLS is consistent)
        H1: Treatment is endogenous (IV needed)
        """
        # OLS estimate
        if X is not None:
            W_ols = np.column_stack([D, X, np.ones(n)])
        else:
            W_ols = np.column_stack([D, np.ones(n)])

        beta_ols = np.linalg.lstsq(W_ols, Y, rcond=None)[0][0]
        resid_ols = Y - W_ols @ np.linalg.lstsq(W_ols, Y, rcond=None)[0]
        var_ols = np.sum(resid_ols**2) / (n - W_ols.shape[1])

        # IV estimate (already computed, but we need variance)
        # First stage
        if X is not None:
            W1 = np.column_stack([Z, X, np.ones(n)])
        else:
            W1 = np.column_stack([Z, np.ones(n)])

        pi = np.linalg.lstsq(W1, D, rcond=None)[0]
        D_hat = W1 @ pi

        # Second stage
        if X is not None:
            W2 = np.column_stack([D_hat, X, np.ones(n)])
        else:
            W2 = np.column_stack([D_hat, np.ones(n)])

        beta_iv = np.linalg.lstsq(W2, Y, rcond=None)[0][0]

        # Use actual D for residuals
        W2_actual = W2.copy()
        W2_actual[:, 0] = D
        resid_iv = Y - W2_actual @ np.linalg.lstsq(W2, Y, rcond=None)[0]
        var_iv = np.sum(resid_iv**2) / (n - W2.shape[1])

        # Hausman statistic
        diff = beta_iv - beta_ols
        var_diff = var_iv - var_ols  # Approximate

        if var_diff > 0:
            hausman_stat = diff**2 / var_diff
            hausman_pvalue = 1 - stats.chi2.cdf(hausman_stat, 1)
        else:
            hausman_stat = 0.0
            hausman_pvalue = 1.0

        return float(hausman_stat), float(hausman_pvalue)
