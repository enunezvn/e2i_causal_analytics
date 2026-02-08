"""
Limited Information Maximum Likelihood (LIML) Estimator

LIML is an alternative to 2SLS that is:
- More robust to weak instruments
- Asymptotically equivalent to 2SLS with strong instruments
- Less biased with weak instruments (especially with Fuller correction)

The LIML estimator solves:
    min_β (Y - D*β)'M_Z(Y - D*β) / (Y - D*β)'M_X(Y - D*β)

where:
    M_Z = I - Z(Z'Z)^{-1}Z' (residual maker for Z)
    M_X = I - X(X'X)^{-1}X' (residual maker for X)

References:
    - Anderson & Rubin (1949) "Estimation of the Parameters of a Single Equation"
    - Fuller (1977) "Some Properties of a Modification of the LIML Estimator"
    - Stock & Yogo (2005) "Testing for Weak Instruments"
"""

from __future__ import annotations

import logging
import time
from typing import Optional, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats

from .base import (
    BaseIVEstimator,
    IVConfig,
    IVDiagnostics,
    IVEstimatorType,
    IVResult,
)

logger = logging.getLogger(__name__)


class LIMLEstimator(BaseIVEstimator):
    """
    Limited Information Maximum Likelihood (LIML) IV Estimator.

    LIML provides better finite-sample properties than 2SLS when
    instruments are weak. The Fuller modification further reduces
    bias at the cost of slightly increased variance.

    Usage:
        # Standard LIML
        estimator = LIMLEstimator()
        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)

        # Fuller's modification (k=1 is common choice)
        config = IVConfig(fuller_k=1.0)
        estimator = LIMLEstimator(config)
        result = estimator.fit(outcome=Y, treatment=D, instruments=Z)
    """

    @property
    def estimator_type(self) -> IVEstimatorType:
        return IVEstimatorType.LIML

    def fit(
        self,
        outcome: NDArray[np.float64],
        treatment: NDArray[np.float64],
        instruments: NDArray[np.float64],
        covariates: Optional[NDArray[np.float64]] = None,
        **kwargs,
    ) -> IVResult:
        """
        Fit LIML estimator.

        Args:
            outcome: Y - observed outcomes (n,)
            treatment: D - endogenous treatment (n,)
            instruments: Z - instruments (n, k)
            covariates: X - exogenous controls (n, p) or None

        Returns:
            IVResult with LIML estimates
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

            # Handle covariates
            if covariates is not None:
                X = covariates.reshape(-1, 1) if covariates.ndim == 1 else covariates
                p = X.shape[1]
            else:
                X = None
                p = 0

            # ============ Partial out exogenous variables ============
            if X is not None:
                # Partial out X from Y, D, and Z
                M_X = self._residual_maker(X)
                Y_tilde = M_X @ Y
                D_tilde = M_X @ D
                Z_tilde = M_X @ Z
            else:
                Y_tilde = Y - np.mean(Y)
                D_tilde = D - np.mean(D)
                Z_tilde = Z - np.mean(Z, axis=0)

            # ============ Compute LIML k-class parameter ============
            kappa = self._compute_liml_kappa(Y_tilde, D_tilde, Z_tilde, n, k)

            # Apply Fuller modification if specified
            if self.config.fuller_k is not None:
                kappa = kappa - self.config.fuller_k / (n - k - p - 1)

            # ============ k-class estimator ============
            # β = ((D'D - κ*D'M_Z*D)^{-1} (D'Y - κ*D'M_Z*Y)
            # where M_Z = I - Z(Z'Z)^{-1}Z'

            # Projection matrix P_Z
            try:
                ZtZ_inv = np.linalg.inv(Z_tilde.T @ Z_tilde)
            except np.linalg.LinAlgError:
                ZtZ_inv = np.linalg.pinv(Z_tilde.T @ Z_tilde)

            P_Z = Z_tilde @ ZtZ_inv @ Z_tilde.T
            M_Z = np.eye(n) - P_Z

            # Numerator and denominator for β
            DtD = D_tilde.T @ D_tilde
            DtY = D_tilde.T @ Y_tilde
            DtMzD = D_tilde.T @ M_Z @ D_tilde
            DtMzY = D_tilde.T @ M_Z @ Y_tilde

            denom = DtD - kappa * DtMzD
            numer = DtY - kappa * DtMzY

            if abs(denom) < 1e-10:
                raise ValueError("LIML denominator near zero - estimation unstable")

            beta = float(numer / denom)

            # ============ Residuals and Standard Errors ============
            # Residuals using actual treatment
            if X is not None:
                np.column_stack([D, X, np.ones(n)])
            else:
                np.column_stack([D, np.ones(n)])

            # Full model coefficients
            residuals = Y_tilde - D_tilde * beta
            sigma_sq = np.sum(residuals**2) / (n - k - p - 1)

            # LIML variance (approximate)
            std_error = self._compute_liml_std_error(D_tilde, Z_tilde, sigma_sq, kappa, n, k, p)

            # t-statistic and p-value
            df = n - k - p - 1
            t_stat = beta / std_error if std_error > 0 else 0.0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            # Confidence interval
            ci_lower, ci_upper = self._compute_confidence_interval(beta, std_error, df)

            # ============ First-stage statistics ============
            f_stat, f_pvalue = self._first_stage_f_test(D_tilde, Z_tilde, n, k)

            # ============ Diagnostics ============
            diagnostics = IVDiagnostics(
                first_stage_f_stat=f_stat,
                first_stage_f_pvalue=f_pvalue,
                instrument_strength=self._classify_instrument_strength(f_stat),
            )

            # Partial R-squared
            diagnostics.partial_r_squared = self._partial_r_squared(D_tilde, Z_tilde, n)

            # Anderson-Rubin test
            ar_stat, ar_pvalue = self._anderson_rubin_test(Y_tilde, D_tilde, Z_tilde, n, k)
            diagnostics.anderson_rubin_stat = ar_stat
            diagnostics.anderson_rubin_pvalue = ar_pvalue

            # First-stage R-squared
            pi = np.linalg.lstsq(Z_tilde, D_tilde, rcond=None)[0]
            D_hat = Z_tilde @ pi
            ss_res = np.sum((D_tilde - D_hat) ** 2)
            ss_tot = np.sum(D_tilde**2)
            first_stage_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            elapsed = (time.perf_counter() - start_time) * 1000

            return IVResult(
                estimator_type=self.estimator_type,
                success=True,
                coefficient=beta,
                std_error=std_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                t_stat=t_stat,
                p_value=p_value,
                first_stage_coef=cast(NDArray[np.float64], pi),
                first_stage_r_squared=float(first_stage_r2),
                diagnostics=diagnostics,
                n_observations=n,
                n_instruments=k,
                n_covariates=p,
                estimation_time_ms=elapsed,
                raw_estimate={"kappa": kappa, "fuller_k": self.config.fuller_k},
            )

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error(f"LIML estimation failed: {e}")
            return IVResult(
                estimator_type=self.estimator_type,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                estimation_time_ms=elapsed,
            )

    def _residual_maker(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Create residual maker matrix M_X = I - X(X'X)^{-1}X'."""
        n = X.shape[0]
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)

        P_X = X @ XtX_inv @ X.T
        M_X = np.eye(n) - P_X
        return M_X

    def _compute_liml_kappa(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        n: int,
        k: int,
    ) -> float:
        """
        Compute LIML kappa parameter.

        LIML finds the smallest root λ of:
            det(W'M_Z W - λ W'M_0 W) = 0

        where:
            W = [Y, D]
            M_Z = I - Z(Z'Z)^{-1}Z' (residual maker for instruments)
            M_0 = I - 1*1'/n (demeaning matrix, for no exogenous covariates)

        κ is this smallest eigenvalue, and should be close to 1 when
        instruments are strong (making LIML ≈ 2SLS).

        Reference:
            Anderson & Rubin (1949), Stock & Yogo (2005)
        """
        # Form W = [Y, D]
        W = np.column_stack([Y, D])

        # M_Z residual maker (residuals after regressing on Z)
        try:
            ZtZ_inv = np.linalg.inv(Z.T @ Z)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(Z.T @ Z)

        P_Z = Z @ ZtZ_inv @ Z.T
        M_Z = np.eye(n) - P_Z

        # M_0 demeaning matrix (residuals after regressing on constant)
        ones = np.ones((n, 1))
        M_0 = np.eye(n) - ones @ ones.T / n

        # W'M_Z*W and W'M_0*W
        WtMzW = W.T @ M_Z @ W
        WtM0W = W.T @ M_0 @ W

        # Solve generalized eigenvalue problem
        # W'M_Z*W * v = κ * W'M_0*W * v
        # The smallest eigenvalue is the LIML kappa
        try:
            eigenvalues = linalg.eigvalsh(WtMzW, WtM0W)
            # Filter out numerical noise (eigenvalues should be >= 0)
            eigenvalues = eigenvalues[eigenvalues > -1e-10]
            kappa = float(np.min(eigenvalues)) if len(eigenvalues) > 0 else 1.0
            # Ensure kappa >= 0 (numerical stability)
            kappa = max(0.0, kappa)
        except Exception:
            # Fallback: direct eigenvalue computation
            try:
                WtM0W_inv = np.linalg.inv(WtM0W)
                A = WtM0W_inv @ WtMzW
                eigenvalues = np.linalg.eigvalsh(A)
                eigenvalues = eigenvalues[eigenvalues > -1e-10]
                kappa = float(np.min(eigenvalues)) if len(eigenvalues) > 0 else 1.0
                kappa = max(0.0, kappa)
            except Exception:
                # Use 2SLS as fallback (κ = 1)
                logger.warning("LIML eigenvalue computation failed, using κ=1 (2SLS)")
                kappa = 1.0

        return kappa

    def _compute_liml_std_error(
        self,
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        sigma_sq: float,
        kappa: float,
        n: int,
        k: int,
        p: int,
    ) -> float:
        """Compute LIML standard error."""
        # P_Z projection
        try:
            ZtZ_inv = np.linalg.inv(Z.T @ Z)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(Z.T @ Z)

        P_Z = Z @ ZtZ_inv @ Z.T
        M_Z = np.eye(n) - P_Z

        # Variance formula
        DtD = D.T @ D
        DtMzD = D.T @ M_Z @ D
        denom = DtD - kappa * DtMzD

        if abs(denom) < 1e-10:
            return float("inf")

        var_beta = sigma_sq / denom

        if self.config.robust_std_errors:
            # Apply HC1 correction
            var_beta *= n / (n - k - p - 1)

        return float(np.sqrt(var_beta))

    def _first_stage_f_test(
        self,
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        n: int,
        k: int,
    ) -> tuple[float, float]:
        """Compute first-stage F-statistic."""
        # Regress D on Z
        coef = np.linalg.lstsq(Z, D, rcond=None)[0]
        D_hat = Z @ coef
        residuals = D - D_hat

        ss_reg = np.sum((D_hat - np.mean(D)) ** 2)
        ss_res = np.sum(residuals**2)

        df1 = k
        df2 = n - k

        if ss_res > 0 and df2 > 0:
            f_stat = (ss_reg / df1) / (ss_res / df2)
            f_pvalue = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            f_stat = 0.0
            f_pvalue = 1.0

        return float(f_stat), float(f_pvalue)

    def _partial_r_squared(
        self,
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        n: int,
    ) -> float:
        """Compute partial R-squared of instruments."""
        coef = np.linalg.lstsq(Z, D, rcond=None)[0]
        D_hat = Z @ coef

        ss_res = np.sum((D - D_hat) ** 2)
        ss_tot = np.sum(D**2)

        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _anderson_rubin_test(
        self,
        Y: NDArray[np.float64],
        D: NDArray[np.float64],
        Z: NDArray[np.float64],
        n: int,
        k: int,
    ) -> tuple[float, float]:
        """Anderson-Rubin test for β = 0."""
        # Under H0: Y = ε, test if Z explains Y
        coef = np.linalg.lstsq(Z, Y, rcond=None)[0]
        Y_hat = Z @ coef
        residuals = Y - Y_hat

        ss_reg = np.sum((Y_hat - np.mean(Y)) ** 2)
        ss_res = np.sum(residuals**2)

        df1 = k
        df2 = n - k

        if ss_res > 0 and df2 > 0:
            ar_stat = (ss_reg / df1) / (ss_res / df2)
            ar_pvalue = 1 - stats.f.cdf(ar_stat, df1, df2)
        else:
            ar_stat = 0.0
            ar_pvalue = 1.0

        return float(ar_stat), float(ar_pvalue)


class FullerEstimator(LIMLEstimator):
    """
    Fuller's modification of LIML.

    Uses κ_Fuller = κ_LIML - c / (n - k - p - 1)

    where c is typically set to 1 or 4.
    - c = 1: Approximately unbiased
    - c = 4: Minimizes MSE approximately

    Reference:
        Fuller (1977) "Some Properties of a Modification of the LIML Estimator"
    """

    def __init__(self, config: Optional[IVConfig] = None, fuller_k: float = 1.0):
        """
        Initialize Fuller estimator.

        Args:
            config: IV configuration
            fuller_k: Fuller modification parameter (default 1.0)
        """
        if config is None:
            config = IVConfig(fuller_k=fuller_k)
        elif config.fuller_k is None:
            config.fuller_k = fuller_k

        super().__init__(config)

    @property
    def estimator_type(self) -> IVEstimatorType:
        return IVEstimatorType.FULLER
