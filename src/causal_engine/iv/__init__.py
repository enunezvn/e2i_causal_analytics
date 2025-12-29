"""
Instrumental Variable (IV) Estimation Module

Provides IV estimators and diagnostics for handling endogeneity in
causal effect estimation.

Estimators:
    - TwoStageLSEstimator: Two-Stage Least Squares (2SLS)
    - LIMLEstimator: Limited Information Maximum Likelihood
    - FullerEstimator: Fuller's modification of LIML

Diagnostics:
    - Weak instrument tests (Cragg-Donald, Stock-Yogo)
    - Overidentification tests (Sargan)
    - Endogeneity tests (Durbin-Wu-Hausman)
    - Anderson-Rubin test (weak-IV robust inference)

Usage:
    # Standard 2SLS estimation
    from src.causal_engine.iv import TwoStageLSEstimator

    estimator = TwoStageLSEstimator()
    result = estimator.fit(
        outcome=Y,
        treatment=D,
        instruments=Z,
        covariates=X,
    )

    if result.success:
        print(f"LATE: {result.coefficient:.3f} ± {result.std_error:.3f}")
        print(f"First-stage F: {result.diagnostics.first_stage_f_stat:.2f}")

    # LIML for weak instruments
    from src.causal_engine.iv import LIMLEstimator, IVConfig

    config = IVConfig(fuller_k=1.0)  # Fuller modification
    liml = LIMLEstimator(config)
    result = liml.fit(outcome=Y, treatment=D, instruments=Z)

    # Comprehensive diagnostics
    from src.causal_engine.iv import run_all_diagnostics

    report = run_all_diagnostics(
        outcome=Y,
        treatment=D,
        instruments=Z,
        residuals=result.raw_estimate.get('residuals', Y - D * result.coefficient),
    )
    print(report.recommendation)

References:
    - Angrist & Pischke (2009) "Mostly Harmless Econometrics"
    - Stock & Yogo (2005) "Testing for Weak Instruments"
    - Fuller (1977) "Some Properties of a Modification of the LIML Estimator"
"""

from src.causal_engine.iv.base import (
    BaseIVEstimator,
    InstrumentStrength,
    IVConfig,
    IVDiagnostics,
    IVEstimatorType,
    IVResult,
)
from src.causal_engine.iv.diagnostics import (
    EndogeneityTest,
    IVDiagnosticReport,
    OveridentificationTest,
    WeakInstrumentTest,
    anderson_rubin_test,
    cragg_donald_test,
    durbin_wu_hausman_test,
    partial_r_squared,
    run_all_diagnostics,
    sargan_test,
    stock_yogo_critical_values,
)
from src.causal_engine.iv.liml import FullerEstimator, LIMLEstimator
from src.causal_engine.iv.two_stage_ls import TwoStageLSEstimator

__all__ = [
    # Base classes
    "BaseIVEstimator",
    "IVConfig",
    "IVResult",
    "IVDiagnostics",
    "IVEstimatorType",
    "InstrumentStrength",
    # Estimators
    "TwoStageLSEstimator",
    "LIMLEstimator",
    "FullerEstimator",
    # Diagnostic types
    "WeakInstrumentTest",
    "OveridentificationTest",
    "EndogeneityTest",
    "IVDiagnosticReport",
    # Diagnostic functions
    "cragg_donald_test",
    "partial_r_squared",
    "sargan_test",
    "anderson_rubin_test",
    "durbin_wu_hausman_test",
    "run_all_diagnostics",
    "stock_yogo_critical_values",
]
