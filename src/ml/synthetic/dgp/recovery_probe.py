"""Self-contained ATE/CATE recovery probe over a generated patient frame.

Mirrors the agents' estimators (EconML LinearDML + CausalForestDML) so the
acceptance gate proves the DGP is recoverable BEFORE the agents run against
the DB. Shard 11 reuses this for gate 3.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.ml.synthetic.dgp.treatment_arm import ARM_CONFOUNDERS

_SEGMENTS = ("high_severity", "medium_severity", "low_severity")
# The backdoor adjustment set for treatment_arm IS exactly the covariates the
# arm is confounded on — sourced from the DGP SSOT so it can never drift from
# assign_treatment_arm (a stale set here would silently false-green the recovery
# gate). See ARM_CONFOUNDERS + tests/unit/test_synthetic/test_arm_confounder_contract.py.
_COVARS = list(ARM_CONFOUNDERS)


def recover_ate_and_cate(
    df: pd.DataFrame,
    *,
    treatment_col: str = "treatment_arm",
    outcome_col: str = "treatment_initiated",
    confounders: list | None = None,
    segment_col: str = "segment_assignment",
    true_ate: float | None = None,
    cate_map: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Recover ATE (LinearDML) + per-segment CATE (CausalForestDML) from a
    synthetic patient frame. Defaults reproduce the original treatment_arm ->
    treatment_initiated probe; pass the keyword args to validate any other arm/
    outcome/confounder/segment tuple (commercial-arms enrichment)."""
    covars = list(confounders) if confounders is not None else _COVARS
    Y = df[outcome_col].to_numpy(dtype=float)
    T = df[treatment_col].to_numpy(dtype=int)
    X = df[covars].to_numpy(dtype=float)
    seg = df[segment_col].to_numpy()

    n_treated, n_control = int(T.sum()), int(len(T) - T.sum())
    propensity_auc = float(
        roc_auc_score(T, LogisticRegression(max_iter=1000).fit(X, T).predict_proba(X)[:, 1])
    )

    # ATE via LinearDML (discrete treatment, RF nuisances — same family as the agents)
    ldml = LinearDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42),
        discrete_treatment=True,
        random_state=42,
    )
    ldml.fit(Y, T, X=X, W=None)
    linear_dml_ate = float(np.mean(ldml.effect(X)))

    # CATE via CausalForestDML, averaged within each DGP segment
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42),
        discrete_treatment=True,
        n_estimators=200,
        subforest_size=4,
        min_samples_leaf=10,
        random_state=42,
    )
    cf.fit(Y, T, X=X, W=None)
    eff = cf.effect(X)
    cate_by_segment_estimate = {
        s: float(np.mean(eff[seg == s])) for s in _SEGMENTS if np.any(seg == s)
    }

    return {
        "true_ate": float(true_ate) if true_ate is not None else float(df.attrs.get("true_ate", np.mean(eff))),
        "linear_dml_ate": linear_dml_ate,
        "cate_by_segment_estimate": cate_by_segment_estimate,
        "propensity_auc": propensity_auc,
        "n_treated": n_treated,
        "n_control": n_control,
    }
