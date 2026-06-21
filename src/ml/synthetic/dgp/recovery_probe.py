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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

_SEGMENTS = ("high_severity", "medium_severity", "low_severity")
_COVARS = ["disease_severity", "academic_hcp"]


def recover_ate_and_cate(df: pd.DataFrame) -> Dict[str, Any]:
    """Recover ATE (LinearDML) + per-segment CATE (CausalForestDML) from a
    synthetic patient frame produced by PatientGenerator (Task 03.4)."""
    Y = df["treatment_initiated"].to_numpy(dtype=float)
    T = df["treatment_arm"].to_numpy(dtype=int)
    X = df[_COVARS].to_numpy(dtype=float)
    seg = df["segment_assignment"].to_numpy()

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
        "true_ate": float(df.attrs.get("true_ate", np.mean(eff))),
        "linear_dml_ate": linear_dml_ate,
        "cate_by_segment_estimate": cate_by_segment_estimate,
        "propensity_auc": propensity_auc,
        "n_treated": n_treated,
        "n_control": n_control,
    }


# The 7 leakage-safe persistence covariates (T9), encoded like FeatureBuilder
# (geographic_region + insurance_type one-hot, the rest passthrough numeric).
_PERSIST_KEEP = [
    "disease_severity",
    "academic_hcp",
    "geographic_region",
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
]


def measure_persistence_signal(df: pd.DataFrame, label: str = "persistent_180d") -> Dict[str, Any]:
    """Fit an LR on the 7 KEEP_COLUMNS (encoded like FeatureBuilder) and return the
    holdout AUC + marginal prevalence.

    This is the measure-don't-assume calibration gate: it reports the ACHIEVED
    Bayes-proxy AUC of the generated data, not an assumption. The DGP coefficients in
    cohort_outcomes.py are tuned until this lands in the realistic [0.78, 0.82] band.
    """
    cats = [c for c in ("geographic_region", "insurance_type") if c in df.columns]
    nums = [c for c in _PERSIST_KEEP if c not in cats and c in df.columns]
    X = df[cats + nums]
    y = df[label].to_numpy(dtype=int)
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cats)],
        remainder="passthrough",
    )
    pipe = Pipeline(
        [("pre", pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    pipe.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]))
    return {"holdout_auc": auc, "prevalence": float(y.mean()), "n": int(len(y))}
