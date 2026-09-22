"""Resolve the real causal cohort parquet the way the Lane A loader would
(``_resolve_agent_estimation_frame`` on branch claude/real-data-causal-estimation,
read 2026-09-22): numeric coercion of every non-categorical MART_SAFE_FEATURES
column, ``<col>=<level>`` drop-first one-hot of the 7 text categoricals with a
``<col>=__missing__`` dummy where NULLs exist. The exact-collinearity prune the
loader ALSO applies is deliberately NOT replicated here: Lane D's pre-flight is
measured on the un-pruned 77-column frame so its own rank prune is exercised.

Import from a heredoc run at the worktree root (a script run BY PATH imports
``src`` from the MAIN checkout via the editable .pth)."""

from __future__ import annotations

import numpy as np
import pandas as pd

PARQUET = (
    "/home/enunez/Projects/e2i_causal_analytics/data/rwd/mart/persistence_causal/"
    "e2i_causal_v1_biologic_persistence.parquet"
)
TREATMENT = "treatment_dupixent"
OUTCOME = "persistent_at_180d_g28"
CATEGORICALS = [
    "gdr_cd",
    "payer_category",
    "payer_product",
    "payer_bus",
    "charlson_risk_band",
    "elixhauser_risk_band",
    "geographic_region",
]


def one_hot(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    names: list[str] = []
    for col in cols:
        levels = sorted(str(v) for v in out[col].dropna().unique())
        for level in levels[1:]:
            name = f"{col}={level}"
            out[name] = (out[col].astype(str) == level).astype(float)
            names.append(name)
        if out[col].isna().any():
            name = f"{col}=__missing__"
            out[name] = out[col].isna().astype(float)
            names.append(name)
        out = out.drop(columns=[col])
    return out, names


def resolve(safe_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(frame, covariates)`` with columns ``[T, Y, *covariates]``."""
    raw = pd.read_parquet(PARQUET)
    numeric = [c for c in safe_features if c not in CATEGORICALS]
    frame = raw[[TREATMENT, OUTCOME] + safe_features].copy()
    for c in [TREATMENT, OUTCOME] + numeric:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
    frame = frame.dropna(subset=[TREATMENT, OUTCOME])
    frame, dummies = one_hot(frame, [c for c in CATEGORICALS if c in safe_features])
    covariates = numeric + dummies
    frame = frame[[TREATMENT, OUTCOME] + covariates].reset_index(drop=True)
    return frame, covariates


def corr_rank(frame: pd.DataFrame) -> int:
    return int(np.linalg.matrix_rank(np.corrcoef(frame.to_numpy(dtype=float).T)))
