"""FeatureBuilder — leakage-safe encoding contract for gold-standard evaluation.

build_from_frame() is the pure, I/O-free encoding contract:
  - Drops the label column, all LEAKAGE_DENYLIST columns, and id/split columns.
  - One-hot-encodes categoricals (with dummy_na so null categories don't silently
    vanish), median-imputes numerics (with an explicit __isna missingness flag).
  - Records feature_columns so downstream callers don't need to re-derive the set.

build_for_split() is an INTENTIONAL STUB — Task 4 implements the live
patient_journeys DB loader (brand+split filtering, include_synthetic=True,
optional journey_start_date < before_month cutoff, → build_from_frame).  Do NOT
add DB access here; the contract boundary is deliberate.

LEAKAGE_DENYLIST rationale:
  All columns that are knowable only AFTER the initiation decision must never
  enter the feature matrix.  outcome_probability and treatment_propensity are
  outcome-derived feature_values (verified against src/data/feature_contract.py
  in Task 3's EXPERIMENT lock).
"""

from __future__ import annotations

import pandas as pd

# Columns knowable only AFTER the initiation decision → never features (anti-leakage).
LEAKAGE_DENYLIST = (
    "treatment_initiated",
    "days_to_treatment",
    "discontinued_180d",
    "persistent_180d",
    "adherence_rate",
    "refill_count",
    "gap_days",
    "is_churned",
    "treatment_arm",
    "outcome_probability",      # outcome-derived feature_value (verify w/ feature_contract)
    "treatment_propensity",     # outcome-derived feature_value (verify w/ feature_contract)
)

# Identity / split columns that are never predictive features.
_ID_COLS = frozenset({"patient_id", "patient_hash", "data_split", "split_config_id"})


class FeatureBuilder:
    """Encode a raw patient_journeys DataFrame into a leakage-safe feature matrix.

    Parameters
    ----------
    spec:
        A ``CohortSpec`` instance that supplies the label column name and the
        leakage-safe base covariate seed.  The final column set is determined
        empirically in Task 3's EXPERIMENT lock; this class implements the
        encoding *mechanism*, not the column selection policy.

    Usage
    -----
    ::

        fb = FeatureBuilder(INITIATION)
        X, y = fb.build_from_frame(raw_df)
        # fb.feature_columns now holds the ordered list of encoded column names.
    """

    def __init__(self, spec: object) -> None:
        self.spec = spec
        self.feature_columns: list[str] = []

    def build_from_frame(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Encode *raw* into (X, y) with no leakage, no NaNs.

        Parameters
        ----------
        raw:
            DataFrame with at least ``spec.label_column`` present.  May contain
            leakage columns, id columns, and split columns — all are dropped.

        Returns
        -------
        X:
            Encoded feature matrix (float dtype throughout).  Categoricals are
            one-hot-encoded with ``dummy_na=True``; numerics are median-imputed
            and accompanied by a ``<col>__isna`` missingness flag.  No NaNs.
        y:
            Integer label series aligned to *raw*'s index.
        """
        y = raw[self.spec.label_column].astype(int)

        drop = set(LEAKAGE_DENYLIST) | _ID_COLS
        feats = raw.drop(columns=[c for c in raw.columns if c in drop], errors="ignore")

        feats = self._encode(feats)
        self.feature_columns = list(feats.columns)
        return feats, y

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot-encode categoricals; median-impute numerics + missingness flag."""
        out: dict[str, pd.Series] = {}
        for col in df.columns:
            s = df[col]
            if s.dtype == object or str(s.dtype) == "category":
                dummies = pd.get_dummies(s, prefix=col, dummy_na=True)
                for val, dummy in dummies.items():
                    out[val] = dummy.astype(float)
            else:
                out[f"{col}__isna"] = s.isnull().astype(float)
                out[col] = s.fillna(s.median()).astype(float)
        return pd.DataFrame(out, index=df.index)

    async def build_for_split(
        self,
        db: object,
        split: str,
        *,
        before_month: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load patient_journeys rows for *split* and encode via build_from_frame.

        STUB — implemented in Task 4.

        Task 4 will:
          - Query patient_journeys filtered by self.spec.brand, data_split=split,
            include_synthetic=True (gold-standard uses synthetic cohort).
          - Apply optional journey_start_date < before_month cutoff (walk-forward
            window experiment).
          - Call self.build_from_frame(raw) and return the result.

        Do NOT implement DB access in this task.
        """
        raise NotImplementedError(
            "build_for_split is implemented in Task 4 (live patient_journeys loader). "
            "See Task 4 for: brand+split filtering, include_synthetic=True, "
            "optional journey_start_date < before_month cutoff."
        )
