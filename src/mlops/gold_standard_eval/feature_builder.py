"""FeatureBuilder — leakage-safe encoding contract for gold-standard evaluation.

build_from_frame() is the pure, I/O-free *fit* contract:
  - Restricts raw columns to ``KEEP_COLUMNS`` (the empirically-locked feature set)
    when that allowlist is non-empty; otherwise keeps everything not denylisted.
  - Drops the label column, all LEAKAGE_DENYLIST columns, and id/split columns.
  - One-hot-encodes categoricals (with dummy_na so null categories don't silently
    vanish), median-imputes numerics (with an explicit __isna missingness flag),
    and *learns* the numeric medians + the ordered ``feature_columns`` so eval
    frames encode identically.

transform() is the *apply* half of the fit/transform split.  It encodes an
eval frame using the medians learned at fit time and REINDEXES to the fitted
``feature_columns`` (one-hot columns absent from an eval split → filled with 0,
columns unseen at fit → dropped).  Without this, ``build_from_frame`` recomputes
``feature_columns`` per call and train/eval column sets silently disagree,
making any cross-split metric meaningless.  Tasks 4/8 (live loader, walk-forward)
depend on this consistency.

load_frame() loads patient_journeys rows from the live DB (Task 4).
build_for_split() is the convenience wrapper that loads + fits in one call.

KEEP_COLUMNS rationale (Task 3 EXPERIMENT lock, 2026-06-14):
  Locked by MEASURED holdout AUC on real synthetic Remibrutinib initiation rows
  (train n=2103, holdout n=5075, positive rate ~0.35).  Three candidate tiers
  were fit on TRAIN and scored on HOLDOUT (LogisticRegression, class_weight
  balanced):
    A) base covariates only ......................... holdout AUC 0.6709
    B) A + leakage-safe patient_journeys extras ..... holdout AUC 0.6694
    C) B + new patient-keyed feature_values ......... holdout AUC 0.6659
  The 3 codebase-intent base covariates ALONE gave the best held-out AUC
  (stable across validation 0.685 / test 0.643 / holdout 0.671); extra columns
  added noise, not signal.  So KEEP_COLUMNS == INITIATION.base_covariates.
  See docs/superpowers/plans/experiments/2026-06-14-initiation-features.md.

LEAKAGE_DENYLIST rationale:
  All columns that are knowable only AFTER the initiation decision must never
  enter the feature matrix.  outcome_probability and treatment_propensity are
  outcome-derived feature_values (verified against src/data/feature_contract.py
  in Task 3's EXPERIMENT lock: both are post_index-knowable causal-graph
  quantities, not pre-decision covariates).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we always SELECT from patient_journeys — label + id/split (for
# filtering / downstream use) + KEEP_COLUMNS raw inputs.
_ALWAYS_SELECT = (
    "patient_id",
    "journey_start_date",
    "data_split",
)
# PostgREST cap-agnostic page size (mirrors the blessed idiom in
# src/repositories/business_metric.py get_by_region_paged).
_PAGE_SIZE = 1000
_MAX_PAGES = 10_000  # runaway guard (~10M rows before emitting a warning)

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
    "outcome_probability",  # outcome-derived feature_value (verify w/ feature_contract)
    "treatment_propensity",  # outcome-derived feature_value (verify w/ feature_contract)
)

# Empirically-locked raw column allowlist for the INITIATION cohort (Task 3).
# Equal to INITIATION.base_covariates — the measured holdout AUC (0.6709) was
# best with these three alone; PJ extras and patient-keyed feature_values did
# not improve held-out AUC. An empty tuple means "no allowlist" (keep all
# non-denylisted columns) so the encoding mechanism stays cohort-agnostic.
KEEP_COLUMNS: tuple[str, ...] = (
    "disease_severity",
    "academic_hcp",
    "geographic_region",
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
        X_train, y_train = fb.build_from_frame(train_df)   # fit: learns columns + medians
        X_eval = fb.transform(holdout_df)                  # apply: aligned to fit
        # fb.feature_columns now holds the ordered list of encoded column names.
    """

    def __init__(self, spec: object, keep_columns: tuple[str, ...] | None = None) -> None:
        self.spec = spec
        # ``None`` → use the module-level locked allowlist; pass () to disable
        # the allowlist (keep all non-denylisted columns), e.g. for ad-hoc probes.
        self.keep_columns: tuple[str, ...] = KEEP_COLUMNS if keep_columns is None else keep_columns
        self.feature_columns: list[str] = []
        # Numeric medians learned at fit time so transform() imputes eval frames
        # with TRAIN statistics (not the eval frame's own median → no train/eval
        # leak, no per-call drift).
        self._numeric_medians: dict[str, float] = {}
        self._fitted: bool = False

    def _select(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Apply the KEEP_COLUMNS allowlist (if set) and drop leakage/id columns.

        The allowlist is intersected with the frame's actual columns so a frame
        missing an allowlisted column does not raise here (it surfaces later as
        a reindex-filled zero column in transform()).
        """
        if self.keep_columns:
            cols = [c for c in raw.columns if c in self.keep_columns]
            feats = raw[cols]
        else:
            drop = set(LEAKAGE_DENYLIST) | _ID_COLS
            feats = raw.drop(columns=[c for c in raw.columns if c in drop], errors="ignore")
        return feats

    def build_from_frame(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """FIT: encode *raw* into (X, y) with no leakage, no NaNs.

        Learns the numeric medians and the ordered ``feature_columns`` so that a
        later :meth:`transform` call encodes eval frames identically.  Re-calling
        ``build_from_frame`` re-fits (overwrites the learned state).

        Parameters
        ----------
        raw:
            DataFrame with at least ``spec.label_column`` present.  Columns are
            restricted to ``keep_columns`` when set; otherwise leakage/id columns
            are dropped.

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

        feats = self._select(raw)
        self._numeric_medians = {}
        encoded = self._encode(feats, learn=True)
        self.feature_columns = list(encoded.columns)
        self._fitted = True
        return encoded, y

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """APPLY: encode an eval frame, ALIGNED to the fitted feature set.

        Uses the medians learned in :meth:`build_from_frame` and reindexes the
        result to the fitted ``feature_columns``: one-hot columns absent from
        this frame are filled with 0.0, and columns unseen at fit time are
        dropped.  This is what makes a holdout/walk-forward AUC meaningful — the
        model is scored on the exact column space it was trained on.

        Returns the encoded feature matrix only (no label); the label column is
        not required to be present.

        Raises
        ------
        RuntimeError:
            If called before :meth:`build_from_frame` (nothing has been fitted).
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureBuilder.transform called before build_from_frame; "
                "fit on the train frame first so feature_columns/medians exist."
            )
        feats = self._select(raw)
        encoded = self._encode(feats, learn=False)
        return encoded.reindex(columns=self.feature_columns, fill_value=0.0)

    def _encode(self, df: pd.DataFrame, *, learn: bool) -> pd.DataFrame:
        """One-hot-encode categoricals; median-impute numerics + missingness flag.

        When ``learn`` is True the numeric medians are computed from *df* and
        stored; when False the medians stored at fit time are reused (NaNs that
        remain — e.g. an all-null eval column — fall back to 0.0).
        """
        out: dict[str, pd.Series] = {}
        for col in df.columns:
            s = df[col]
            if s.dtype == object or str(s.dtype) == "category":
                dummies = pd.get_dummies(s, prefix=col, dummy_na=True)
                for val, dummy in dummies.items():
                    out[val] = dummy.astype(float)
            else:
                out[f"{col}__isna"] = s.isnull().astype(float)
                if learn:
                    median = s.median()
                    # All-null numeric at fit time has no median (NaN) → 0.0.
                    median = 0.0 if pd.isna(median) else float(median)
                    self._numeric_medians[col] = median
                else:
                    median = self._numeric_medians.get(col, 0.0)
                out[col] = s.fillna(median).astype(float)
        return pd.DataFrame(out, index=df.index)

    async def load_frame(
        self,
        db: Any,
        *,
        splits: Sequence[str] | None = None,
        before_month: str | None = None,
    ) -> pd.DataFrame:
        """Load patient_journeys rows for ``self.spec.brand`` with ``is_synthetic=True``.

        Mirrors the cap-agnostic PK-ordered ``.range()`` pagination idiom from
        ``src/repositories/business_metric.py`` (``get_by_region_paged``,
        ``get_distinct_values``): advances by the rows ACTUALLY returned and stops
        on an EMPTY page — correct under any PostgREST ``db-max-rows`` cap.

        Parameters
        ----------
        db:
            An async supabase-py client (``AsyncClient`` from
            ``src.memory.services.factories.get_async_supabase_client``).
        splits:
            Optional list of ``data_split`` values to include (e.g. ``["train"]``
            or ``["holdout"]``).  ``None`` → no split filter (all rows for the
            brand).
        before_month:
            Optional ISO-8601 date string (e.g. ``"2026-05-01"``).  When set,
            adds a ``journey_start_date < before_month`` predicate so walk-forward
            windows can load the training prefix.

        Returns
        -------
        pandas.DataFrame:
            Concatenated rows with at least:
            ``patient_id``, ``journey_start_date``, ``data_split``,
            ``self.spec.label_column``, and the columns in ``KEEP_COLUMNS``.
        """
        # Build the SELECT column list: always-select + label + raw feature cols.
        select_cols = set(_ALWAYS_SELECT)
        select_cols.add(self.spec.label_column)
        select_cols.update(self.keep_columns)
        select_expr = ",".join(sorted(select_cols))

        all_rows: list[dict[str, Any]] = []
        exhausted = False
        offset = 0

        for _page in range(_MAX_PAGES):
            query = (
                db.table("patient_journeys")
                .select(select_expr)
                .eq("brand", self.spec.brand)
                # Gold-standard eval REQUIRES synthetic rows (opt-in explicit):
                # patient_journeys.is_synthetic=True is the provenance flag for
                # the synthetic cohort used in gold-standard evaluation.
                .eq("is_synthetic", True)
            )

            if splits is not None:
                # PostgREST .in_() for list membership.
                query = query.in_("data_split", list(splits))

            if before_month is not None:
                query = query.lt("journey_start_date", before_month)

            # PK-ordered range window — cap-agnostic, deterministic under concurrency.
            query = query.order("patient_id").range(offset, offset + _PAGE_SIZE - 1)

            result = await query.execute()
            page_rows: list[dict[str, Any]] = result.data or []

            if not page_rows:
                exhausted = True
                break

            all_rows.extend(page_rows)
            offset += len(page_rows)

        if not exhausted:
            logger.warning(
                "FeatureBuilder.load_frame hit the max_pages=%d runaway guard for "
                "brand=%s splits=%s before_month=%s; rows beyond page %d are omitted.",
                _MAX_PAGES,
                self.spec.brand,
                splits,
                before_month,
                _MAX_PAGES,
            )

        if not all_rows:
            return pd.DataFrame()

        return pd.DataFrame(all_rows)

    async def build_for_split(
        self,
        db: Any,
        split: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load patient_journeys rows for *split* and FIT-encode via build_from_frame.

        Convenience wrapper: calls :meth:`load_frame` with ``splits=[split]`` then
        passes the result to :meth:`build_from_frame` (FIT path — learns medians
        and ``feature_columns`` from this frame).

        Walk-forward callers (Task 8) should call :meth:`load_frame` directly once
        and alternate :meth:`build_from_frame` / :meth:`transform` per month window
        themselves — this method is the simple single-split entry point.

        Parameters
        ----------
        db:
            Async Supabase client (see :meth:`load_frame`).
        split:
            ``data_split`` value, e.g. ``"train"`` or ``"holdout"``.

        Returns
        -------
        (X, y):
            Encoded feature matrix and label series (same contract as
            :meth:`build_from_frame`).

        Raises
        ------
        ValueError:
            If no rows are found for the given split.
        """
        frame = await self.load_frame(db, splits=[split])
        if frame.empty:
            raise ValueError(
                f"FeatureBuilder.build_for_split: no rows found for "
                f"brand={self.spec.brand!r} split={split!r} is_synthetic=True"
            )
        return self.build_from_frame(frame)
