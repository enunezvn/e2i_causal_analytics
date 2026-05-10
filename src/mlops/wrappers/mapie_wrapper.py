"""MAPIE conformal wrapper with honest split / cross-conformal selection.

Plan v3 §4 T2.1 (3h) — MAPIE training-set conformal honesty fix.

Pre-fix behaviour (cycle-9 codex F1 / D1 known-defect): the wrapper
hardcoded MAPIE ``cv="prefit"`` and pre-fit the base on the FULL ``(X, y)``
— then calibrated MAPIE on that SAME ``(X, y)``. This is "training-set
conformal" — the marginal-coverage guarantee is NOT honoured for new data
because the calibration set was seen during base estimator training.

Post-fix behaviour:
  * **n_minority ≥ MIN_HONEST_SPLIT_N** (default 50): honest split-conformal.
    Hold out a stratified ``calib_fraction`` (default 0.20) of ``(X, y)``,
    fit base on the train fold, calibrate MAPIE prefit on the held-out
    calib fold. The 50-positive floor follows Vovk 2005 / Lei et al. 2018:
    below ~50 calibration positives the empirical-CDF inversion is too
    noisy for marginal-coverage to track the requested 1-α.
  * **n_minority < MIN_HONEST_SPLIT_N**: K-fold cross-conformal via MAPIE's
    native ``cv=int`` mode. Base is re-fit ``cv`` times on stratified
    K-1/K splits; the union of out-of-fold residuals provides the
    conformity scores. This is the right small-sample alternative
    because it preserves the coverage guarantee without sacrificing
    further calibration data.

Both modes return ``predict_proba`` from the BASE estimator (not MAPIE's
sets). The conformal contribution is the ``predict_sets`` artifact + the
honest coverage guarantee on it.

Plan §6 T2.1 acceptance:
  * split-conformal coverage ∈ [0.85, 0.95] for α=0.10 on n_minority ≥ 50;
  * cross-conformal switch tested at n_minority < 50.

Reference: shard 19 §B of ``.claude/plans/adaptive_criteria_v3_followup/``;
plan v3 ``.claude/plans/adaptive_disease_agnostic_quality_uplift.md`` §4 T2.1.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# Plan v3 §4 T2.1: minimum n_minority count for honest split-conformal.
# Below this, cross-conformal (K-fold) is the statistically-honest fallback.
# 50 is the Vovk / Lei threshold below which empirical-CDF inversion noise
# makes marginal-coverage drift outside the requested band.
MIN_HONEST_SPLIT_N: int = 50

# Default stratified hold-out fraction for split-conformal calibration.
DEFAULT_CALIB_FRACTION: float = 0.20

# Default cross-conformal fold count when n_minority < MIN_HONEST_SPLIT_N.
# 5 folds preserves marginal-coverage validity at low N without exhausting
# the per-fold positive count.
DEFAULT_CROSS_CONFORMAL_FOLDS: int = 5

ConformalMode = Literal["split", "cross", "prefit_legacy"]


class MapieConformalBinaryClassifier:
    """Cross-conformal calibration wrapper for binary classifiers.

    The wrapped estimator must implement sklearn-compatible fit / predict_proba.
    `method='lac'` (Least Ambiguous set-valued Classifier; Sadinle 2019) is the
    MAPIE 0.8+ default and produces coverage-guaranteed prediction SETS (NOT
    recalibrated softmax probabilities — see amendment 2 below; predict_proba
    delegates to the base estimator).

    Phase 1 alpha is a single value (0.10) — 90% marginal coverage. For
    multi-alpha reporting (80/90/95% bands), iterate alpha at evaluation time;
    this wrapper fits once and exposes the configured alpha via `predict_sets`.

    Plan v3 §4 T2.1 amendment: ``cv="prefit"`` (the training-set conformal
    that was the legacy default) is no longer the default. ``fit()`` now
    selects the conformal mode based on the minority-class count of ``y``:

      * n_minority ≥ ``MIN_HONEST_SPLIT_N`` → split-conformal (honest hold-out)
      * n_minority < ``MIN_HONEST_SPLIT_N`` → cross-conformal (K-fold)

    Pass ``conformal_mode="prefit_legacy"`` to opt back into the legacy
    training-set behavior (NOT recommended; preserved for back-compat with
    callers that pre-validated against pre-T2.1 outputs).

    Amendments vs shard 19 §B.2 verbatim:
      1. `method="cv"` → `method="lac"`. MAPIE pre-0.7 accepted `"cv"`; MAPIE
         0.8.6 (pinned) requires one of `{naive, score, lac, cumulated_score,
         aps, top_k, raps}`. `lac` is MAPIE's own default and the most direct
         replacement.
      2. `MapieClassifier` in MAPIE 0.8.6 does NOT expose `predict_proba` —
         only `predict(X, alpha=...) → (y_pred, prediction_sets)`. The
         wrapper's `predict_proba` therefore delegates to the BASE estimator
         (the original probabilities). The Phase-1 conformal contribution is
         the prediction-set artifact (`predict_sets`), which is logged for
         future work but not consumed by the v3 contract. Re-evaluate at W4
         multi-disease results: if the non-calibration-native base entries
         (LightGBM_Conformal, LogisticRegression_Conformal) underperform on
         calibration metrics, revisit the `skip_post_hoc_calibration=True`
         registry default for those entries.
      3. Plan v3 §4 T2.1: legacy `cv="prefit"` mode IS still supported via
         ``conformal_mode="prefit_legacy"`` but is no longer the default.
         The default switch (split vs cross) restores the marginal-coverage
         guarantee that the prefit-on-full-data path silently broke.
    """

    def __init__(
        self,
        base_estimator: Any,
        method: str = "lac",
        cv: int = 5,
        alpha: float = 0.10,
        random_state: int = 42,
        conformal_mode: Optional[ConformalMode] = None,
        calib_fraction: float = DEFAULT_CALIB_FRACTION,
        min_honest_split_n: int = MIN_HONEST_SPLIT_N,
    ) -> None:
        if not 0.0 < calib_fraction < 1.0:
            raise ValueError(f"calib_fraction must be in (0, 1); got {calib_fraction}")
        if min_honest_split_n < 1:
            raise ValueError(f"min_honest_split_n must be >= 1; got {min_honest_split_n}")
        if conformal_mode is not None and conformal_mode not in (
            "split",
            "cross",
            "prefit_legacy",
        ):
            raise ValueError(
                f"conformal_mode={conformal_mode!r} not in "
                "{'split', 'cross', 'prefit_legacy', None}"
            )
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.alpha = alpha
        self.random_state = random_state
        self.conformal_mode = conformal_mode
        self.calib_fraction = calib_fraction
        self.min_honest_split_n = min_honest_split_n
        # Populated by fit().
        self._mapie: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None
        self.fitted_conformal_mode_: Optional[ConformalMode] = None
        self.n_minority_: Optional[int] = None
        self.n_calib_: Optional[int] = None

    def _select_mode(self, y: np.ndarray) -> ConformalMode:
        """Choose conformal mode from the minority count of y, unless the
        caller forced one via ``conformal_mode``."""
        if self.conformal_mode is not None:
            return self.conformal_mode
        n_minority = int(min(np.sum(y == 0), np.sum(y == 1)))
        self.n_minority_ = n_minority
        if n_minority >= self.min_honest_split_n:
            return "split"
        return "cross"

    def fit(self, X: Any, y: Any) -> "MapieConformalBinaryClassifier":
        """Fit base + MAPIE under the selected conformal mode.

        Plan v3 §4 T2.1 routes to one of three implementations:

          * ``split``: stratified hold-out (1 - calib_fraction) for base
            fit, calib_fraction for MAPIE prefit. Honest marginal coverage.
          * ``cross``: MAPIE ``cv=K`` re-fits base K times. Honest marginal
            coverage, more expensive (K-1 fits per `cv`).
          * ``prefit_legacy``: legacy training-set conformal. Pre-T2.1
            behavior; marginal-coverage guarantee NOT honoured.
        """
        from mapie.classification import MapieClassifier

        y_arr = np.asarray(y)
        # Track minority count even when caller forced a mode.
        if self.n_minority_ is None:
            self.n_minority_ = int(min(np.sum(y_arr == 0), np.sum(y_arr == 1)))
        mode = self._select_mode(y_arr)
        self.fitted_conformal_mode_ = mode

        if mode == "prefit_legacy":
            # MAPIE adversarial-review MEDIUM-2: warn callers that prefit_legacy
            # uses training-set conformal calibration, which breaks the marginal-
            # coverage guarantee. Preserved for back-compat only; use with caution.
            warnings.warn(
                "MapieConformalBinaryClassifier: conformal_mode='prefit_legacy' "
                "calibrates MAPIE on the SAME data used to fit the base estimator. "
                "The marginal-coverage guarantee (plan §6 T2.1) is NOT honoured in "
                "this mode. Use conformal_mode=None (auto) or 'split'/'cross' for "
                "statistically-honest conformal intervals. prefit_legacy is preserved "
                "for back-compat with pre-T2.1 outputs only.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.base_estimator.fit(X, y_arr)
            self._mapie = MapieClassifier(
                estimator=self.base_estimator,
                method=self.method,
                cv="prefit",
                random_state=self.random_state,
            )
            self._mapie.fit(X, y_arr)
            self.n_calib_ = int(len(y_arr))
            logger.info(
                "MAPIE conformal: legacy prefit mode (n_minority=%d, n_calib=%d)",
                self.n_minority_,
                self.n_calib_,
            )
        elif mode == "split":
            # MAPIE adversarial-review MEDIUM-1: sklearn stratified split
            # raises ValueError when n_minority is too small to stratify
            # (e.g. n_minority=1 → "least populated class has only 1 member").
            # Re-raise with an actionable message rather than letting the raw
            # sklearn error propagate without context.
            try:
                X_train, X_calib, y_train, y_calib = train_test_split(
                    X,
                    y_arr,
                    test_size=self.calib_fraction,
                    random_state=self.random_state,
                    stratify=y_arr,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Stratified calibration split failed for split-conformal mode "
                    f"(n_minority={self.n_minority_}, calib_fraction={self.calib_fraction}). "
                    "sklearn requires ≥ 2 samples per class in each split. "
                    "Options: (1) lower calib_fraction so fewer minority samples end up in calib, "
                    "(2) use conformal_mode='cross' for small-sample datasets, "
                    "(3) collect more minority-class observations."
                ) from exc
            self.base_estimator.fit(X_train, y_train)
            self._mapie = MapieClassifier(
                estimator=self.base_estimator,
                method=self.method,
                cv="prefit",
                random_state=self.random_state,
            )
            self._mapie.fit(X_calib, y_calib)
            self.n_calib_ = int(len(y_calib))
            logger.info(
                "MAPIE conformal: split-honest mode (n_minority=%d, "
                "n_calib=%d, calib_fraction=%.2f)",
                self.n_minority_,
                self.n_calib_,
                self.calib_fraction,
            )
        else:  # mode == "cross"
            cross_folds = max(2, min(self.cv, DEFAULT_CROSS_CONFORMAL_FOLDS))
            # MAPIE will internally fit base `cross_folds` times — pass a
            # CLONE so the user's `base_estimator` reference is not mutated
            # by any fold's partial fit (e.g., warm-start incompat).
            self._mapie = MapieClassifier(
                estimator=clone(self.base_estimator),
                method=self.method,
                cv=cross_folds,
                random_state=self.random_state,
            )
            self._mapie.fit(X, y_arr)
            # Refit the wrapper's base estimator on full (X, y) so
            # predict_proba reflects the model the user wrapped — this is
            # NOT used for MAPIE calibration (MAPIE has its own K-fold
            # estimators internally), only for downstream proba reporting.
            self.base_estimator.fit(X, y_arr)
            self.n_calib_ = int(len(y_arr))
            logger.info(
                "MAPIE conformal: cross-conformal mode (n_minority=%d, n_calib=%d, cv=%d)",
                self.n_minority_,
                self.n_calib_,
                cross_folds,
            )

        self.classes_ = np.array(getattr(self.base_estimator, "classes_", np.unique(y_arr)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return base-estimator probabilities (amendment 2 — see class docstring).

        MAPIE 0.8.6 doesn't expose `predict_proba` on `MapieClassifier`, so the
        wrapper delegates to the wrapped base estimator. The conformal-set
        artifact lives on `predict_sets`.
        """
        if self._mapie is None:
            raise RuntimeError("MapieConformalBinaryClassifier called before fit()")
        raw = np.asarray(self.base_estimator.predict_proba(X))
        if raw.ndim == 1:
            return np.column_stack([1.0 - raw, raw])
        if raw.ndim == 2 and raw.shape[1] == 1:
            p1 = raw[:, 0]
            return np.column_stack([1.0 - p1, p1])
        return raw

    def predict_sets(self, X: Any) -> Any:
        """Return prediction sets at the configured alpha.

        Phase 1: not consumed by the v3 contract; logged as future-work artifact.
        Plan §6 T2.1 acceptance: marginal coverage on these sets must be in
        [0.85, 0.95] at α=0.10 when ``fit()`` ran in ``"split"`` mode on
        ``n_minority ≥ MIN_HONEST_SPLIT_N``.
        """
        if self._mapie is None:
            raise RuntimeError("MapieConformalBinaryClassifier called before predict_sets()")
        _, sets = self._mapie.predict(X, alpha=self.alpha)
        return sets
